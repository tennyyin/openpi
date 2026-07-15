"""Convert the TRI/LBM BimanualBikeRotorInstall teleop demos to a LeRobot dataset for openpi.

Raw layout (a TRI/LBM task under cv_unified/videos)::

    <root>/BimanualBikeRotorInstall/<station>/real/teleop/<batch>/episode_<N>/
        rgb/{scene_left_0,scene_right_0,wrist_left_plus,wrist_right_plus,
             wrist_left_minus,wrist_right_minus}.mp4     # 6 cameras (variable native res)
        lowdim/<cam>.npz    # shared robot state/action dict (identical across the 6 npz)
        metadata.json       # language.prompt, num_frames, specific.metadata.is_successful, ...

This is a **bimanual dual-Panda** (7-DoF x 2) setup at 10 fps. We keep only the 534
`teleop` (human) demonstrations -- the `rollout` episodes are policy-eval trajectories
(many unsuccessful) and are excluded from behavior-cloning fine-tuning.

We store three of the six views, mapped to pi0's three image slots -- matching the exact
views the tri_bike world model trained on:

    base_0_rgb        <- scene_right_0
    left_wrist_0_rgb  <- wrist_left_plus
    right_wrist_0_rgb <- wrist_right_plus

State (proprioceptive conditioning) is the *measured* joint state (16-d)::

    observation.state = actual joint_position_left(7) + joint_position_right(7)
                        + gripper_left(1) + gripper_right(1)

Action is the native LBM ``xyzrot6g`` command (20-d, absolute end-effector targets)::

    actions = packed 'action' field = per arm [xyz(3) + rot_6d(6) + gripper(1)] x2

Normalization is NOT done here -- openpi computes q01/q99 + mean/std with
`scripts/compute_norm_stats.py` after this conversion (see examples/bike_rotor/README.md).

Usage (run inside the openpi uv env)::

    uv run examples/bike_rotor/convert_bike_rotor_to_lerobot.py \
        --raw-root /home/vguizilini/workspace/data/predict2/data/cv_unified/videos/LBM \
        --repo-id tri/bike_rotor_cartesian
    # smoke test on a few episodes:
    uv run examples/bike_rotor/convert_bike_rotor_to_lerobot.py --repo-id tri/bike_rotor_smoke --limit 3
"""

import dataclasses
import glob
import json
import os
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

TASK = "BimanualBikeRotorInstall"
DEFAULT_RAW_ROOT = "/home/vguizilini/workspace/data/predict2/data/cv_unified/videos/LBM"

# Camera view -> pi0 image slot. These three views match the tri_bike world model.
CAMERA_MAP = {
    "base": "scene_right_0",
    "left_wrist": "wrist_left_plus",
    "right_wrist": "wrist_right_plus",
}

# 16-d joint state (measured / "actual"), layout: jpos_left, jpos_right, grip_left, grip_right.
STATE_KEYS = [
    "robot__actual__joint_position__left::panda",   # 7
    "robot__actual__joint_position__right::panda",  # 7
    "robot__actual__grippers__left::panda_hand",    # 1
    "robot__actual__grippers__right::panda_hand",   # 1
]

RESIZE_H, RESIZE_W = 224, 224  # pi0 resizes images to 224 anyway; store at model res.


@dataclasses.dataclass
class Args:
    repo_id: str
    raw_root: str = DEFAULT_RAW_ROOT
    limit: int = 0                     # cap #episodes (0 = all); for smoke tests
    teleop_only: bool = True           # exclude policy-eval rollout episodes
    successful_only: bool = False      # keep only metadata is_successful == True
    fps: int = 10
    push_to_hub: bool = False
    # Parallel sharding: run N processes with shard_id=0..N-1 over disjoint episode
    # subsets (each writes <repo_id>_shardXXXofNNN), then merge_bike_rotor_shards.py.
    num_shards: int = 1
    shard_id: int = 0
    # Video encode codec. h264 is ~3-5x faster than lerobot's default libsvtav1.
    vcodec: str = "h264"


def _patch_encoder(vcodec: str):
    """lerobot hardcodes libsvtav1 in save_episode's encode path; inject a faster codec."""
    import lerobot.common.datasets.lerobot_dataset as lrd

    orig = lrd.encode_video_frames

    def fast(imgs_dir, video_path, fps, **kw):
        kw.setdefault("vcodec", vcodec)
        return orig(imgs_dir, video_path, fps, **kw)

    lrd.encode_video_frames = fast


def shard_repo_id(repo_id: str, shard_id: int, num_shards: int) -> str:
    return repo_id if num_shards <= 1 else f"{repo_id}_shard{shard_id:03d}of{num_shards:03d}"


def list_teleop_episodes(raw_root: str, *, teleop_only: bool):
    modes = ("teleop",) if teleop_only else ("teleop", "rollout")
    eps = []
    for mode in modes:
        eps += sorted(glob.glob(os.path.join(raw_root, TASK, "*", "real", mode, "*", "episode_*")))
    return [e for e in eps if os.path.isdir(e)]


def read_frames(video_path: str) -> np.ndarray:
    """Decode + stretch-resize to (RESIZE_H, RESIZE_W). Returns uint8 [T, H, W, 3] RGB."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        cap.release()
    return np.asarray(frames, dtype=np.uint8)


def load_lowdim(ep_dir: str):
    """Return (state[T,16], actions[T,20]) as float32, both raw (unnormalized)."""
    npzs = sorted(glob.glob(os.path.join(ep_dir, "lowdim", "*.npz")))
    if not npzs:
        raise FileNotFoundError(f"{ep_dir}: no lowdim npz")
    a = np.load(npzs[0], allow_pickle=True)["action"].item()
    state = np.concatenate([np.asarray(a[k], dtype=np.float32) for k in STATE_KEYS], axis=1)  # [T,16]
    actions = np.asarray(a["action"], dtype=np.float32)                                        # [T,20]
    return state, actions


def episode_prompt(ep_dir: str) -> str:
    md = json.load(open(os.path.join(ep_dir, "metadata.json")))
    pr = md.get("language", {}).get("prompt")
    if isinstance(pr, list) and pr:
        return str(pr[0])
    if isinstance(pr, str) and pr:
        return pr
    return str(md.get("language", {}).get("task", TASK))


def is_successful(ep_dir: str) -> bool:
    md = json.load(open(os.path.join(ep_dir, "metadata.json")))
    return bool(md.get("specific", {}).get("metadata", {}).get("is_successful", False))


def main(args: Args):
    if not (0 <= args.shard_id < max(1, args.num_shards)):
        raise SystemExit(f"--shard-id {args.shard_id} out of range for --num-shards {args.num_shards}")
    _patch_encoder(args.vcodec)

    repo_id = shard_repo_id(args.repo_id, args.shard_id, args.num_shards)
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    img_feat = {"dtype": "video", "shape": (RESIZE_H, RESIZE_W, 3), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="bimanual_panda",
        fps=args.fps,
        features={
            "observation.images.base": img_feat,
            "observation.images.left_wrist": img_feat,
            "observation.images.right_wrist": img_feat,
            "observation.state": {"dtype": "float32", "shape": (16,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (20,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    eps = list_teleop_episodes(args.raw_root, teleop_only=args.teleop_only)
    if args.successful_only:
        eps = [e for e in eps if is_successful(e)]
    if args.limit:
        eps = eps[: args.limit]
    if args.num_shards > 1:
        eps = eps[args.shard_id :: args.num_shards]
    print(f"[shard {args.shard_id}/{args.num_shards}] converting {len(eps)} episodes -> {output_path} (vcodec={args.vcodec})")

    written = skipped = 0
    for ep_dir in tqdm.tqdm(eps):
        try:
            state, actions = load_lowdim(ep_dir)
            base = read_frames(os.path.join(ep_dir, "rgb", f"{CAMERA_MAP['base']}.mp4"))
            lw = read_frames(os.path.join(ep_dir, "rgb", f"{CAMERA_MAP['left_wrist']}.mp4"))
            rw = read_frames(os.path.join(ep_dir, "rgb", f"{CAMERA_MAP['right_wrist']}.mp4"))
            T = min(len(base), len(lw), len(rw), state.shape[0], actions.shape[0])
            if T < 2:
                skipped += 1
                continue
            prompt = episode_prompt(ep_dir)
            for t in range(T):
                dataset.add_frame(
                    {
                        "observation.images.base": base[t],
                        "observation.images.left_wrist": lw[t],
                        "observation.images.right_wrist": rw[t],
                        "observation.state": state[t],
                        "actions": actions[t],
                        "task": prompt,
                    }
                )
            dataset.save_episode()
            written += 1
        except Exception as ex:  # noqa: BLE001
            skipped += 1
            print(f"SKIP {ep_dir}: {type(ex).__name__}: {ex}", flush=True)

    print(f"Done: {written} episodes written, {skipped} skipped. Dataset at {output_path}")

    if args.push_to_hub:
        dataset.push_to_hub(tags=["tri", "lbm", "bimanual", "panda"], private=True, push_videos=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
