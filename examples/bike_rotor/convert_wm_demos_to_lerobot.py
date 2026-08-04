"""Convert world-model teleop demos into a LeRobot dataset for openpi fine-tuning.

These demos are collected INSIDE the tri_bike AR world model by driving it with two
SpaceMice (open-world ``scripts/interactive_ar.py --record-dir``). Each seeded scene
is saved as its own ``demo_XXXX/`` subdirectory:

    <record-root>/demo_0000/
        actions_raw.npy    # [num_blocks, 20]  de-normalized xyzrot6g EEF command
        state_raw.npy      # [num_blocks, 16]  de-normalized joint proprioception (aux head)
        view_base.mp4       # dreamed scene_right_0  view (base_0_rgb slot)
        view_left_wrist.mp4  # dreamed wrist_left_plus  view (left_wrist_0_rgb slot)
        view_right_wrist.mp4 # dreamed wrist_right_plus view (right_wrist_0_rgb slot)
        meta.json          # view_names, prompt, frames_per_block, rgb_frames_per_block, ...
    <record-root>/demo_0001/ ...

Output is the SAME LeRobot schema as convert_bike_rotor_to_lerobot.py (the real TRI
demos), so a converted WM-demo dataset drops into the ``pi0_bike_rotor`` /
``pi05_bike_rotor`` configs unchanged:

    observation.images.base / left_wrist / right_wrist   video (224, 224, 3)
    observation.state                                     float32 (16,)
    actions                                               float32 (20,)
    task                                                  language prompt

RATES -- why this script RESAMPLES instead of emitting one row per block.
The world model logs one (action, state) per AR BLOCK, and a block is
``rgb_frames_per_block`` RGB frames, so the low-dim stream is at
``block_hz = data_fps / rgb_frames_per_block`` -- 2.5 Hz for the shipped 10 fps
tri_bike checkpoint, 1.25 Hz for a 5 fps one. The REAL TRI demos this dataset is mixed
with are 10 Hz. Emitting one row per block while declaring ``fps=10`` (what this script
used to do) therefore wrote a dataset whose timeline was 4x too slow -- the frames were
right, the declared rate was right, and every velocity implied by consecutive rows was
wrong by exactly the block/target ratio. Nothing downstream can notice.

So each demo is resampled onto a fixed ``RECORD_TARGET_HZ`` (10 Hz) grid using the
timing meta the recorder writes (``data_fps``, ``block_hz``, ``record_target_hz``):
low-dim is interpolated in time (rot6d via a real SO(3) slerp -- component-wise lerp
would leave a non-orthonormal frame that Gram-Schmidt silently reads as some other
rotation), and each row takes the RGB frame nearest its timestamp. A 10 fps and a 5 fps
checkpoint's demos therefore land on the SAME 10 Hz timeline and are directly mixable.
``--resample hold`` keeps the commanded pose piecewise-constant instead (truthful to
what the operator held, but it makes 3 of every 4 rows a repeat).

ACTION MODE -- absolute cartesian only. ``actions`` here are absolute 20-d bimanual EEF
poses. A demo whose ``meta.json`` declares a joint, delta, or action-free
(``action_cond_mode="none"``) mode is SKIPPED, not converted: each of those would produce
a dataset with correct shapes and wrong meaning (see ``REQUIRED_ACTION_MODE``). This
mirrors open-world's own gate, and the check is possible only because the recorder writes
the mode into meta.

Normalization is NOT done here -- openpi computes q01/q99 + mean/std afterwards
(examples/bike_rotor/compute_norm_stats_from_raw.py or scripts/compute_norm_stats.py).

Usage (run inside the openpi uv env)::

    uv run examples/bike_rotor/convert_wm_demos_to_lerobot.py \
        --record-root /path/to/open-world/runs/wm_demos \
        --repo-id tri/bike_rotor_wm
    # smoke test on a few demos:
    uv run examples/bike_rotor/convert_wm_demos_to_lerobot.py \
        --record-root ... --repo-id tri/bike_rotor_wm_smoke --limit 3
"""

import dataclasses
import glob
import json
import os
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import tyro

# WM demo view name (meta.json / view_<name>.mp4) -> pi0 image slot. The tri_bike
# world model's 3 views were named to match the openpi bike_rotor slots, so this is
# identity; kept explicit so a differently-named embodiment can be remapped here.
VIEW_TO_SLOT = {
    "base": "observation.images.base",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
STATE_DIM = 16
ACTION_DIM = 20
RESIZE_H, RESIZE_W = 224, 224  # pi0 model resolution (matches convert_bike_rotor_to_lerobot.py)

# Fixed output rate. Matches the REAL TRI bike-rotor demos (10 Hz), so WM demos from a
# 10 fps and a 5 fps checkpoint both land here and are mixable with each other and with
# real data. Mirrors open-world's openworld/autoregressive/data/rates.RECORD_TARGET_HZ.
RECORD_TARGET_HZ = 10.0

# Rotation slices of the TRI 20-d xyzrot6g bimanual action vector, which packs
#   [R_xyz(0:3), R_rot6d(3:9), L_xyz(9:12), L_rot6d(12:18), gripR(18), gripL(19)]
# -- right arm first, BOTH grippers appended at the end. rot6d is the first two COLUMNS
# of the rotation matrix. These slices must be interpolated on SO(3), never component-
# wise: a lerp of two rot6d vectors is not orthonormal, and the Gram-Schmidt every
# consumer applies then silently reads it as a DIFFERENT rotation than either endpoint.
ACTION_ROT6D_SLICES = ((3, 9), (12, 18))

# The ONLY action mode this converter can read: absolute cartesian EEF poses fed to the
# world model through a live cross-attn/adaln path. open-world's teleop refuses to launch
# in any other mode (interactive_ar.require_cartesian_absolute), and the recorder writes
# these three fields into meta.json precisely so this side can VERIFY rather than assume.
# Each wrong mode is a silent corruption here, not a crash:
#   * action_space="joint_pos" -> `actions` are joint targets, but they land in the 20-d
#     cartesian slot and get slerped as if dims 3:9 were rot6d columns.
#   * action_delta=True        -> `actions` are per-block INCREMENTS; interpolating them
#     in time is meaningless, and pi0 would be trained to predict absolute poses from a
#     delta stream.
#   * action_cond_mode="none"  -> the model's action path is a zero context, so the video
#     was NOT driven by these commands. Shapes and rates are perfect; the demo is a
#     recording of two unrelated streams.
REQUIRED_ACTION_MODE = {
    "action_space": ("cartesian",),
    "action_delta": (False,),
    "action_cond_mode": ("cross_attn", "cross_attn_pe", "cross_attn_aligned",
                         "adaln", "adaln_aligned"),
}


def require_absolute_cartesian(meta: dict) -> None:
    """Raise unless ``meta`` declares the absolute-cartesian mode (see REQUIRED_ACTION_MODE).

    A field the recorder did not write is accepted: pre-gate demos predate these keys and
    could only have been collected in this mode (the alternatives had no driving path)."""
    for field, accepted in REQUIRED_ACTION_MODE.items():
        got = meta.get(field, accepted[0])
        if got is None or got in accepted:
            continue
        raise ValueError(
            f"demo meta declares {field}={got!r}; this converter reads ABSOLUTE CARTESIAN "
            f"demos only (accepted: {', '.join(repr(v) for v in accepted)}). Re-record with "
            f"an absolute-cartesian checkpoint -- converting this one would emit a dataset "
            f"with the right shapes and the wrong meaning")


@dataclasses.dataclass
class Args:
    repo_id: str
    record_root: str                  # dir holding demo_XXXX/ subdirectories
    limit: int = 0                    # cap #demos (0 = all); for smoke tests
    prompt: str = ""                  # override the per-demo prompt (else meta.json's)
    # Output rate, in Hz, for EVERY demo regardless of the checkpoint's own rate. Rows
    # are resampled onto this grid (see module docstring); it is not a relabel.
    target_fps: int = int(RECORD_TARGET_HZ)
    resample: str = "interp"          # "interp" (time-interpolate, rot6d via SO(3)) | "hold"
    block_hz: float = 0.0             # override the demo's meta block rate (0 = trust meta)
    min_frames: int = 2               # skip demos with fewer than this many OUTPUT rows
    min_blocks: int = 2               # skip demos with fewer than this many usable blocks
    push_to_hub: bool = False
    vcodec: str = "h264"              # faster than lerobot's default libsvtav1


def _patch_encoder(vcodec: str):
    """lerobot hardcodes libsvtav1 in save_episode's encode path; inject a faster codec."""
    import lerobot.common.datasets.lerobot_dataset as lrd

    orig = lrd.encode_video_frames

    def fast(imgs_dir, video_path, fps, **kw):
        kw.setdefault("vcodec", vcodec)
        return orig(imgs_dir, video_path, fps, **kw)

    lrd.encode_video_frames = fast


def list_demos(record_root: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(record_root, "demo_*")) if os.path.isdir(d))


def block_frame_index(block: int, rgb_per_block: int, n_frames: int) -> int:
    """RGB frame index paired with AR ``block``: the CENTER frame of that block.

    The recorder emits ``rgb_per_block`` frames per block contiguously, so block b
    spans ``[b*rgb_per_block, (b+1)*rgb_per_block)``; we take its midpoint (clamped to
    the frames actually written, e.g. if --record-max-frames truncated the tail)."""
    idx = block * rgb_per_block + rgb_per_block // 2
    return min(idx, n_frames - 1)


def align_blocks(num_blocks: int, rgb_per_block: int, n_frames: int) -> list[int]:
    """Per-block RGB frame indices (one row per block), dropping blocks whose frames
    were truncated away (all map to the last frame once we run past n_frames).

    This is the ``target_hz == block_hz`` case of :func:`resample_indices`, which is what
    the conversion actually uses; kept because it defines the pairing convention in its
    simplest form and pins it in the tests."""
    if n_frames <= 0:
        return []
    idxs = []
    for b in range(num_blocks):
        start = b * rgb_per_block
        if start >= n_frames:               # this block's frames were never written
            break
        idxs.append(block_frame_index(b, rgb_per_block, n_frames))
    return idxs


def demo_rates(meta: dict, block_hz_override: float = 0.0) -> tuple[float, float, int]:
    """``meta.json`` -> (data_fps, block_hz, rgb_per_block), cross-checked.

    ``data_fps`` is the true rate of the ``view_*.mp4`` files (the latent root's stored
    RGB rate: 10 for the shipped tri_bike checkpoint, 5 for a 5 fps one) and
    ``block_hz = data_fps / rgb_per_block`` is the rate of the action/state rows. When
    meta states both, they must agree -- they come from the same resolver in open-world,
    so disagreement means a hand-edited or stitched meta and every timestamp below would
    be wrong with no other symptom.

    Raises ValueError if the video rate cannot be established at all: guessing it silently
    rescales the whole episode."""
    rgb_per_block = int(meta.get("rgb_frames_per_block")
                        or int(meta.get("frames_per_block", 1)) * 4)
    if rgb_per_block < 1:
        raise ValueError(f"rgb_frames_per_block={rgb_per_block} (need >= 1)")

    data_fps = meta.get("data_fps")
    if not data_fps:
        # Pre-rates demos: the recorder used to write only `fps`, and on old sessions that
        # was the MJPEG PREVIEW rate rather than the root's stored rate -- so this fallback
        # can itself be wrong. Warn rather than fail; there is nothing better on disk.
        data_fps = meta.get("video_fps") or meta.get("fps")
        if not data_fps:
            raise ValueError("meta has no data_fps/video_fps/fps: cannot know the video's "
                             "true rate, so rows cannot be placed on a timeline")
        print(f"WARN: meta has no 'data_fps'; assuming the video is {float(data_fps):g} fps "
              f"from meta['fps'] (old demo: this may be the preview rate, not the "
              f"latent root's rate)", flush=True)
    data_fps = float(data_fps)

    block_hz = float(block_hz_override) if block_hz_override else float(
        meta.get("block_hz") or 0.0) or data_fps / rgb_per_block
    derived = data_fps / rgb_per_block
    if not block_hz_override and abs(block_hz - derived) > 1e-6 * max(1.0, derived):
        raise ValueError(f"meta block_hz={block_hz:g} disagrees with data_fps/"
                         f"rgb_frames_per_block={data_fps:g}/{rgb_per_block}={derived:g}; "
                         f"one of them is wrong and both timestamps and frame pairing "
                         f"depend on it (pass --block-hz to override deliberately)")
    if block_hz <= 0:
        raise ValueError(f"block_hz={block_hz:g} (need > 0)")
    return data_fps, block_hz, rgb_per_block


def resample_indices(num_blocks: int, rgb_per_block: int, n_frames: int,
                     block_hz: float, data_fps: float,
                     target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Output-grid sampling plan: (block_coords[N] float, frame_idx[N] int).

    Timeline convention (same as :func:`block_frame_index`): block b is timestamped at
    the CENTER of its own RGB frames, ``t_b = (b + 0.5) / block_hz``. Output row j sits
    at ``t_j = t_0 + j / target_hz``, so

        block_coords[j] = j * block_hz / target_hz     (fractional -> interpolate)
        frame_idx[j]    = floor(rgb_per_block/2 + j * data_fps / target_hz + 0.5)

    Rounding is half-UP, not numpy's half-to-even: at a 0.5 frame/row cadence (a 5 fps
    root upsampled to 10 Hz) banker's rounding turns the sequence 2.0, 2.5, 3.0, 3.5 into
    2, 2, 3, 4 -- a visibly stuttering frame advance -- where half-up gives 2, 3, 3, 4.

    At ``target_hz == block_hz`` this reduces EXACTLY to one row per block paired with
    its centre frame, i.e. the old behaviour. N is chosen so neither stream is
    EXTRAPOLATED: it stops at the last block centre, and at the last frame actually
    written (``--record-max-frames`` truncates video while keeping every action, and
    running past it would emit rows that all repeat the final frame)."""
    if num_blocks <= 0 or n_frames <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    step_blocks = block_hz / target_hz          # blocks advanced per output row
    step_frames = data_fps / target_hz          # frames advanced per output row
    n_from_blocks = int(np.floor((num_blocks - 1) / step_blocks + 1e-9)) + 1
    first_frame = rgb_per_block / 2.0
    n_from_frames = int(np.floor((n_frames - 1 - first_frame) / step_frames + 1e-9)) + 1
    n_rows = max(0, min(n_from_blocks, n_from_frames))
    j = np.arange(n_rows, dtype=np.float64)
    coords = j * step_blocks
    frames = np.clip(np.floor(first_frame + j * step_frames + 0.5),
                     0, n_frames - 1).astype(np.int64)
    return coords, frames


def _slerp_rot6d(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Interpolate rot6d rows ``a``->``b`` by weights ``w`` ON SO(3). Shapes [N,6],[N,6],[N].

    rot6d is the first two columns of R; we Gram-Schmidt each endpoint to a real rotation,
    slerp the quaternions, and write the first two columns back. A component-wise lerp
    would produce a near-but-not-orthonormal frame whose Gram-Schmidt-recovered rotation
    is a different rotation than either endpoint -- wrong in a way no shape check sees."""
    def to_mat(r6):
        b1 = r6[:, 0:3] / (np.linalg.norm(r6[:, 0:3], axis=1, keepdims=True) + 1e-12)
        a2 = r6[:, 3:6]
        b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-12)
        return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)     # columns

    qa = Rotation.from_matrix(to_mat(a)).as_quat()
    qb = Rotation.from_matrix(to_mat(b)).as_quat()
    # Shortest arc: unit quaternions double-cover SO(3), so q and -q are the same
    # rotation but interpolate the LONG way round. Flip b where the dot is negative.
    dot = np.sum(qa * qb, axis=1, keepdims=True)
    qb = np.where(dot < 0.0, -qb, qb)
    dot = np.abs(np.clip(dot, -1.0, 1.0))
    theta = np.arccos(dot)                                   # [N,1], in [0, pi/2]
    sin_t = np.sin(theta)
    w = np.asarray(w, dtype=np.float64)[:, None]
    # nlerp near theta=0 (sin_t -> 0); slerp elsewhere. Both are then re-normalized, so
    # the small-angle branch is accurate to O(theta^2) and never divides by ~0.
    small = sin_t < 1e-7
    ca = np.where(small, 1.0 - w, np.sin((1.0 - w) * theta) / np.where(small, 1.0, sin_t))
    cb = np.where(small, w, np.sin(w * theta) / np.where(small, 1.0, sin_t))
    q = ca * qa + cb * qb
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    m = Rotation.from_quat(q).as_matrix()                     # [N,3,3]
    return np.concatenate([m[:, :, 0], m[:, :, 1]], axis=1)   # first two COLUMNS


def resample_lowdim(arr: np.ndarray, coords: np.ndarray,
                    rot_slices: tuple[tuple[int, int], ...] = (),
                    mode: str = "interp") -> np.ndarray:
    """Sample ``arr[B, D]`` at fractional block positions ``coords[N]`` -> ``[N, D]``.

    ``mode="interp"`` linearly interpolates ordinary dims and slerps every ``rot_slices``
    span on SO(3); ``mode="hold"`` takes floor(coords), keeping the commanded pose
    piecewise-constant (truthful to what the operator held, but with target_hz/block_hz
    identical rows in a row)."""
    if arr.ndim != 2:
        raise ValueError(f"expected [B, D], got {arr.shape}")
    n_blocks = arr.shape[0]
    coords = np.clip(np.asarray(coords, dtype=np.float64), 0.0, max(0.0, n_blocks - 1))
    lo = np.floor(coords).astype(np.int64)
    if mode == "hold":
        return arr[lo].astype(np.float32)
    if mode != "interp":
        raise ValueError(f"unknown resample mode {mode!r} (want 'interp' or 'hold')")
    hi = np.minimum(lo + 1, n_blocks - 1)
    w = (coords - lo)[:, None]
    src = arr.astype(np.float64)
    out = src[lo] * (1.0 - w) + src[hi] * w
    for s, e in rot_slices:
        if e <= s or e > arr.shape[1]:
            continue
        out[:, s:e] = _slerp_rot6d(src[lo, s:e], src[hi, s:e], w[:, 0])
    return out.astype(np.float32)


def valid_segments(mask: np.ndarray, min_blocks: int = 2) -> list[tuple[int, int]]:
    """Maximal runs of valid blocks as [start, stop) pairs, runs shorter than
    ``min_blocks`` dropped.

    Blocks whose aux state head produced nothing are recorded as NaN and flagged in
    ``state_valid.npy``. They cannot be interpolated THROUGH (the result would be NaN or,
    worse, a plausible number bridging a hole the operator never commanded), so each
    contiguous valid run becomes its OWN LeRobot episode instead of leaving a silent
    time discontinuity inside one episode. In practice NaNs sit at the drain edges, so
    this usually just trims a head/tail block."""
    segs, start = [], None
    for i, ok in enumerate(np.asarray(mask, dtype=bool).tolist() + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_blocks:
                segs.append((start, i))
            start = None
    return segs


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
            if rgb.shape[:2] != (RESIZE_H, RESIZE_W):
                rgb = cv2.resize(rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        cap.release()
    return np.asarray(frames, dtype=np.uint8)


def load_demo(demo_dir: str):
    """Load one WM demo -> (views {slot: [T,H,W,3]}, state[B,16], actions[B,20], valid[B],
    prompt, meta).

    Raises on any missing/mismatched piece so a bad demo is skipped, not silently
    written with garbage. Requires the state sidecar (openpi needs observation.state).

    ``valid[B]`` marks blocks whose aux state head actually produced a value. The recorder
    writes NaN rows plus ``state_valid.npy`` for the blocks it missed (async-decode drain
    edges); we also re-derive the mask from the arrays themselves, because a NaN that
    reached the dataset trains the policy to NaN loss on the very first batch that
    touches it."""
    meta = json.load(open(os.path.join(demo_dir, "meta.json")))
    require_absolute_cartesian(meta)   # before any decode: a wrong mode is unconvertible
    actions = np.load(os.path.join(demo_dir, "actions_raw.npy")).astype(np.float32)  # [B,20]
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions dim {actions.shape[1]} != {ACTION_DIM}")
    state_path = os.path.join(demo_dir, "state_raw.npy")
    if not os.path.exists(state_path):
        raise FileNotFoundError("state_raw.npy missing (demo recorded without a state head; "
                                "openpi needs observation.state)")
    state = np.load(state_path).astype(np.float32)                                   # [B,16]
    if state.shape[1] != STATE_DIM:
        raise ValueError(f"state dim {state.shape[1]} != {STATE_DIM}")

    num_blocks = min(state.shape[0], actions.shape[0])
    state, actions = state[:num_blocks], actions[:num_blocks]
    valid = np.isfinite(state).all(axis=1) & np.isfinite(actions).all(axis=1)
    valid_path = os.path.join(demo_dir, "state_valid.npy")
    if os.path.exists(valid_path):
        # AND with the recorder's own flags: it knows about blocks that came back empty
        # even in the (unlikely) case the array happens to hold finite garbage.
        flags = np.load(valid_path).astype(bool)
        if flags.shape[0] < num_blocks:
            raise ValueError(f"state_valid.npy has {flags.shape[0]} flags for "
                             f"{num_blocks} blocks")
        valid &= flags[:num_blocks]

    view_files = meta.get("view_files") or {name: f"view_{name}.mp4" for name in meta.get("view_names", [])}
    missing = [s for s in VIEW_TO_SLOT if s not in view_files]
    if missing:
        raise ValueError(f"demo missing required views {missing}; has {list(view_files)}")
    views = {VIEW_TO_SLOT[name]: read_frames(os.path.join(demo_dir, view_files[name]))
             for name in VIEW_TO_SLOT}

    prompt = meta.get("prompt") or meta.get("seed_episode") or "BimanualBikeRotorInstall"
    return views, state, actions, valid, str(prompt), meta


def plan_demo(state: np.ndarray, actions: np.ndarray, valid: np.ndarray, n_frames: int,
              meta: dict, args: Args) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One WM demo -> a list of resampled episodes ``(state[N,16], actions[N,20], frame_idx[N])``.

    Splits on invalid blocks first (each valid run becomes its own episode -- see
    :func:`valid_segments`), then puts each run on the fixed ``args.target_fps`` grid.
    Segment-local block indices are offset back into the demo's own frame timeline so the
    RGB pairing stays correct after a head trim."""
    data_fps, block_hz, rgb_per_block = demo_rates(meta, args.block_hz)
    out = []
    for start, stop in valid_segments(valid, min_blocks=args.min_blocks):
        # This run starts at block `start`, i.e. at frame `start * rgb_per_block`; ask for
        # the plan relative to the run, then shift the frame indices by that offset.
        frames_left = n_frames - start * rgb_per_block
        coords, frames = resample_indices(stop - start, rgb_per_block, frames_left,
                                          block_hz, data_fps, float(args.target_fps))
        if coords.size < args.min_frames:
            continue
        seg_state = resample_lowdim(state[start:stop], coords, mode=args.resample)
        seg_actions = resample_lowdim(actions[start:stop], coords,
                                      rot_slices=ACTION_ROT6D_SLICES, mode=args.resample)
        if not (np.isfinite(seg_state).all() and np.isfinite(seg_actions).all()):
            raise ValueError("resampled rows contain non-finite values")
        out.append((seg_state, seg_actions, frames + start * rgb_per_block))
    return out


def main(args: Args):
    _patch_encoder(args.vcodec)
    if args.resample not in ("interp", "hold"):
        raise SystemExit(f"--resample must be 'interp' or 'hold', got {args.resample!r}")
    if args.target_fps <= 0:
        raise SystemExit(f"--target-fps must be > 0, got {args.target_fps}")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    img_feat = {"dtype": "video", "shape": (RESIZE_H, RESIZE_W, 3), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="bimanual_panda",
        fps=args.target_fps,
        features={
            "observation.images.base": img_feat,
            "observation.images.left_wrist": img_feat,
            "observation.images.right_wrist": img_feat,
            "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    demos = list_demos(args.record_root)
    if args.limit:
        demos = demos[: args.limit]
    print(f"converting {len(demos)} WM demos from {args.record_root} -> {output_path} "
          f"(vcodec={args.vcodec}, target {args.target_fps} Hz, resample={args.resample})")

    written = skipped = split = 0
    rows_total = dropped_blocks = 0
    for demo_dir in tqdm.tqdm(demos):
        try:
            views, state, actions, valid, prompt, meta = load_demo(demo_dir)
            if args.prompt:
                prompt = args.prompt
            n_frames = min(v.shape[0] for v in views.values())
            episodes = plan_demo(state, actions, valid, n_frames, meta, args)
            if not episodes:
                skipped += 1
                print(f"SKIP {demo_dir}: no usable segment "
                      f"({int(valid.sum())}/{valid.size} valid blocks, "
                      f"{n_frames} frames)", flush=True)
                continue
            if len(episodes) > 1:
                # A demo with an interior NaN hole becomes >1 episode rather than one
                # episode with a silent time jump in the middle.
                split += 1
                print(f"NOTE {demo_dir}: split into {len(episodes)} episodes at invalid "
                      f"blocks ({int((~valid).sum())} of {valid.size})", flush=True)
            dropped_blocks += int((~valid).sum())
            for seg_state, seg_actions, frame_idx in episodes:
                for row in range(seg_state.shape[0]):
                    fi = int(frame_idx[row])
                    frame = {slot: views[slot][fi] for slot in VIEW_TO_SLOT.values()}
                    frame["observation.state"] = seg_state[row]
                    frame["actions"] = seg_actions[row]
                    frame["task"] = prompt
                    dataset.add_frame(frame)
                dataset.save_episode()
                written += 1
                rows_total += int(seg_state.shape[0])
        except Exception as ex:  # noqa: BLE001
            skipped += 1
            print(f"SKIP {demo_dir}: {type(ex).__name__}: {ex}", flush=True)

    print(f"Done: {written} episodes ({rows_total} rows @ {args.target_fps} Hz) written, "
          f"{skipped} demos skipped, {split} split at invalid blocks, "
          f"{dropped_blocks} invalid blocks dropped. Dataset at {output_path}")
    print("Next: compute norm stats, e.g.\n"
          "  uv run scripts/compute_norm_stats.py --config-name pi0_bike_rotor")

    if args.push_to_hub:
        dataset.push_to_hub(tags=["tri", "lbm", "bimanual", "panda", "world-model"],
                            private=True, push_videos=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
