"""Convert world-model DAgger rollouts into a LeRobot dataset for openpi fine-tuning.

Input is a run root written by open-world's ``scripts/interactive_ar.py --policy-dagger``:
the pi05 policy drives inside the tri_bike AR world model, the operator interrupts it with
the SpaceMice, and each take lands in its own directory::

    <run-root>/rollout_0000/
        actions_cmd_raw.npy   [B, 20] the pose actually COMMANDED   <-- the label
        actions_raw.npy       [B, 20] the pose the world model was CONDITIONED on
        obs_state_raw.npy     [B, 16] proprioception the block was DRIVEN FROM  <-- the obs
        obs_state_valid.npy   [B]     False where none was known
        state_raw.npy         [B, 16] aux state head AFTER the block
        source.npy            [B]     0 = policy drove this block, 1 = human drove it
        corr_id.npy           [B]     intervention id on human blocks, -1 elsewhere
        view_base.mp4 / view_left_wrist.mp4 / view_right_wrist.mp4
        events.jsonl, meta.json
    <run-root>/rollout_0001/ ...

Everything rate- and geometry-related is shared with the teleop converter
(:mod:`convert_wm_demos_to_lerobot`): the same 10 Hz output grid, the same
centre-of-block frame pairing, the same SO(3) slerp for rot6d, the same
absolute-cartesian gate. This script only adds what DAgger means.

Which arrays a DAgger dump is read from
======================================
``actions_cmd_raw`` not ``actions_raw``
    The command is what a human or the policy issued; the conditioning stream is the
    action adapter's output (the plant). Regressing the adapter's output teaches the
    policy to imitate the plant instead of the operator. On policy blocks the adapter is
    bypassed so the two are equal -- which this script CHECKS, because that equality is
    the on-disk evidence that the bypass happened.
``obs_state_raw`` not ``state_raw``
    A BC frame pairs an action with the observation that PRECEDED it. ``state_raw`` is the
    aux head's reading AFTER the block; the two files differ by one block, and using the
    wrong one trains the policy to predict its own past. If a dump has no
    ``obs_state_raw`` (written before it existed) this script reconstructs it by shifting
    ``state_raw`` one block and dropping block 0, and says so.

Recipes are ``--include`` / ``--human-weight``, not separate collections
=======================================================================
One dump holds the whole trajectory with a source label per block, so:

    --include human                 HG-DAgger: train on the operator's blocks only.
    --include all                   full-episode BC over policy + human blocks.
    --include all --human-weight 3  IWR: the run once, plus its human spans twice more.
    --include policy                the policy's own blocks, e.g. to measure it unaided.

``--human-weight`` upweights by DUPLICATION because a LeRobot dataset carries no
per-sample weight; N=1 (default) is plain BC.

Never interpolate across a takeover
===================================
Output rows sit on a 10 Hz grid and are interpolated between neighbouring blocks -- except
where the two neighbours have DIFFERENT sources (or where the later one has no valid
observation), in which case the row HOLDS the earlier block. Interpolating a policy block
into a human block would emit a pose neither of them commanded, exactly at the moment the
data is most valuable. The run still becomes ONE episode wherever it stays continuous;
episodes only split where blocks are actually dropped.

Usage (run inside the openpi uv env)::

    uv run examples/bike_rotor/convert_dagger_to_lerobot.py \
        --record-root /path/to/open-world/runs/dagger_pi05 \
        --repo-id tri/bike_rotor_dagger --include all --human-weight 3
"""

import dataclasses
import json
import os
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

from convert_wm_demos_to_lerobot import ACTION_DIM
from convert_wm_demos_to_lerobot import ACTION_ROT6D_SLICES
from convert_wm_demos_to_lerobot import RECORD_TARGET_HZ
from convert_wm_demos_to_lerobot import RESIZE_H
from convert_wm_demos_to_lerobot import RESIZE_W
from convert_wm_demos_to_lerobot import STATE_DIM
from convert_wm_demos_to_lerobot import VIEW_TO_SLOT
from convert_wm_demos_to_lerobot import _patch_encoder
from convert_wm_demos_to_lerobot import demo_rates
from convert_wm_demos_to_lerobot import read_frames
from convert_wm_demos_to_lerobot import require_absolute_cartesian
from convert_wm_demos_to_lerobot import resample_indices
from convert_wm_demos_to_lerobot import resample_lowdim
from convert_wm_demos_to_lerobot import valid_segments

#: Per-block source labels, as written by open-world's dagger session. Part of the on-disk
#: format on both sides, so these values are not free to change.
SOURCE_POLICY = 0
SOURCE_HUMAN = 1

#: The two gripper columns of the 20-d action (right 18, left 19 -- NOT arm order).
#:
#: These are NEVER interpolated. Recorded bimanual gripper commands are effectively binary
#: (checked on the AR latent root: 0.0 or 0.1, with at most one intermediate sample per
#: episode even at the native 10 Hz), while a DAgger dump carries one command per 0.4 s
#: block. Lerping 0.1 -> 0.0 across a block boundary at 2.5 -> 10 Hz emits rows at 0.075 /
#: 0.05 / 0.025 -- values that essentially do not occur in the training distribution, placed
#: exactly on the frames that decide WHEN to close. Worse, a policy that learns to emit them
#: sits under the open threshold and over the close one, so the grasp never commits. Holding
#: the earlier block's value keeps the label binary and is also what the dump means: one
#: command was issued and held for the whole block.
GRIP_COLS = (18, 19)


@dataclasses.dataclass
class Args:
    repo_id: str
    record_root: str                  # dir holding rollout_XXXX/ subdirectories
    include: str = "all"              # "all" | "human" | "policy" -- the DAgger recipe
    # IWR: how many times the human spans appear, in ADDITION to whatever `include`
    # already emits. 1 = no upweighting. Only meaningful with --include all.
    human_weight: int = 1
    label: str = "command"            # "command" (actions_cmd_raw) | "conditioning" (actions_raw)
    limit: int = 0                    # cap #rollouts (0 = all); for smoke tests
    prompt: str = ""                  # override the per-rollout prompt (else meta.json's)
    target_fps: int = int(RECORD_TARGET_HZ)
    resample: str = "interp"          # "interp" (time-interpolate, rot6d via SO(3)) | "hold"
    # Keep the gripper columns piecewise-constant even under --resample interp. Effectively
    # always on; the flag exists to make the behaviour visible and reproducible, not tunable.
    gripper_hold: bool = True
    block_hz: float = 0.0             # override the dump's meta block rate (0 = trust meta)
    min_frames: int = 2               # skip episodes with fewer than this many OUTPUT rows
    min_blocks: int = 2               # skip runs with fewer than this many usable blocks
    # Tolerance for the "adapter was bypassed on policy blocks" check, in normalized-action
    # units de-normalized to metres/rot6d components. 0 disables the check.
    bypass_tol: float = 1e-4
    push_to_hub: bool = False
    vcodec: str = "h264"


def list_rollouts(record_root: str) -> list[str]:
    """Every rollout directory under ``record_root``, sorted; one nesting level allowed.

    Mirrors open-world's ``dagger.index.scan_run_root``: a directory counts if it holds a
    ``meta.json``, whatever it is named, so a hand-renamed or hand-grouped run still
    converts. Anything without a meta.json is silently a container, not a take.
    """
    out: list[str] = []
    if not os.path.isdir(record_root):
        return out
    for name in sorted(os.listdir(record_root)):
        d = os.path.join(record_root, name)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "meta.json")):
            out.append(d)
            continue
        for sub in sorted(os.listdir(d)):
            sd = os.path.join(d, sub)
            if os.path.isdir(sd) and os.path.exists(os.path.join(sd, "meta.json")):
                out.append(sd)
    return out


def _load(path: str):
    return np.load(path) if os.path.exists(path) else None


def load_rollout(dump_dir: str, *, label: str = "command", bypass_tol: float = 1e-4):
    """Load one DAgger take -> (views, obs_state[B,16], actions[B,20], valid[B], source[B],
    prompt, meta).

    Raises on anything unconvertible so a bad take is skipped rather than written with
    garbage. ``valid[B]`` is the AND of the writer's own ``obs_state_valid`` flags and a
    fresh finiteness check on the arrays -- a NaN that reaches the dataset trains the
    policy to NaN loss on the first batch that touches it.
    """
    meta = json.load(open(os.path.join(dump_dir, "meta.json")))
    require_absolute_cartesian(meta)   # before any decode: a wrong mode is unconvertible
    if not meta.get("dagger"):
        raise ValueError("meta.json has no 'dagger' flag: this is a teleop demo, not a "
                         "policy-DAgger rollout -- convert it with "
                         "convert_wm_demos_to_lerobot.py")

    act_file = "actions_cmd_raw.npy" if label == "command" else "actions_raw.npy"
    actions = np.load(os.path.join(dump_dir, act_file)).astype(np.float32)      # [B, 20]
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"{act_file} dim {actions.shape[1]} != {ACTION_DIM}")
    B = actions.shape[0]

    source = _load(os.path.join(dump_dir, "source.npy"))
    if source is None:
        raise FileNotFoundError("source.npy missing: without per-block labels there is no "
                                "way to tell a policy block from a correction, which is the "
                                "only thing that makes this a DAgger dump")
    source = np.asarray(source, dtype=np.int64).reshape(-1)[:B]
    if source.shape[0] != B:
        raise ValueError(f"source.npy has {source.shape[0]} labels for {B} blocks")

    # --- the observation each block was DRIVEN FROM -------------------------------------
    obs = _load(os.path.join(dump_dir, "obs_state_raw.npy"))
    shifted = False
    if obs is None:
        # Pre-obs_state dumps: state_raw[i] is the state AFTER block i, so the state BEFORE
        # block i is state_raw[i-1] and block 0's is unrecoverable. Reconstruct exactly
        # that and drop block 0, rather than silently pairing each action with the state it
        # produced (which trains the policy to predict its own past).
        st = _load(os.path.join(dump_dir, "state_raw.npy"))
        if st is None:
            raise FileNotFoundError(
                "neither obs_state_raw.npy nor state_raw.npy present (take recorded "
                "without a state head; openpi needs observation.state)")
        obs = np.full_like(np.asarray(st, dtype=np.float32), np.nan)
        obs[1:] = np.asarray(st, dtype=np.float32)[:-1]
        shifted = True
        print(f"NOTE {dump_dir}: no obs_state_raw.npy; reconstructed the driven-from state "
              f"by shifting state_raw one block (block 0 dropped)", flush=True)
    obs = np.asarray(obs, dtype=np.float32)[:B]
    if obs.shape[1] != STATE_DIM:
        raise ValueError(f"observation state dim {obs.shape[1]} != {STATE_DIM}")

    valid = np.isfinite(obs).all(axis=1) & np.isfinite(actions).all(axis=1)
    flags = _load(os.path.join(dump_dir, "obs_state_valid.npy"))
    if flags is not None and not shifted:
        flags = np.asarray(flags, dtype=bool).reshape(-1)
        if flags.shape[0] < B:
            raise ValueError(f"obs_state_valid.npy has {flags.shape[0]} flags for {B} blocks")
        valid &= flags[:B]

    # --- the bypass check ---------------------------------------------------------------
    # On a policy block the action adapter is bypassed, so the conditioning pose and the
    # command must be bit-identical. If they differ, this dump was written by a build that
    # ran the plant on the policy too, and `--label command` is then the adapter's INPUT
    # while the video followed its OUTPUT -- a mismatch worth failing on, not warning about.
    if bypass_tol > 0:
        cond = _load(os.path.join(dump_dir, "actions_raw.npy"))
        cmd = _load(os.path.join(dump_dir, "actions_cmd_raw.npy"))
        if cond is not None and cmd is not None:
            pol = (source == SOURCE_POLICY) & valid
            if pol.any():
                d = float(np.nanmax(np.abs(np.asarray(cond)[:B][pol]
                                           - np.asarray(cmd)[:B][pol])))
                if d > bypass_tol:
                    raise ValueError(
                        f"policy blocks disagree between actions_raw and actions_cmd_raw by "
                        f"{d:g} > {bypass_tol:g}: the action adapter was NOT bypassed for the "
                        f"policy, so the commands here are not what drove the video "
                        f"(pass --bypass-tol 0 to convert anyway)")

    view_files = (meta.get("view_files")
                  or {n: f"view_{n}.mp4" for n in meta.get("view_names", [])})
    missing = [s for s in VIEW_TO_SLOT if s not in view_files]
    if missing:
        raise ValueError(f"take missing required views {missing}; has {list(view_files)}")
    views = {VIEW_TO_SLOT[n]: read_frames(os.path.join(dump_dir, view_files[n]))
             for n in VIEW_TO_SLOT}

    anchor = meta.get("anchor") or {}
    prompt = (meta.get("prompt") or anchor.get("instruction")
              or meta.get("seed_episode") or "BimanualBikeRotorInstall")
    return views, obs, actions, valid, source, str(prompt), meta


def keep_mask(valid: np.ndarray, source: np.ndarray, include: str) -> np.ndarray:
    """Per-block mask for a DAgger recipe. ``include`` in {all, human, policy}."""
    keep = np.asarray(valid, dtype=bool).copy()
    if include == "human":
        keep &= (source == SOURCE_HUMAN)
    elif include == "policy":
        keep &= (source == SOURCE_POLICY)
    elif include != "all":
        raise ValueError(f"include must be all/human/policy, got {include!r}")
    return keep


def plan_rollout(obs: np.ndarray, actions: np.ndarray, keep: np.ndarray,
                 valid: np.ndarray, source: np.ndarray, n_frames: int,
                 meta: dict, args: Args) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One take -> resampled episodes ``(obs[N,16], actions[N,20], frame_idx[N])``.

    ONE global output grid is laid over the whole take (so timestamps never restart
    mid-run), then:

    * a row is dropped when the block it is attributed to (``floor(coord)``) is not kept --
      dropping rows is what splits a take into more than one episode;
    * a row HOLDS its earlier block instead of interpolating when the block pair straddles
      a takeover (different ``source``) or the later block has no valid observation. This
      is the "never invent a command across a boundary" rule; it costs at most one row of
      smoothness per intervention edge and is applied even under ``--resample hold``
      (where it is already a no-op).
    """
    data_fps, block_hz, rgb_per_block = demo_rates(meta, args.block_hz)
    B = int(actions.shape[0])
    if not np.asarray(valid, dtype=bool).any() or not np.asarray(keep, dtype=bool).any():
        return []
    coords, frames = resample_indices(B, rgb_per_block, n_frames, block_hz, data_fps,
                                      float(args.target_fps))
    if coords.size == 0:
        return []
    lo = np.clip(np.floor(coords).astype(np.int64), 0, B - 1)
    hi = np.minimum(lo + 1, B - 1)

    obs = _fill_invalid(obs, valid)
    actions = _fill_invalid(actions, valid)
    st = resample_lowdim(obs, coords, mode=args.resample)
    ac = resample_lowdim(actions, coords, rot_slices=ACTION_ROT6D_SLICES,
                         mode=args.resample)
    # Grippers step, they do not ramp (see GRIP_COLS). resample_lowdim slerps the rot6d
    # spans and lerps everything else, so cols 18/19 have to be put back by hand.
    if args.gripper_hold:
        ac[:, list(GRIP_COLS)] = actions[np.ix_(lo, GRIP_COLS)]
    hold = (source[lo] != source[hi]) | (~valid[hi])
    if hold.any():
        st[hold] = obs[lo[hold]]
        ac[hold] = actions[lo[hold]]

    rows_ok = keep[lo]
    out = []
    for r0, r1 in valid_segments(rows_ok, min_blocks=args.min_frames):
        seg_st, seg_ac, seg_fi = st[r0:r1], ac[r0:r1], frames[r0:r1]
        if not (np.isfinite(seg_st).all() and np.isfinite(seg_ac).all()):
            raise ValueError("resampled rows contain non-finite values")
        # A run of rows can still come from too few distinct blocks to be a trajectory
        # (target_fps/block_hz rows per block means 4 rows can be one held pose).
        if len(set(lo[r0:r1].tolist())) < args.min_blocks:
            continue
        out.append((seg_st, seg_ac, seg_fi))
    return out


def _fill_invalid(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid rows of ``arr`` with the nearest valid row (previous, else next).

    Purely defensive: every row an invalid block would produce is dropped downstream
    (``keep[lo]`` is False there, and a straddling row holds its valid neighbour). But the
    resampler slerps the WHOLE array in one call, and ``Rotation.from_matrix`` raises on a
    non-finite matrix -- so one NaN block at a drain edge would otherwise take the entire
    take down instead of costing it one block.
    """
    a = np.asarray(arr, dtype=np.float32).copy()
    v = np.asarray(valid, dtype=bool)
    if v.all():
        return a
    idx = np.where(v, np.arange(v.size), -1)
    idx = np.maximum.accumulate(idx)                       # nearest valid at or before i
    first = int(np.argmax(v))                              # v.any() is guaranteed by caller
    idx[idx < 0] = first                                   # leading invalid rows -> first valid
    return a[idx]


def human_spans(source: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``keep`` restricted to human blocks -- the extra episodes ``--human-weight`` writes."""
    return np.asarray(keep, dtype=bool) & (np.asarray(source) == SOURCE_HUMAN)


def main(args: Args):
    _patch_encoder(args.vcodec)
    if args.include not in ("all", "human", "policy"):
        raise SystemExit(f"--include must be all/human/policy, got {args.include!r}")
    if args.resample not in ("interp", "hold"):
        raise SystemExit(f"--resample must be 'interp' or 'hold', got {args.resample!r}")
    if args.label not in ("command", "conditioning"):
        raise SystemExit(f"--label must be command/conditioning, got {args.label!r}")
    if args.target_fps <= 0:
        raise SystemExit(f"--target-fps must be > 0, got {args.target_fps}")
    if args.human_weight < 1:
        raise SystemExit(f"--human-weight must be >= 1, got {args.human_weight}")
    if args.human_weight > 1 and args.include != "all":
        # With include=human the human blocks are already the whole dataset, so a weight
        # would just duplicate everything and change nothing but the epoch length.
        raise SystemExit("--human-weight > 1 only means something with --include all "
                         "(that is what IWR is: everything, with the corrections upweighted)")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    img_feat = {"dtype": "video", "shape": (RESIZE_H, RESIZE_W, 3),
                "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="bimanual_panda",
        fps=args.target_fps,
        features={
            "observation.images.base": img_feat,
            "observation.images.left_wrist": img_feat,
            "observation.images.right_wrist": img_feat,
            "observation.state": {"dtype": "float32", "shape": (STATE_DIM,),
                                  "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    takes = list_rollouts(args.record_root)
    if args.limit:
        takes = takes[: args.limit]
    print(f"converting {len(takes)} DAgger takes from {args.record_root} -> {output_path} "
          f"(include={args.include}, human_weight={args.human_weight}, label={args.label}, "
          f"target {args.target_fps} Hz, resample={args.resample})")

    # Per-episode provenance. LeRobot carries no free-form per-episode field, so the map
    # from episode index back to (take, source composition, whether it is an IWR duplicate)
    # is written beside the dataset -- without it, a mixed dataset cannot be audited.
    manifest: list[dict] = []
    written = skipped = 0
    rows_total = human_rows = dropped_blocks = dup_eps = 0
    for d in tqdm.tqdm(takes):
        try:
            views, obs, actions, valid, source, prompt, meta = load_rollout(
                d, label=args.label, bypass_tol=args.bypass_tol)
            if args.prompt:
                prompt = args.prompt
            n_frames = min(v.shape[0] for v in views.values())
            base_keep = keep_mask(valid, source, args.include)
            passes = [(base_keep, False)]
            if args.human_weight > 1:
                hs = human_spans(source, base_keep)
                if hs.any():
                    passes += [(hs, True)] * (args.human_weight - 1)
            n_eps = 0
            for keep, is_dup in passes:
                for seg_st, seg_ac, frame_idx in plan_rollout(
                        obs, actions, keep, valid, source, n_frames, meta, args):
                    for row in range(seg_st.shape[0]):
                        fi = int(frame_idx[row])
                        frame = {slot: views[slot][fi] for slot in VIEW_TO_SLOT.values()}
                        frame["observation.state"] = seg_st[row]
                        frame["actions"] = seg_ac[row]
                        frame["task"] = prompt
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    n_rows = int(seg_st.shape[0])
                    manifest.append({
                        "episode_index": written,
                        "take": os.path.relpath(d, args.record_root),
                        "rollout_id": meta.get("rollout_id"),
                        "anchor": meta.get("anchor") or {},
                        "rows": n_rows,
                        "iwr_duplicate": bool(is_dup),
                        "human_only": bool(is_dup or args.include == "human"),
                        "prompt": prompt,
                    })
                    written += 1
                    n_eps += 1
                    rows_total += n_rows
                    if is_dup or args.include == "human":
                        human_rows += n_rows
                    if is_dup:
                        dup_eps += 1
            if n_eps == 0:
                skipped += 1
                print(f"SKIP {d}: no usable segment ({int(base_keep.sum())}/"
                      f"{base_keep.size} blocks kept, {n_frames} frames)", flush=True)
                continue
            dropped_blocks += int((~base_keep).sum())
            if n_eps > len(passes):
                print(f"NOTE {d}: split into {n_eps} episodes at dropped blocks "
                      f"({int((~base_keep).sum())} of {base_keep.size})", flush=True)
        except Exception as ex:  # noqa: BLE001
            skipped += 1
            print(f"SKIP {d}: {type(ex).__name__}: {ex}", flush=True)

    man_path = output_path / "dagger_manifest.json"
    try:
        with open(man_path, "w") as f:
            json.dump({"record_root": os.path.abspath(args.record_root),
                       "include": args.include, "human_weight": args.human_weight,
                       "label": args.label, "target_fps": args.target_fps,
                       "resample": args.resample, "episodes": manifest}, f, indent=2)
    except OSError as ex:                                            # noqa: BLE001
        print(f"WARN: could not write {man_path}: {ex}", flush=True)

    print(f"Done: {written} episodes ({rows_total} rows @ {args.target_fps} Hz, "
          f"{human_rows} from corrections, {dup_eps} IWR duplicates) written, "
          f"{skipped} takes skipped, {dropped_blocks} blocks dropped. "
          f"Dataset at {output_path}")
    print("Next: compute norm stats, e.g.\n"
          "  uv run scripts/compute_norm_stats.py --config-name pi05_bike_rotor")

    if args.push_to_hub:
        dataset.push_to_hub(tags=["tri", "lbm", "bimanual", "panda", "world-model", "dagger"],
                            private=True, push_videos=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
