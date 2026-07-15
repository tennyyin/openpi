"""Compute openpi normalization stats for the bike-rotor configs directly from the raw
LBM lowdim -- no video decode needed (fast, and avoids a local ffmpeg/torchcodec dep).

This is equivalent to `scripts/compute_norm_stats.py` for these configs: our data
transforms do not modify the state/action values numerically (no delta transform), so
the stats over the raw per-frame state(16)/action(20) match what the dataloader would
produce. We reuse openpi's own RunningStats so the format + quantile method are identical.

Writes norm_stats.json into BOTH configs' asset dirs (the raw mean/std/q01/q99 are the
same; pi0 uses z-score, pi05 uses quantile -- both read from the same file):
    assets/pi0_bike_rotor/tri/bike_rotor_cartesian/norm_stats.json
    assets/pi05_bike_rotor/tri/bike_rotor_cartesian/norm_stats.json

Usage:
    uv run examples/bike_rotor/compute_norm_stats_from_raw.py
"""

import dataclasses
import pathlib
import sys

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config

# Make the sibling converter module importable regardless of CWD (run this from the
# repo root so openpi's ./assets resolves to <repo>/assets, where the image bakes them).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from convert_bike_rotor_to_lerobot import DEFAULT_RAW_ROOT
from convert_bike_rotor_to_lerobot import is_successful
from convert_bike_rotor_to_lerobot import list_teleop_episodes
from convert_bike_rotor_to_lerobot import load_lowdim


@dataclasses.dataclass
class Args:
    raw_root: str = DEFAULT_RAW_ROOT
    repo_id: str = "tri/bike_rotor_cartesian"
    configs: tuple[str, ...] = ("pi0_bike_rotor", "pi05_bike_rotor")
    teleop_only: bool = True
    successful_only: bool = False
    limit: int = 0


def main(args: Args):
    eps = list_teleop_episodes(args.raw_root, teleop_only=args.teleop_only)
    if args.successful_only:
        eps = [e for e in eps if is_successful(e)]
    if args.limit:
        eps = eps[: args.limit]
    print(f"Computing norm stats over {len(eps)} episodes")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    for ep in tqdm.tqdm(eps):
        try:
            state, actions = load_lowdim(ep)
        except Exception as ex:  # noqa: BLE001
            print(f"SKIP {ep}: {ex}")
            continue
        stats["state"].update(np.asarray(state, dtype=np.float32))
        stats["actions"].update(np.asarray(actions, dtype=np.float32))

    norm_stats = {k: v.get_statistics() for k, v in stats.items()}
    print("state  mean/std/q01/q99 dims:", norm_stats["state"].mean.shape)
    print("actions mean/std/q01/q99 dims:", norm_stats["actions"].mean.shape)

    for cfg_name in args.configs:
        cfg = _config.get_config(cfg_name)
        out = cfg.assets_dirs / args.repo_id
        normalize.save(out, norm_stats)
        print(f"wrote {out / 'norm_stats.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
