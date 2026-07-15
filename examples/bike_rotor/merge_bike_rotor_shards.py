"""Merge the per-shard LeRobot datasets produced by parallel bike-rotor conversion into
one standard LeRobot v2.1 dataset that openpi loads by `repo_id`.

Each shard `<repo_id>_shardKKKofNNN` is a complete little LeRobot dataset. Merging =
concatenate episodes in shard order, renumbering episode_index / global index / task_index,
copying the parquet + video files, and rebuilding meta/. The temp `images/` scratch dirs
are ignored (not part of the dataset contract).

Usage (after running all shards):
    uv run examples/bike_rotor/merge_bike_rotor_shards.py --repo-id tri/bike_rotor_cartesian --num-shards 32
"""

import dataclasses
import json
import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import pandas as pd
import tyro


@dataclasses.dataclass
class Args:
    repo_id: str
    num_shards: int
    delete_shards: bool = False  # remove the per-shard datasets after a successful merge


def _read_jsonl(path: pathlib.Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: pathlib.Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main(args: Args):
    root = HF_LEROBOT_HOME
    shard_ids = [f"{args.repo_id}_shard{i:03d}of{args.num_shards:03d}" for i in range(args.num_shards)]
    shard_dirs = [root / s for s in shard_ids]
    for d in shard_dirs:
        if not d.exists():
            raise SystemExit(f"missing shard dataset: {d}")

    out = root / args.repo_id
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    info = json.loads((shard_dirs[0] / "meta" / "info.json").read_text())
    chunks_size = info["chunks_size"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]

    ep_offset = 0
    frame_offset = 0
    tasks = {}          # task string -> merged task_index
    episodes_meta = []
    episodes_stats = []

    for sdir in shard_dirs:
        s_info = json.loads((sdir / "meta" / "info.json").read_text())
        s_eps = {e["episode_index"]: e for e in _read_jsonl(sdir / "meta" / "episodes.jsonl")}
        s_stats = {e["episode_index"]: e for e in _read_jsonl(sdir / "meta" / "episodes_stats.jsonl")}
        s_tasks = {t["task_index"]: t["task"] for t in _read_jsonl(sdir / "meta" / "tasks.jsonl")}

        for le in sorted(s_eps):  # local episode indices, contiguous 0..n-1
            new_ep = ep_offset
            chunk = new_ep // chunks_size

            # --- parquet: renumber episode_index, global index, task_index ---
            src_pq = sdir / data_tmpl.format(episode_chunk=le // s_info["chunks_size"], episode_index=le)
            df = pd.read_parquet(src_pq)
            ep_task = s_tasks[int(df["task_index"].iloc[0])]
            new_task_idx = tasks.setdefault(ep_task, len(tasks))
            df["episode_index"] = new_ep
            df["task_index"] = new_task_idx
            df["index"] = frame_offset + df["frame_index"].to_numpy()
            dst_pq = out / data_tmpl.format(episode_chunk=chunk, episode_index=new_ep)
            dst_pq.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_pq, index=False)

            # --- videos: copy each view's mp4 to the renumbered path ---
            for vk in video_keys:
                src_v = sdir / video_tmpl.format(episode_chunk=le // s_info["chunks_size"], video_key=vk, episode_index=le)
                dst_v = out / video_tmpl.format(episode_chunk=chunk, video_key=vk, episode_index=new_ep)
                dst_v.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_v, dst_v)

            em = dict(s_eps[le]); em["episode_index"] = new_ep; episodes_meta.append(em)
            es = dict(s_stats[le]); es["episode_index"] = new_ep; episodes_stats.append(es)

            frame_offset += int(s_eps[le]["length"])
            ep_offset += 1

    # --- meta ---
    _write_jsonl(out / "meta" / "episodes.jsonl", episodes_meta)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", episodes_stats)
    _write_jsonl(out / "meta" / "tasks.jsonl",
                 [{"task_index": i, "task": t} for t, i in sorted(tasks.items(), key=lambda kv: kv[1])])

    info["total_episodes"] = ep_offset
    info["total_frames"] = frame_offset
    info["total_tasks"] = len(tasks)
    info["total_videos"] = ep_offset * len(video_keys)
    info["total_chunks"] = (ep_offset - 1) // chunks_size + 1 if ep_offset else 0
    info["splits"] = {"train": f"0:{ep_offset}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"Merged {args.num_shards} shards -> {out}")
    print(f"  episodes={ep_offset}  frames={frame_offset}  tasks={len(tasks)}  videos={info['total_videos']}")

    if args.delete_shards:
        for d in shard_dirs:
            shutil.rmtree(d)
        print(f"  removed {args.num_shards} shard datasets")


if __name__ == "__main__":
    main(tyro.cli(Args))
