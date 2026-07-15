#!/bin/bash
# Parallel bike-rotor -> LeRobot conversion: launch N shard processes over disjoint
# episode subsets, then merge into one dataset. Each shard encodes video (h264) itself,
# so this scales ~linearly with shard count (CPU/codec bound).
#
# Usage: bash examples/bike_rotor/run_conversion_parallel.sh [REPO_ID] [NUM_SHARDS] [EXTRA convert flags...]
#   e.g. bash examples/bike_rotor/run_conversion_parallel.sh tri/bike_rotor_cartesian 32
set -euo pipefail

REPO_ID=${1:-tri/bike_rotor_cartesian}
NUM_SHARDS=${2:-32}
shift $(( $# < 2 ? $# : 2 )) || true
EXTRA=("$@")

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."   # repo root, so uv + openpi assets resolve correctly
LOGDIR=$(mktemp -d /tmp/bike_rotor_convert.XXXXXX)
echo "Launching $NUM_SHARDS shard converters for $REPO_ID (logs: $LOGDIR)"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    uv run examples/bike_rotor/convert_bike_rotor_to_lerobot.py \
        --repo-id "$REPO_ID" --num-shards "$NUM_SHARDS" --shard-id "$i" "${EXTRA[@]}" \
        > "$LOGDIR/shard$i.log" 2>&1 &
    pids+=($!)
done

fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "SHARD $idx FAILED -- see $LOGDIR/shard$idx.log"; tail -5 "$LOGDIR/shard$idx.log"; fail=1
    fi
done
[ "$fail" -eq 0 ] || { echo "One or more shards failed; NOT merging."; exit 1; }

echo "All shards done. Merging..."
uv run examples/bike_rotor/merge_bike_rotor_shards.py --repo-id "$REPO_ID" --num-shards "$NUM_SHARDS" --delete-shards
echo "Conversion complete: $REPO_ID"
