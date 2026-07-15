#!/bin/bash
# Stage the converted LeRobot dataset + the pi0/pi05 base checkpoints to S3 so the
# SageMaker jobs can mount them offline (no gs:// egress needed on the training box).
#
# Uploads to the robotics-new bucket with the rob-sm profile:
#   s3://tri-ml-sandbox-16011-us-west-2-datasets/openpi/datasets/<repo_id>/
#   s3://tri-ml-sandbox-16011-us-west-2-datasets/openpi/base_ckpts/{pi0_base,pi05_base}/params/
#
# Usage: bash sagemaker/stage_to_s3.sh [repo_id]
set -euo pipefail

REPO_ID=${1:-tri/bike_rotor_cartesian}
PROFILE=${AWS_PROFILE_STAGE:-rob-sm}
REGION=us-west-2
S3_ROOT="s3://tri-ml-sandbox-16011-us-west-2-datasets/openpi"
LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
STAGE_DIR=${STAGE_DIR:-/tmp/openpi_base_ckpts}

echo "== 1/3 Fetching base checkpoints from gs://openpi-assets (public) =="
mkdir -p "$STAGE_DIR"
for M in pi0_base pi05_base; do
    if [ ! -d "$STAGE_DIR/$M/params" ]; then
        # openpi's downloader handles gs://openpi-assets via gsutil/gcsfs and caches locally.
        uv run python -c "
import openpi.shared.download as d, shutil, pathlib
src = d.maybe_download('gs://openpi-assets/checkpoints/$M/params')
dst = pathlib.Path('$STAGE_DIR/$M/params')
dst.parent.mkdir(parents=True, exist_ok=True)
if not dst.exists(): shutil.copytree(src, dst)
print('staged', dst)
"
    fi
done

echo "== 2/3 Uploading base checkpoints to S3 =="
aws s3 sync "$STAGE_DIR/pi0_base/params"  "$S3_ROOT/base_ckpts/pi0_base/params"  --profile "$PROFILE" --region "$REGION"
aws s3 sync "$STAGE_DIR/pi05_base/params" "$S3_ROOT/base_ckpts/pi05_base/params" --profile "$PROFILE" --region "$REGION"

echo "== 3/3 Uploading LeRobot dataset ($REPO_ID) to S3 =="
aws s3 sync "$LEROBOT_HOME/$REPO_ID" "$S3_ROOT/datasets/$REPO_ID" --profile "$PROFILE" --region "$REGION"

echo "Done. Dataset: $S3_ROOT/datasets/$REPO_ID   Base ckpts: $S3_ROOT/base_ckpts/"
