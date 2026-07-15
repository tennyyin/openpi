#!/bin/bash
# Thin wrapper around launch_sm.py for the bike-rotor fine-tunes.
# Usage: bash sagemaker/run_sm.sh <pi0_bike_rotor|pi05_bike_rotor> <NAME> [QUEUE] [BUILD_TYPE] [extra launch_sm.py flags...]
#
# Env: SM_USER (required), WANDB_API_KEY (or `wandb login`).
# Prereq: dataset + base checkpoints already staged to S3 (see stage_to_s3.sh), and
#         norm stats computed + baked into the image (see README.md).
set -euo pipefail

# Drop any ambient env creds (e.g. ml-dgx-bot in robotics-old) so the launcher uses the
# SSO profile it selects internally (rob-sm / robotics-new). Leaving these set makes the
# Batch submit run from the wrong account -> "Cross-account pass role is not allowed".
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE

CONFIG=${1:?usage: run_sm.sh <config> <name> [queue] [build_type]}
NAME=${2:?usage: run_sm.sh <config> <name> [queue] [build_type]}
QUEUE=${3:-cv-wfm}
BUILD_TYPE=${4:-full}
shift $(( $# < 4 ? $# : 4 ))

: "${SM_USER:?set SM_USER}"

SM_LAUNCH_PYTHON=${SM_LAUNCH_PYTHON:-python}
exec "$SM_LAUNCH_PYTHON" "$(dirname "$0")/launch_sm.py" \
    --config "$CONFIG" \
    --name "$NAME" \
    --queue "$QUEUE" \
    --build-type "$BUILD_TYPE" \
    "$@"
