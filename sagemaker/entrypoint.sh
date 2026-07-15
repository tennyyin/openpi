#!/bin/bash
set -euo pipefail

# Activate the uv-managed venv so `python`/jax/etc. are the openpi env, not system python.
source /opt/ml/code/.venv/bin/activate
cd /opt/ml/code

# CRITICAL: the SageMaker training toolkit runs under the base image's system Python 3.12
# and injects the 3.12 stdlib into PYTHONPATH (…/lib/python3.12…). Our venv is Python 3.11,
# so that leak makes the 3.11 interpreter import 3.12's re/dataclasses -> "SRE module
# mismatch". openpi is installed editable in the venv, so reset PYTHONPATH to code dirs only.
export PYTHONPATH=/opt/ml/code/src:/opt/ml/code

# --- SageMaker channel / path wiring -----------------------------------------
# dataset channel (FastFile): its root is used as HF_LEROBOT_HOME, so a config's
# repo_id (e.g. "tri/bike_rotor_cartesian") resolves to <channel>/tri/bike_rotor_cartesian.
export HF_LEROBOT_HOME=${SM_CHANNEL_DATASET:-/opt/ml/input/data/dataset}
# base_ckpt channel (File): holds {pi0_base,pi05_base}/params, overriding the config's
# gs:// weight_loader path so no GCS egress is needed at train time.
BASE_CKPT_ROOT=${SM_CHANNEL_BASE_CKPT:-/opt/ml/input/data/base_ckpt}

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Let JAX use most of the H200's 141GB (default is 75%).
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

CONFIG=${SM_HP_CONFIG:?SM_HP_CONFIG (openpi config name) is required}
EXP_NAME=${SM_HP_EXP_NAME:-${SM_JOB_NAME:-run}}
NPROC=$(nvidia-smi -L | wc -l)

# Base-model params dir for this config. pi05* configs load pi05_base; else pi0_base.
case "$CONFIG" in
    pi05_*) BASE_PARAMS="$BASE_CKPT_ROOT/pi05_base/params" ;;
    *)      BASE_PARAMS="$BASE_CKPT_ROOT/pi0_base/params" ;;
esac

echo "=== openpi SageMaker entrypoint ==="
echo "CONFIG=$CONFIG  EXP_NAME=$EXP_NAME  NPROC=$NPROC"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "BASE_PARAMS=$BASE_PARAMS"
echo "==================================="

if [ "${SM_HP_STAGE:-train}" = "smoke_test" ]; then
    python -c "import jax; print('jax devices:', jax.devices())"
    ls -la "$HF_LEROBOT_HOME" || true
    ls -la "$BASE_PARAMS" || true
    exit 0
fi

# Checkpoints go to /opt/ml/checkpoints -> SageMaker syncs them to checkpoint_s3_uri.
# JAX single-node uses all local GPUs automatically; --fsdp-devices shards the model
# across them to cut memory (set to NPROC here; harmless data-parallel-equivalent for
# a model that fits, essential if it would otherwise OOM).
exec python scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir=/opt/ml/checkpoints \
    --weight-loader.params-path="$BASE_PARAMS" \
    --fsdp-devices="${SM_HP_FSDP_DEVICES:-$NPROC}" \
    --overwrite
