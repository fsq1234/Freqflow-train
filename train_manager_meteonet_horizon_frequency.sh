#!/usr/bin/env bash
# Fine-tune the MeteoNet VAE-latent FreqFlow with horizon-aware frequency routing.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FREQFLOW_PROJECT_DIR="${SCRIPT_DIR}"
export FREQFLOW_MANAGER_NAME="$(basename -- "$0")"
export FREQFLOW_DATASET=meteo
export FREQFLOW_DATA_PATH=/mnt/sdb1/user/FSQ/Freqflow-train/data/meteo_latent_vae32.h5
export FREQFLOW_RAW_DATA_PATH=/mnt/sdb1/user/WRF/stlP/data/meteo_radar.h5
export FREQFLOW_VAE_CHECKPOINT=/mnt/sdb1/user/FSQ/Freqflow-train/runs/meteo-vae32/20260915_220910/checkpoints/best_val_loss.pt
export FREQFLOW_INIT_CHECKPOINT=/mnt/sdb1/user/FSQ/Freqflow-train/runs/meteo-vae32-5to20/20260916_114600/checkpoints/best.pt
export FREQFLOW_INIT_CHECKPOINT_STATE="${FREQFLOW_INIT_CHECKPOINT_STATE:-ema}"

export FREQFLOW_RUN_GROUP="${FREQFLOW_RUN_GROUP:-meteo-vae32-horizon-frequency}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FREQFLOW_NUM_PROCESSES="${FREQFLOW_NUM_PROCESSES:-1}"
export FREQFLOW_BATCH_SIZE="${FREQFLOW_BATCH_SIZE:-1}"
export FREQFLOW_NUM_WORKERS="${FREQFLOW_NUM_WORKERS:-4}"
export FREQFLOW_EPOCHS="${FREQFLOW_EPOCHS:-30}"
export FREQFLOW_LR="${FREQFLOW_LR:-1e-5}"
export FREQFLOW_WEIGHT_DECAY=0.0
export FREQFLOW_WARMUP_STEPS="${FREQFLOW_WARMUP_STEPS:-200}"
export FREQFLOW_EMA_RATE=0.95
export FREQFLOW_EMA_UPDATE_EVERY=20
export FREQFLOW_MIXED_PRECISION=bf16
export FREQFLOW_SAMPLE_STEPS=20
export FREQFLOW_DECODE_BATCH_SIZE=20
export FREQFLOW_LOG_EVERY=20
export FREQFLOW_USE_WANDB=1
export FREQFLOW_WANDB_PROJECT=freqflow-meteo-vae32-horizon-frequency
export FREQFLOW_WANDB_GROUP=meteo-vae32-horizon-frequency
export FREQFLOW_VALIDATE=1
export FREQFLOW_VALID_LIMIT="${FREQFLOW_VALID_LIMIT:-0}"
export FREQFLOW_VLNUM="${FREQFLOW_VLNUM:-10}"
export FREQFLOW_FINAL_TEST_CHECKPOINT="${FREQFLOW_FINAL_TEST_CHECKPOINT:-best}"
export FREQFLOW_RUN_FINAL_TEST="${FREQFLOW_RUN_FINAL_TEST:-1}"
export FREQFLOW_SEED=0

# First experiment: condition the gate on flow time, forecast lead and history.
export FREQFLOW_HORIZON_FREQUENCY_ROUTER=1
export FREQFLOW_FREQUENCY_ROUTER_MODE="${FREQFLOW_FREQUENCY_ROUTER_MODE:-time_lead_history}"

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec bash "${SCRIPT_DIR}/_train_manager_common.sh" "$@"
