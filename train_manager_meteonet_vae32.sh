#!/usr/bin/env bash
# Train the standalone Meteo 128 -> 4x32x32 radar VAE.
# Edit the exports below for a different server, GPU count, or data path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FREQFLOW_PROJECT_DIR="${FREQFLOW_PROJECT_DIR:-${SCRIPT_DIR}}"
export FREQFLOW_DATA_PATH="${FREQFLOW_DATA_PATH:-/mnt/sdb1/user/WRF/stlP/data/meteo_radar.h5}"
export FREQFLOW_RUN_GROUP="${FREQFLOW_RUN_GROUP:-meteo-vae32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FREQFLOW_NUM_PROCESSES="${FREQFLOW_NUM_PROCESSES:-1}"
export FREQFLOW_VAE_BATCH_SIZE="${FREQFLOW_VAE_BATCH_SIZE:-3}"
export FREQFLOW_NUM_WORKERS="${FREQFLOW_NUM_WORKERS:-12}"
export FREQFLOW_VAE_EPOCHS="${FREQFLOW_VAE_EPOCHS:-250}"
export FREQFLOW_VAE_LR="${FREQFLOW_VAE_LR:-1e-4}"
export FREQFLOW_VAE_WEIGHT_DECAY="${FREQFLOW_VAE_WEIGHT_DECAY:-1e-5}"
export FREQFLOW_VAE_WARMUP_RATIO="${FREQFLOW_VAE_WARMUP_RATIO:-0.2}"
export FREQFLOW_VAE_WARMUP_MIN_LR_RATIO="${FREQFLOW_VAE_WARMUP_MIN_LR_RATIO:-0.1}"
export FREQFLOW_VAE_MIN_LR_RATIO="${FREQFLOW_VAE_MIN_LR_RATIO:-1e-3}"
export FREQFLOW_VAE_KL_WEIGHT="${FREQFLOW_VAE_KL_WEIGHT:-1e-4}"
export FREQFLOW_VAE_DISC_WEIGHT="${FREQFLOW_VAE_DISC_WEIGHT:-0.5}"
export FREQFLOW_VAE_DISC_START_EPOCH="${FREQFLOW_VAE_DISC_START_EPOCH:-35}"
# Keep VAE pretraining in FP32 first.  Change to bf16 only after the FP32 smoke run.
export FREQFLOW_MIXED_PRECISION="${FREQFLOW_MIXED_PRECISION:-no}"
export FREQFLOW_VAE_VAL_LIMIT="${FREQFLOW_VAE_VAL_LIMIT:-0}"
export FREQFLOW_LOG_EVERY="${FREQFLOW_LOG_EVERY:-50}"
export FREQFLOW_USE_WANDB="${FREQFLOW_USE_WANDB:-1}"
export FREQFLOW_WANDB_PROJECT="${FREQFLOW_WANDB_PROJECT:-freqflow-meteo-vae32}"
export FREQFLOW_WANDB_GROUP="${FREQFLOW_WANDB_GROUP:-meteo-vae32}"
export FREQFLOW_SEED="${FREQFLOW_SEED:-0}"
export PYTHONUNBUFFERED=1

# Required by Accelerate on the RTX 4000-series server used for this project.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

PROJECT_DIR="${FREQFLOW_PROJECT_DIR}"
DATA_PATH="${FREQFLOW_DATA_PATH}"
RUNS_DIR="${FREQFLOW_RUNS_DIR:-${PROJECT_DIR}/runs}"
RUN_GROUP="${FREQFLOW_RUN_GROUP}"
LATEST_FILE="${RUNS_DIR}/meteo_vae32_latest_run.txt"
NUM_PROCESSES="${FREQFLOW_NUM_PROCESSES}"

die() { echo "Error: $*" >&2; exit 1; }

make_command() {
    local run_dir="$1"
    VAE_COMMAND=(
        accelerate launch --num_processes "${NUM_PROCESSES}" train_vae_meteo.py
        --data-path "${DATA_PATH}" --workdir "${run_dir}"
        --img-size 128 --latent-channels 4
        --batch-size "${FREQFLOW_VAE_BATCH_SIZE}" --num-workers "${FREQFLOW_NUM_WORKERS}"
        --epochs "${FREQFLOW_VAE_EPOCHS}" --lr "${FREQFLOW_VAE_LR}"
        --weight-decay "${FREQFLOW_VAE_WEIGHT_DECAY}" --warmup-ratio "${FREQFLOW_VAE_WARMUP_RATIO}"
        --warmup-min-lr-ratio "${FREQFLOW_VAE_WARMUP_MIN_LR_RATIO}" --min-lr-ratio "${FREQFLOW_VAE_MIN_LR_RATIO}"
        --kl-weight "${FREQFLOW_VAE_KL_WEIGHT}" --disc-weight "${FREQFLOW_VAE_DISC_WEIGHT}"
        --disc-start-epoch "${FREQFLOW_VAE_DISC_START_EPOCH}"
        --mixed-precision "${FREQFLOW_MIXED_PRECISION}" --val-limit "${FREQFLOW_VAE_VAL_LIMIT}"
        --log-every "${FREQFLOW_LOG_EVERY}" --seed "${FREQFLOW_SEED}"
    )
    if [[ "${FREQFLOW_USE_WANDB}" == "1" ]]; then
        VAE_COMMAND+=(--use-wandb --wandb-project "${FREQFLOW_WANDB_PROJECT}" --wandb-group "${FREQFLOW_WANDB_GROUP}")
    fi
}

start() {
    [[ -f "${DATA_PATH}" ]] || die "Meteo HDF5 not found: ${DATA_PATH}"
    local run_dir="${RUNS_DIR}/${RUN_GROUP}/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${run_dir}/checkpoints" "${run_dir}/logs"
    make_command "${run_dir}"
    printf '%s\n' "${run_dir}" > "${LATEST_FILE}"
    printf '%q ' "${VAE_COMMAND[@]}" > "${run_dir}/command.txt"
    printf '\n' >> "${run_dir}/command.txt"
    (
        cd "${PROJECT_DIR}"
        nohup "${VAE_COMMAND[@]}" > "${run_dir}/logs/train.log" 2>&1 &
        echo $! > "${run_dir}/train.pid"
    )
    echo "Started VAE run: ${run_dir}"
    echo "Monitor: tail -f ${run_dir}/logs/train.log"
}

resume() {
    local checkpoint="${1:-}"
    if [[ -z "${checkpoint}" ]]; then
        [[ -f "${LATEST_FILE}" ]] || die "no previous VAE run recorded"
        checkpoint="$(<"${LATEST_FILE}")/checkpoints/last.pt"
    fi
    [[ -f "${checkpoint}" ]] || die "checkpoint not found: ${checkpoint}"
    local run_dir
    run_dir="$(cd -- "$(dirname -- "${checkpoint}")/.." && pwd)"
    make_command "${run_dir}"
    VAE_COMMAND+=(--resume "${checkpoint}")
    (
        cd "${PROJECT_DIR}"
        nohup "${VAE_COMMAND[@]}" >> "${run_dir}/logs/train.log" 2>&1 &
        echo $! > "${run_dir}/train.pid"
    )
    echo "Resumed VAE run: ${run_dir}"
}

status() {
    [[ -f "${LATEST_FILE}" ]] || die "no previous VAE run recorded"
    local run_dir
    run_dir="$(<"${LATEST_FILE}")"
    echo "Run: ${run_dir}"
    [[ -f "${run_dir}/logs/train.log" ]] && tail -n 30 "${run_dir}/logs/train.log"
}

case "${1:-}" in
    start) start ;;
    resume) resume "${2:-}" ;;
    status) status ;;
    dir) [[ -f "${LATEST_FILE}" ]] && cat "${LATEST_FILE}" || die "no previous VAE run recorded" ;;
    *) echo "Usage: bash $(basename "$0") [start|resume [checkpoint]|status|dir]"; exit 1 ;;
esac
