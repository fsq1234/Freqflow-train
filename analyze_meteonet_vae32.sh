#!/usr/bin/env bash
# Run pixel-domain diagnostics for a trained Meteo latent-FreqFlow checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FREQFLOW_PROJECT_DIR="${FREQFLOW_PROJECT_DIR:-${SCRIPT_DIR}}"
export FREQFLOW_LATENT_PATH="${FREQFLOW_LATENT_PATH:-${SCRIPT_DIR}/data/meteo_latent_vae32.h5}"
export FREQFLOW_RAW_DATA_PATH="${FREQFLOW_RAW_DATA_PATH:-/mnt/sdb1/user/WRF/stlP/data/meteo_radar.h5}"
# Set both checkpoint paths before starting the diagnostic run.
export FREQFLOW_VAE_CHECKPOINT="${FREQFLOW_VAE_CHECKPOINT:-}"
export FREQFLOW_CHECKPOINT="${FREQFLOW_CHECKPOINT:-}"
export FREQFLOW_CHECKPOINT_STATE="${FREQFLOW_CHECKPOINT_STATE:-ema}"
export FREQFLOW_DIAGNOSTIC_GROUP="${FREQFLOW_DIAGNOSTIC_GROUP:-meteo-vae32-diagnostics}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FREQFLOW_NUM_PROCESSES="${FREQFLOW_NUM_PROCESSES:-1}"
export FREQFLOW_BATCH_SIZE="${FREQFLOW_BATCH_SIZE:-1}"
export FREQFLOW_NUM_WORKERS="${FREQFLOW_NUM_WORKERS:-4}"
export FREQFLOW_SAMPLE_STEPS="${FREQFLOW_SAMPLE_STEPS:-10}"
export FREQFLOW_DECODE_BATCH_SIZE="${FREQFLOW_DECODE_BATCH_SIZE:-20}"
export FREQFLOW_MIXED_PRECISION="${FREQFLOW_MIXED_PRECISION:-bf16}"
export FREQFLOW_LIMIT_BATCHES="${FREQFLOW_LIMIT_BATCHES:-0}"
export FREQFLOW_ORACLE_MAX_SHIFT="${FREQFLOW_ORACLE_MAX_SHIFT:-24}"
export FREQFLOW_SEED="${FREQFLOW_SEED:-0}"
export PYTHONUNBUFFERED=1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

PROJECT_DIR="${FREQFLOW_PROJECT_DIR}"
RUNS_DIR="${FREQFLOW_RUNS_DIR:-${PROJECT_DIR}/runs}"
LATEST_FILE="${RUNS_DIR}/meteo_diagnostics_latest_run.txt"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

die() { echo "Error: $*" >&2; exit 1; }

validate_setup() {
    [[ -f "${PROJECT_DIR}/analyze_nowcast_alphapre.py" ]] || die "diagnostic entrypoint not found"
    [[ -f "${FREQFLOW_LATENT_PATH}" ]] || die "latent HDF5 not found: ${FREQFLOW_LATENT_PATH}"
    [[ -f "${FREQFLOW_RAW_DATA_PATH}" ]] || die "raw Meteo HDF5 not found: ${FREQFLOW_RAW_DATA_PATH}"
    [[ -f "${FREQFLOW_VAE_CHECKPOINT}" ]] || die "VAE checkpoint not found: ${FREQFLOW_VAE_CHECKPOINT}"
    [[ -f "${FREQFLOW_CHECKPOINT}" ]] || die "FreqFlow checkpoint not found: ${FREQFLOW_CHECKPOINT}"
}

make_command() {
    local output_dir="$1"
    DIAGNOSTIC_COMMAND=(
        accelerate launch --num_processes "${FREQFLOW_NUM_PROCESSES}"
        analyze_nowcast_alphapre.py
        --data-path "${FREQFLOW_LATENT_PATH}"
        --raw-data-path "${FREQFLOW_RAW_DATA_PATH}"
        --vae-checkpoint "${FREQFLOW_VAE_CHECKPOINT}"
        --checkpoint "${FREQFLOW_CHECKPOINT}"
        --checkpoint-state "${FREQFLOW_CHECKPOINT_STATE}"
        --output-dir "${output_dir}"
        --batch-size "${FREQFLOW_BATCH_SIZE}"
        --num-workers "${FREQFLOW_NUM_WORKERS}"
        --sample-steps "${FREQFLOW_SAMPLE_STEPS}"
        --decode-batch-size "${FREQFLOW_DECODE_BATCH_SIZE}"
        --mixed-precision "${FREQFLOW_MIXED_PRECISION}"
        --limit-batches "${FREQFLOW_LIMIT_BATCHES}"
        --oracle-max-shift "${FREQFLOW_ORACLE_MAX_SHIFT}"
        --seed "${FREQFLOW_SEED}"
    )
}

start() {
    validate_setup
    local output_dir="${RUNS_DIR}/${FREQFLOW_DIAGNOSTIC_GROUP}/${TIMESTAMP}"
    mkdir -p "${output_dir}/logs"
    make_command "${output_dir}"
    printf '%s\n' "${output_dir}" > "${LATEST_FILE}"
    printf '%q ' "${DIAGNOSTIC_COMMAND[@]}" > "${output_dir}/command.txt"
    printf '\n' >> "${output_dir}/command.txt"
    (
        cd "${PROJECT_DIR}"
        nohup "${DIAGNOSTIC_COMMAND[@]}" > "${output_dir}/logs/diagnostics.log" 2>&1 &
        echo $! > "${output_dir}/diagnostics.pid"
    )
    echo "Started diagnostic run: ${output_dir}"
    echo "Monitor: tail -f ${output_dir}/logs/diagnostics.log"
}

smoke() {
    FREQFLOW_LIMIT_BATCHES=2
    FREQFLOW_DIAGNOSTIC_GROUP="meteo-vae32-diagnostics-smoke"
    start
}

status() {
    [[ -f "${LATEST_FILE}" ]] || die "no diagnostic run recorded"
    local output_dir
    output_dir="$(<"${LATEST_FILE}")"
    echo "Run: ${output_dir}"
    [[ -f "${output_dir}/logs/diagnostics.log" ]] && tail -n 40 "${output_dir}/logs/diagnostics.log"
}

case "${1:-}" in
    start) start ;;
    smoke) smoke ;;
    status) status ;;
    dir) [[ -f "${LATEST_FILE}" ]] && cat "${LATEST_FILE}" || die "no diagnostic run recorded" ;;
    *) echo "Usage: bash $(basename "$0") [smoke|start|status|dir]"; exit 1 ;;
esac
