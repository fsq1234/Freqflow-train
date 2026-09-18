#!/usr/bin/env bash
# Shared implementation for the SEVIR and MeteoNet FreqFlow launchers.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${FREQFLOW_PROJECT_DIR:-${SCRIPT_DIR}}"
MANAGER_NAME="${FREQFLOW_MANAGER_NAME:-$(basename -- "$0")}"
DATASET="${FREQFLOW_DATASET:?Set FREQFLOW_DATASET in a dataset launcher.}"
DATA_PATH="${FREQFLOW_DATA_PATH:?Set FREQFLOW_DATA_PATH in a dataset launcher.}"
RAW_DATA_PATH="${FREQFLOW_RAW_DATA_PATH:-}"
VAE_CHECKPOINT="${FREQFLOW_VAE_CHECKPOINT:-}"
RUN_GROUP="${FREQFLOW_RUN_GROUP:-${DATASET}-5to20}"
RUNS_DIR="${FREQFLOW_RUNS_DIR:-${PROJECT_DIR}/runs}"
ACCELERATE_BIN="${FREQFLOW_ACCELERATE_BIN:-accelerate}"
NUM_PROCESSES="${FREQFLOW_NUM_PROCESSES:-1}"
IMG_SIZE="${FREQFLOW_IMG_SIZE:-128}"
FRAMES_IN="${FREQFLOW_FRAMES_IN:-5}"
FRAMES_OUT="${FREQFLOW_FRAMES_OUT:-20}"
STRIDE="${FREQFLOW_STRIDE:-13}"
BATCH_SIZE="${FREQFLOW_BATCH_SIZE:-1}"
NUM_WORKERS="${FREQFLOW_NUM_WORKERS:-4}"
EPOCHS="${FREQFLOW_EPOCHS:-100}"
LR="${FREQFLOW_LR:-1e-4}"
WEIGHT_DECAY="${FREQFLOW_WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${FREQFLOW_WARMUP_STEPS:-1000}"
EMA_RATE="${FREQFLOW_EMA_RATE:-0.95}"
EMA_UPDATE_EVERY="${FREQFLOW_EMA_UPDATE_EVERY:-20}"
SAMPLE_STEPS="${FREQFLOW_SAMPLE_STEPS:-20}"
DECODE_BATCH_SIZE="${FREQFLOW_DECODE_BATCH_SIZE:-20}"
LOG_EVERY="${FREQFLOW_LOG_EVERY:-20}"
USE_WANDB="${FREQFLOW_USE_WANDB:-1}"
WANDB_PROJECT="${FREQFLOW_WANDB_PROJECT:-freqflow-nowcast}"
WANDB_GROUP="${FREQFLOW_WANDB_GROUP:-${RUN_GROUP}}"
WANDB_NAME="${FREQFLOW_WANDB_NAME:-}"
MIXED_PRECISION="${FREQFLOW_MIXED_PRECISION:-fp16}"
SEED="${FREQFLOW_SEED:-0}"
VALIDATE="${FREQFLOW_VALIDATE:-1}"
VALID_LIMIT="${FREQFLOW_VALID_LIMIT:-0}"
VLNUM="${FREQFLOW_VLNUM:-10}"
FINAL_TEST_CHECKPOINT="${FREQFLOW_FINAL_TEST_CHECKPOINT:-last}"
RUN_FINAL_TEST="${FREQFLOW_RUN_FINAL_TEST:-1}"
HORIZON_FREQUENCY_ROUTER="${FREQFLOW_HORIZON_FREQUENCY_ROUTER:-0}"
FREQUENCY_ROUTER_MODE="${FREQFLOW_FREQUENCY_ROUTER_MODE:-time_lead_history}"
FREQUENCY_ROUTER_LR="${FREQFLOW_FREQUENCY_ROUTER_LR:-}"
INIT_CHECKPOINT="${FREQFLOW_INIT_CHECKPOINT:-}"
INIT_CHECKPOINT_STATE="${FREQFLOW_INIT_CHECKPOINT_STATE:-ema}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_DIR}/${RUN_GROUP}/${TIMESTAMP}"
LATEST_RUN_FILE="${RUNS_DIR}/${DATASET}_latest_run.txt"
PID_FILE="${RUN_DIR}/train.pid"
LOG_FILE="${RUN_DIR}/logs/train.log"
declare -a LAUNCH_CMD

die() {
    echo -e "${RED}Error: $*${NC}" >&2
    exit 1
}

validate_setup() {
    [[ -f "${PROJECT_DIR}/train_nowcast_alphapre.py" ]] || die "not a FreqFlow project: ${PROJECT_DIR}"
    case "${DATASET}" in
        sevir)
            die "this VAE branch currently has a generated latent contract only for MeteoNet"
            ;;
        meteo)
            [[ -f "${DATA_PATH}" ]] || die "MeteoNet latent HDF5 not found: ${DATA_PATH}"
            [[ -f "${RAW_DATA_PATH}" ]] || die "MeteoNet raw HDF5 not found: ${RAW_DATA_PATH}"
            [[ -f "${VAE_CHECKPOINT}" ]] || die "Meteo VAE checkpoint not found: ${VAE_CHECKPOINT}"
            if [[ -n "${INIT_CHECKPOINT}" ]]; then
                [[ -f "${INIT_CHECKPOINT}" ]] || die "initial checkpoint not found: ${INIT_CHECKPOINT}"
            fi
            ;;
        *)
            die "unsupported dataset: ${DATASET}"
            ;;
    esac
}

load_latest_run() {
    [[ -f "${LATEST_RUN_FILE}" ]] || die "no recorded ${DATASET} run; use start first"
    RUN_DIR="$(<"${LATEST_RUN_FILE}")"
    PID_FILE="${RUN_DIR}/train.pid"
    LOG_FILE="${RUN_DIR}/logs/train.log"
}

make_launch_command() {
    LAUNCH_CMD=(
        "${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}"
        train_nowcast_alphapre.py
        --dataset "${DATASET}"
        --data-path "${DATA_PATH}"
        --raw-data-path "${RAW_DATA_PATH}"
        --vae-checkpoint "${VAE_CHECKPOINT}"
        --workdir "${RUN_DIR}"
        --img-size "${IMG_SIZE}"
        --stride "${STRIDE}"
        --frames-in "${FRAMES_IN}"
        --frames-out "${FRAMES_OUT}"
        --batch-size "${BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --epochs "${EPOCHS}"
        --lr "${LR}"
        --weight-decay "${WEIGHT_DECAY}"
        --warmup-steps "${WARMUP_STEPS}"
        --ema-rate "${EMA_RATE}"
        --ema-update-every "${EMA_UPDATE_EVERY}"
        --sample-steps "${SAMPLE_STEPS}"
        --decode-batch-size "${DECODE_BATCH_SIZE}"
        --log-every "${LOG_EVERY}"
        --mixed-precision "${MIXED_PRECISION}"
        --seed "${SEED}"
        --final-test-checkpoint "${FINAL_TEST_CHECKPOINT}"
    )
    if [[ "${VALIDATE}" == "1" ]]; then
        LAUNCH_CMD+=(--valid)
    fi
    if [[ "${VALID_LIMIT}" == "1" ]]; then
        LAUNCH_CMD+=(--valid-limit --vlnum "${VLNUM}")
    fi
    if [[ "${USE_WANDB}" == "1" ]]; then
        local wandb_name
        if [[ -n "${WANDB_NAME}" ]]; then
            wandb_name="${WANDB_NAME}"
        else
            wandb_name="${RUN_GROUP}-$(basename -- "${RUN_DIR}")"
        fi
        LAUNCH_CMD+=(
            --use-wandb
            --wandb-project "${WANDB_PROJECT}"
            --wandb-name "${wandb_name}"
            --wandb-group "${WANDB_GROUP}"
        )
    fi
    if [[ "${RUN_FINAL_TEST}" != "1" ]]; then
        LAUNCH_CMD+=(--skip-final-test)
    fi
    if [[ "${HORIZON_FREQUENCY_ROUTER}" == "1" ]]; then
        LAUNCH_CMD+=(--horizon-frequency-router --frequency-router-mode "${FREQUENCY_ROUTER_MODE}")
        if [[ -n "${FREQUENCY_ROUTER_LR}" ]]; then
            LAUNCH_CMD+=(--frequency-router-lr "${FREQUENCY_ROUTER_LR}")
        fi
    fi
    if [[ -n "${INIT_CHECKPOINT}" ]]; then
        LAUNCH_CMD+=(--init-checkpoint "${INIT_CHECKPOINT}" --init-checkpoint-state "${INIT_CHECKPOINT_STATE}")
    fi
    return 0
}

write_run_config() {
    {
        echo "FreqFlow training configuration"
        echo "==============================="
        echo "Time: $(date)"
        echo "Dataset: ${DATASET}"
        echo "Data path: ${DATA_PATH}"
        echo "Raw metric data: ${RAW_DATA_PATH}"
        echo "Frozen VAE: ${VAE_CHECKPOINT}"
        echo "Project: ${PROJECT_DIR}"
        echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
        echo "Processes: ${NUM_PROCESSES}"
        echo "Frames: ${FRAMES_IN} input + ${FRAMES_OUT} future"
        echo "VAE decode batch size: ${DECODE_BATCH_SIZE}"
        echo "Batch size per process: ${BATCH_SIZE}"
        echo "Mixed precision: ${MIXED_PRECISION}"
        echo "Progress log interval: ${LOG_EVERY} steps"
        echo "W&B: ${USE_WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"
        echo "Automatic full test: ${RUN_FINAL_TEST} (checkpoint=${FINAL_TEST_CHECKPOINT}.pt EMA)"
        echo "Horizon frequency router: ${HORIZON_FREQUENCY_ROUTER} (${FREQUENCY_ROUTER_MODE})"
        echo "Frequency router learning rate: ${FREQUENCY_ROUTER_LR:-same as backbone}"
        echo "Initialization checkpoint: ${INIT_CHECKPOINT:-none} (${INIT_CHECKPOINT_STATE})"
        echo ""
        printf 'Command: '
        printf '%q ' "${LAUNCH_CMD[@]}"
        printf '\n'
    } > "${RUN_DIR}/config.txt"
}

start_training() {
    validate_setup
    mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUNS_DIR}"
    make_launch_command
    write_run_config
    printf '%s\n' "${RUN_DIR}" > "${LATEST_RUN_FILE}"

    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  FreqFlow ${DATASET} training${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo "Run dir: ${RUN_DIR}"
    echo "Data path: ${DATA_PATH}"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}; processes: ${NUM_PROCESSES}"
    echo "Batch size per process: ${BATCH_SIZE}"
    echo "Log: ${LOG_FILE}"

    cd "${PROJECT_DIR}"
    nohup "${LAUNCH_CMD[@]}" > "${LOG_FILE}" 2>&1 &
    local pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"

    echo -e "${GREEN}Started PID: ${pid}${NC}"
    echo "Monitor: tail -f ${LOG_FILE}"
    echo "Status:  bash ${MANAGER_NAME} status"
}

smoke_training() {
    EPOCHS=1
    VALIDATE=1
    VALID_LIMIT=1
    VLNUM=2
    RUN_FINAL_TEST=0
    RUN_DIR="${RUNS_DIR}/${RUN_GROUP}-smoke/${TIMESTAMP}"
    PID_FILE="${RUN_DIR}/train.pid"
    LOG_FILE="${RUN_DIR}/logs/train.log"
    start_training
}

resolve_resume_target() {
    local requested_target="${1:-}"
    if [[ -z "${requested_target}" ]]; then
        load_latest_run
        RESUME_CHECKPOINT="${RUN_DIR}/checkpoints/last.pt"
    elif [[ -d "${requested_target}" ]]; then
        RUN_DIR="$(cd -- "${requested_target}" && pwd)"
        RESUME_CHECKPOINT="${RUN_DIR}/checkpoints/last.pt"
    elif [[ -f "${requested_target}" ]]; then
        RESUME_CHECKPOINT="$(cd -- "$(dirname -- "${requested_target}")" && pwd)/$(basename -- "${requested_target}")"
        RUN_DIR="$(cd -- "$(dirname -- "${RESUME_CHECKPOINT}")/.." && pwd)"
    else
        die "run directory or checkpoint not found: ${requested_target}"
    fi
    [[ -f "${RESUME_CHECKPOINT}" ]] || die "resume checkpoint not found: ${RESUME_CHECKPOINT}"
    PID_FILE="${RUN_DIR}/train.pid"
    LOG_FILE="${RUN_DIR}/logs/train.log"
}

resume_training() {
    validate_setup
    resolve_resume_target "${1:-}"
    mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUNS_DIR}"
    make_launch_command
    LAUNCH_CMD+=(--resume "${RESUME_CHECKPOINT}")
    printf '%s\n' "${RUN_DIR}" > "${LATEST_RUN_FILE}"
    {
        echo
        echo "Resume time: $(date)"
        echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
    } >> "${RUN_DIR}/config.txt"

    echo -e "${GREEN}Resuming ${DATASET} from: ${RESUME_CHECKPOINT}${NC}"
    cd "${PROJECT_DIR}"
    nohup "${LAUNCH_CMD[@]}" >> "${LOG_FILE}" 2>&1 &
    local pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"
    echo -e "${GREEN}Resumed PID: ${pid}${NC}"
    echo "Monitor: tail -f ${LOG_FILE}"
}

stop_training() {
    load_latest_run
    [[ -f "${PID_FILE}" ]] || die "no PID file: ${PID_FILE}"
    local pid
    pid="$(<"${PID_FILE}")"
    if ! ps -p "${pid}" > /dev/null 2>&1; then
        echo -e "${YELLOW}PID ${pid} is not running.${NC}"
        return
    fi
    local command
    command="$(ps -p "${pid}" -o args=)"
    [[ "${command}" == *"train_nowcast_alphapre.py"* ]] || die "PID ${pid} does not look like this training job: ${command}"
    echo -e "${YELLOW}Stopping PID ${pid}${NC}"
    kill "${pid}"
}

show_status() {
    load_latest_run
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  FreqFlow ${DATASET} status${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo "Run dir: ${RUN_DIR}"
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(<"${PID_FILE}")"
        if ps -p "${pid}" > /dev/null 2>&1; then
            echo -e "${GREEN}Running, PID: ${pid}${NC}"
        else
            echo -e "${YELLOW}PID ${pid} is not running.${NC}"
        fi
    else
        echo -e "${YELLOW}No PID file.${NC}"
    fi
    echo "Log: ${LOG_FILE}"
    [[ -f "${LOG_FILE}" ]] && tail -n 20 "${LOG_FILE}"
    echo "Checkpoints:"
    ls -lh "${RUN_DIR}/checkpoints"/*.pt 2>/dev/null || echo "  (none)"
}

usage() {
    echo "Usage: bash ${MANAGER_NAME} [smoke|start|resume [run_dir|checkpoint]|stop|status|dir]"
    echo ""
    echo "Override settings without editing, for example:"
    echo "  CUDA_VISIBLE_DEVICES=0 FREQFLOW_BATCH_SIZE=1 bash ${MANAGER_NAME} start"
}

case "${1:-}" in
    smoke)  smoke_training ;;
    start)  start_training ;;
    resume) resume_training "${2:-}" ;;
    stop)   stop_training ;;
    status) show_status ;;
    dir)    load_latest_run; echo "${RUN_DIR}" ;;
    *)      usage; exit 1 ;;
esac
