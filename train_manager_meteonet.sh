#!/usr/bin/env bash
# FreqFlow MeteoNet 5 -> 20 training manager.
# Edit the values below only when this server's paths or hardware change.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FREQFLOW_PROJECT_DIR="${SCRIPT_DIR}"
export FREQFLOW_MANAGER_NAME="$(basename -- "$0")"
export FREQFLOW_DATASET=meteo
export FREQFLOW_DATA_PATH=/mnt/sdb1/user/WRF/stlP/data/meteo_radar.h5
export FREQFLOW_RUN_GROUP=meteo-5to20
export CUDA_VISIBLE_DEVICES=0
export FREQFLOW_NUM_PROCESSES=1
export FREQFLOW_BATCH_SIZE=1
export FREQFLOW_NUM_WORKERS=4
export FREQFLOW_EPOCHS=100
export FREQFLOW_LR=1e-4
export FREQFLOW_WEIGHT_DECAY=0.0
export FREQFLOW_WARMUP_STEPS=1000
export FREQFLOW_EMA_RATE=0.95
export FREQFLOW_EMA_UPDATE_EVERY=20
export FREQFLOW_MIXED_PRECISION=fp16
export FREQFLOW_SAMPLE_STEPS=20
export FREQFLOW_VALIDATE=1
export FREQFLOW_VALID_LIMIT=0
export FREQFLOW_VLNUM=10
export FREQFLOW_SEED=0

exec bash "${SCRIPT_DIR}/_train_manager_common.sh" "$@"
