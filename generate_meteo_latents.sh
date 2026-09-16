#!/usr/bin/env bash
# Generate a static latent HDF5 after the Meteo VAE has passed reconstruction validation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FREQFLOW_DATA_PATH=/mnt/sdb1/user/WRF/stlP/data/meteo_radar.h5
# Set this to the selected VAE checkpoint before running this script.
export FREQFLOW_VAE_CHECKPOINT="${FREQFLOW_VAE_CHECKPOINT:-}"
export FREQFLOW_LATENT_PATH="${FREQFLOW_LATENT_PATH:-${SCRIPT_DIR}/data/meteo_latent_vae32.h5}"
export FREQFLOW_ENCODE_BATCH_SIZE="${FREQFLOW_ENCODE_BATCH_SIZE:-32}"
export FREQFLOW_ENCODE_WORKERS="${FREQFLOW_ENCODE_WORKERS:-2}"

[[ -f "${FREQFLOW_DATA_PATH}" ]] || { echo "Meteo HDF5 not found: ${FREQFLOW_DATA_PATH}" >&2; exit 1; }
[[ -n "${FREQFLOW_VAE_CHECKPOINT}" && -f "${FREQFLOW_VAE_CHECKPOINT}" ]] || {
    echo "Set FREQFLOW_VAE_CHECKPOINT to checkpoints/best_val_loss.pt before generating latents." >&2; exit 1;
}

cd "${SCRIPT_DIR}"
exec python generate_meteo_latents.py \
    --data-path "${FREQFLOW_DATA_PATH}" \
    --vae-checkpoint "${FREQFLOW_VAE_CHECKPOINT}" \
    --output-path "${FREQFLOW_LATENT_PATH}" \
    --batch-size 2 \
    --encode-batch-size "${FREQFLOW_ENCODE_BATCH_SIZE}" \
    --num-workers "${FREQFLOW_ENCODE_WORKERS}" \
    "$@"
