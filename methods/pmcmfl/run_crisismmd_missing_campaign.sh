#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/pmcmfl"
mkdir -p "${LOG_DIR}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

Q_VALUES=(${Q_VALUES:-0.1 0.2 0.3 0.4 0.5})
STRATEGIES=(${STRATEGIES:-pmcmfl})
ALPHA="${ALPHA:-5.0}"
SAMPLE_RATE="${SAMPLE_RATE:-0.1}"
LR="${LR:-0.05}"
EPOCHS="${EPOCHS:-200}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HID_SIZE="${HID_SIZE:-128}"
TEST_FREQUENCY="${TEST_FREQUENCY:-1}"

PMCMFL_REG_START="${PMCMFL_REG_START:-2}"
PMCMFL_TEMPERATURE="${PMCMFL_TEMPERATURE:-0.5}"
PMCMFL_IMG_WEIGHT="${PMCMFL_IMG_WEIGHT:-1e-8}"
PMCMFL_TEXT_WEIGHT="${PMCMFL_TEXT_WEIGHT:-1e-8}"
PMCMFL_FUSION_WEIGHT="${PMCMFL_FUSION_WEIGHT:-0.01}"
PMCMFL_PROTO_INIT_STD="${PMCMFL_PROTO_INIT_STD:-0.0}"

for strategy in "${STRATEGIES[@]}"; do
  for q in "${Q_VALUES[@]}"; do
    q_tag="${q/./}"
    log_path="${LOG_DIR}/crisismmd_${strategy}_q${q_tag}_${RUN_TS}.log"
    "${PYTHON_BIN}" -u experiment/crisis-mmd/train.py \
      --alpha "${ALPHA}" \
      --sample_rate "${SAMPLE_RATE}" \
      --learning_rate "${LR}" \
      --hid_size "${HID_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --local_epochs "${LOCAL_EPOCHS}" \
      --num_epochs "${EPOCHS}" \
      --test_frequency "${TEST_FREQUENCY}" \
      --en_missing_modality \
      --missing_modailty_rate "${q}" \
      --missing_strategy "${strategy}" \
      --pmcmfl_regularization_start_epoch "${PMCMFL_REG_START}" \
      --pmcmfl_temperature "${PMCMFL_TEMPERATURE}" \
      --pmcmfl_img_loss_weight "${PMCMFL_IMG_WEIGHT}" \
      --pmcmfl_text_loss_weight "${PMCMFL_TEXT_WEIGHT}" \
      --pmcmfl_fusion_loss_weight "${PMCMFL_FUSION_WEIGHT}" \
      --pmcmfl_prototype_init_std "${PMCMFL_PROTO_INIT_STD}" \
      > "${log_path}" 2>&1
    echo "[INFO] strategy=${strategy} q=${q} finished, log=${log_path}"
  done
done
