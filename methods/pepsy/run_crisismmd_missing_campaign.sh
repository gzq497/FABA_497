#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/pepsy"
mkdir -p "${LOG_DIR}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

Q_VALUES=(${Q_VALUES:-0.1 0.2 0.3 0.4 0.5})
ALPHA="${ALPHA:-5.0}"
SAMPLE_RATE="${SAMPLE_RATE:-0.1}"
LR="${LR:-0.05}"
EPOCHS="${EPOCHS:-200}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HID_SIZE="${HID_SIZE:-128}"
TEST_FREQUENCY="${TEST_FREQUENCY:-1}"

PEPSY_POOL_SIZE="${PEPSY_POOL_SIZE:-10}"
PEPSY_TOP_K="${PEPSY_TOP_K:-3}"
PEPSY_CONTRASTIVE_WEIGHT="${PEPSY_CONTRASTIVE_WEIGHT:-0.2}"
PEPSY_PROMPT_REG_WEIGHT="${PEPSY_PROMPT_REG_WEIGHT:-0.1}"
PEPSY_TEMPERATURE="${PEPSY_TEMPERATURE:-0.2}"
PEPSY_PROMPT_DROPOUT="${PEPSY_PROMPT_DROPOUT:-0.0}"

for q in "${Q_VALUES[@]}"; do
  q_tag="${q/./}"
  log_path="${LOG_DIR}/crisismmd_pepsy_q${q_tag}_${RUN_TS}.log"
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
    --missing_strategy pepsy \
    --pepsy_pool_size "${PEPSY_POOL_SIZE}" \
    --pepsy_top_k "${PEPSY_TOP_K}" \
    --pepsy_contrastive_weight "${PEPSY_CONTRASTIVE_WEIGHT}" \
    --pepsy_prompt_reg_weight "${PEPSY_PROMPT_REG_WEIGHT}" \
    --pepsy_temperature "${PEPSY_TEMPERATURE}" \
    --pepsy_prompt_dropout "${PEPSY_PROMPT_DROPOUT}" \
    > "${log_path}" 2>&1
  echo "[INFO] strategy=pepsy q=${q} finished, log=${log_path}"
done
