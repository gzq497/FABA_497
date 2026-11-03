#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/bad_mfl"
mkdir -p "${LOG_DIR}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

MALICIOUS_RATIO="${MALICIOUS_RATIO:-0.1}"
POISON_RATIO="${POISON_RATIO:-0.1}"
TARGET_LABEL="${TARGET_LABEL:-4}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-200}"
SAMPLE_RATE="${SAMPLE_RATE:-0.1}"
LR="${LR:-0.05}"
HID_SIZE="${HID_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
TEST_FREQUENCY="${TEST_FREQUENCY:-1}"
ATTACK_WEIGHT="${ATTACK_WEIGHT:-${POISON_RATIO}}"
ALPHA="${ALPHA:-5.0}"
ALPHA_TAG="${ALPHA/./}"

for mm in 0.1 0.2 0.3 0.4 0.5; do
  rate_tag=${mm/./}
  log_path="${LOG_DIR}/bad_mfl_alpha${ALPHA_TAG}_mm${rate_tag}_${RUN_TS}.log"
  "${PYTHON_BIN}" experiment/crisis-mmd/train.py \
    --alpha "${ALPHA}" \
    --sample_rate "${SAMPLE_RATE}" \
    --learning_rate "${LR}" \
    --hid_size "${HID_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --num_epochs "${EPOCHS}" \
    --test_frequency "${TEST_FREQUENCY}" \
    --en_missing_modality \
    --missing_modailty_rate "${mm}" \
    --en_backdoor_attack \
    --attack_type bad_mfl \
    --bd_target_label "${TARGET_LABEL}" \
    --bd_malicious_ratio "${MALICIOUS_RATIO}" \
    --bd_poison_ratio "${POISON_RATIO}" \
    --bd_seed "${SEED}" \
    --bad_mfl_attack_weight "${ATTACK_WEIGHT}" \
    --bad_mfl_paper_dropout 0.2 \
    --bad_mfl_paper_heads 6 > "${log_path}" 2>&1
done
