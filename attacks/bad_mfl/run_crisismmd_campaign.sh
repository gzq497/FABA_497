#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/bad_mfl"
mkdir -p "${LOG_DIR}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

ALPHAS=(${ALPHAS:-5.0})
TARGET_LABEL="${TARGET_LABEL:-4}"
MALICIOUS_RATIO="${MALICIOUS_RATIO:-0.1}"
POISON_RATIO="${POISON_RATIO:-0.1}"
EPOCHS="${EPOCHS:-200}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LR="${LR:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HID_SIZE="${HID_SIZE:-128}"
SAMPLE_RATE="${SAMPLE_RATE:-0.1}"
SEED="${SEED:-2024}"
GEN_EPOCHS="${GEN_EPOCHS:-2}"
SUPPORT_SIZE="${SUPPORT_SIZE:-64}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
FORCE_PREPARE="${FORCE_PREPARE:-1}"
RUN_CLEAN="${RUN_CLEAN:-1}"
RUN_ATTACK="${RUN_ATTACK:-1}"
ATTACK_WEIGHT="${ATTACK_WEIGHT:-}"
if [[ -z "${ATTACK_WEIGHT}" ]]; then
  ATTACK_WEIGHT="${POISON_RATIO}"
fi

for alpha in "${ALPHAS[@]}"; do
  alpha_tag="$(printf '%s' "${alpha}" | tr -d '.')"
  mr_tag="$(printf '%s' "${MALICIOUS_RATIO}" | tr -d '.')"
  pr_tag="$(printf '%s' "${POISON_RATIO}" | tr -d '.')"
  prepare_log="${LOG_DIR}/bad_mfl_prepare_alpha${alpha_tag}_mr${mr_tag}_pr${pr_tag}_tl${TARGET_LABEL}_${RUN_TS}.log"

  prepare_args=(
    --alpha "${alpha}"
    --bd_target_label "${TARGET_LABEL}"
    --bd_malicious_ratio "${MALICIOUS_RATIO}"
    --bd_poison_ratio "${POISON_RATIO}"
    --bd_seed "${SEED}"
    --bad_mfl_allow_remote_model_download
    --bad_mfl_use_t5
    --bad_mfl_support_size "${SUPPORT_SIZE}"
    --bad_mfl_gen_epochs "${GEN_EPOCHS}"
    --bad_mfl_gen_batch_size "${GEN_BATCH_SIZE}"
  )
  if [[ "${FORCE_PREPARE}" == "1" ]]; then
    prepare_args+=(--bad_mfl_force_prepare)
  else
    prepare_args+=(--bad_mfl_use_cached_generators)
  fi

  "${PYTHON_BIN}" attacks/bad_mfl/prepare_crisismmd.py "${prepare_args[@]}" > "${prepare_log}" 2>&1

  clean_log="${LOG_DIR}/bad_mfl_clean_alpha${alpha_tag}_base_${RUN_TS}.log"
  attack_log="${LOG_DIR}/bad_mfl_alpha${alpha_tag}_base_${RUN_TS}.log"

  if [[ "${RUN_CLEAN}" == "1" ]]; then
    "${PYTHON_BIN}" experiment/crisis-mmd/train.py \
      --alpha "${alpha}" \
      --sample_rate "${SAMPLE_RATE}" \
      --learning_rate "${LR}" \
      --hid_size "${HID_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --local_epochs "${LOCAL_EPOCHS}" \
      --num_epochs "${EPOCHS}" \
      --test_frequency 1 \
      --en_eval_attack \
      --attack_type bad_mfl \
      --bd_target_label "${TARGET_LABEL}" \
      --bd_malicious_ratio "${MALICIOUS_RATIO}" \
      --bd_poison_ratio "${POISON_RATIO}" \
      --bd_seed "${SEED}" \
      --bad_mfl_attack_weight "${ATTACK_WEIGHT}" \
      --bad_mfl_paper_dropout 0.2 \
      --bad_mfl_paper_heads 6 > "${clean_log}" 2>&1
  fi

  if [[ "${RUN_ATTACK}" == "1" ]]; then
    "${PYTHON_BIN}" experiment/crisis-mmd/train.py \
      --alpha "${alpha}" \
      --sample_rate "${SAMPLE_RATE}" \
      --learning_rate "${LR}" \
      --hid_size "${HID_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --local_epochs "${LOCAL_EPOCHS}" \
      --num_epochs "${EPOCHS}" \
      --test_frequency 1 \
      --en_backdoor_attack \
      --attack_type bad_mfl \
      --bd_target_label "${TARGET_LABEL}" \
      --bd_malicious_ratio "${MALICIOUS_RATIO}" \
      --bd_poison_ratio "${POISON_RATIO}" \
      --bd_seed "${SEED}" \
      --bad_mfl_attack_weight "${ATTACK_WEIGHT}" \
      --bad_mfl_paper_dropout 0.2 \
      --bad_mfl_paper_heads 6 > "${attack_log}" 2>&1
  fi
done
