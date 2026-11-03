#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/attack_test2"
mkdir -p "${LOG_DIR}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

Q_VALUES=(${Q_VALUES:-0.1 0.2 0.3 0.4 0.5})
STRATEGIES=(${STRATEGIES:-zero pmcmfl pepsy})
ALPHA="${ALPHA:-5.0}"
SAMPLE_RATE="${SAMPLE_RATE:-0.1}"
LR="${LR:-0.05}"
EPOCHS="${EPOCHS:-200}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HID_SIZE="${HID_SIZE:-128}"
TEST_FREQUENCY="${TEST_FREQUENCY:-1}"
MALICIOUS_RATIO="${MALICIOUS_RATIO:-0.2}"
POISON_RATIO="${POISON_RATIO:-0.2}"
TARGET_LABEL="${TARGET_LABEL:-0}"
SEED="${SEED:-2024}"

TRIGGER_SIZE="${TRIGGER_SIZE:-16}"
TRIGGER_VALUE="${TRIGGER_VALUE:-10.0}"
IMG_TRIGGER_STD="${IMG_TRIGGER_STD:-0.05}"
IMG_TRIGGER_EPS="${IMG_TRIGGER_EPS:-0.5}"
TEXT_CHARSET="${TEXT_CHARSET:-abcdefghijklmnopqrstuvwxyz0123456789}"
TEXT_TRIGGER_BUDGET="${TEXT_TRIGGER_BUDGET:-5}"
TEXT_BEAM_WIDTH="${TEXT_BEAM_WIDTH:-5}"
TEXT_TRIGGER_SCALE="${TEXT_TRIGGER_SCALE:-0.1}"
TEXT_REFRESH_INTERVAL="${TEXT_REFRESH_INTERVAL:-10}"
LAMBDA_IMG_SHIFT="${LAMBDA_IMG_SHIFT:-0.1}"
LAMBDA_IMG_CLUSTER="${LAMBDA_IMG_CLUSTER:-0.1}"
LAMBDA_TEXT_SHIFT="${LAMBDA_TEXT_SHIFT:-0.1}"
CLIP_TEMPERATURE="${CLIP_TEMPERATURE:-0.07}"
USE_PRETRAINED_CLIP="${USE_PRETRAINED_CLIP:-0}"

TEST2_LAMBDA_CLEAN="${TEST2_LAMBDA_CLEAN:-1.0}"
TEST2_LAMBDA_JOINT="${TEST2_LAMBDA_JOINT:-1.5}"
TEST2_LAMBDA_IMG="${TEST2_LAMBDA_IMG:-1.0}"
TEST2_LAMBDA_TEXT="${TEST2_LAMBDA_TEXT:-1.0}"
TEST2_LAMBDA_AUX="${TEST2_LAMBDA_AUX:-0.5}"
TEST2_LAMBDA_ANTI="${TEST2_LAMBDA_ANTI:-0.5}"
TEST2_LAMBDA_CONS="${TEST2_LAMBDA_CONS:-0.5}"
TEST2_LAMBDA_LOGIT_CONS="${TEST2_LAMBDA_LOGIT_CONS:-0.5}"
TEST2_LAMBDA_ALIGN="${TEST2_LAMBDA_ALIGN:-0.1}"
TEST2_LAMBDA_MARGIN="${TEST2_LAMBDA_MARGIN:-0.5}"
TEST2_LAMBDA_REG="${TEST2_LAMBDA_REG:-1e-3}"
TEST2_JOINT_MARGIN="${TEST2_JOINT_MARGIN:-1.0}"
TEST2_JOINT_BOOST_THRESHOLD="${TEST2_JOINT_BOOST_THRESHOLD:-0.8}"
TEST2_DYNAMIC_MAX="${TEST2_DYNAMIC_MAX:-5.0}"

PMCMFL_REG_START="${PMCMFL_REG_START:-2}"
PMCMFL_TEMPERATURE="${PMCMFL_TEMPERATURE:-0.5}"
PMCMFL_IMG_WEIGHT="${PMCMFL_IMG_WEIGHT:-1e-8}"
PMCMFL_TEXT_WEIGHT="${PMCMFL_TEXT_WEIGHT:-1e-8}"
PMCMFL_FUSION_WEIGHT="${PMCMFL_FUSION_WEIGHT:-0.01}"
PMCMFL_PROTO_INIT_STD="${PMCMFL_PROTO_INIT_STD:-0.0}"

PEPSY_POOL_SIZE="${PEPSY_POOL_SIZE:-10}"
PEPSY_TOP_K="${PEPSY_TOP_K:-3}"
PEPSY_CONTRASTIVE_WEIGHT="${PEPSY_CONTRASTIVE_WEIGHT:-0.2}"
PEPSY_PROMPT_REG_WEIGHT="${PEPSY_PROMPT_REG_WEIGHT:-0.1}"
PEPSY_TEMPERATURE="${PEPSY_TEMPERATURE:-0.2}"
PEPSY_PROMPT_DROPOUT="${PEPSY_PROMPT_DROPOUT:-0.0}"

POISON_SAVE_LIMIT="${POISON_SAVE_LIMIT:-64}"
POISON_SAVE_INTERVAL="${POISON_SAVE_INTERVAL:-50}"

for strategy in "${STRATEGIES[@]}"; do
  for q in "${Q_VALUES[@]}"; do
    q_tag="${q/./}"
    log_path="${LOG_DIR}/attack_test2_crisismmd_${strategy}_q${q_tag}_${RUN_TS}.log"
    clip_args=()
    if [[ "${USE_PRETRAINED_CLIP}" == "1" ]]; then
      clip_args+=(--attack_test1_use_pretrained_clip)
    fi

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
      --en_backdoor_attack \
      --attack_type attack_test2 \
      --bd_target_label "${TARGET_LABEL}" \
      --bd_malicious_ratio "${MALICIOUS_RATIO}" \
      --bd_poison_ratio "${POISON_RATIO}" \
      --bd_seed "${SEED}" \
      --attack_test1_trigger_size "${TRIGGER_SIZE}" \
      --attack_test1_trigger_value "${TRIGGER_VALUE}" \
      --attack_test1_img_trigger_std "${IMG_TRIGGER_STD}" \
      --attack_test1_img_trigger_eps "${IMG_TRIGGER_EPS}" \
      --attack_test1_text_charset "${TEXT_CHARSET}" \
      --attack_test1_text_trigger_budget "${TEXT_TRIGGER_BUDGET}" \
      --attack_test1_text_beam_width "${TEXT_BEAM_WIDTH}" \
      --attack_test1_text_trigger_scale "${TEXT_TRIGGER_SCALE}" \
      --attack_test1_text_refresh_interval "${TEXT_REFRESH_INTERVAL}" \
      --attack_test1_lambda_img_shift "${LAMBDA_IMG_SHIFT}" \
      --attack_test1_lambda_img_cluster "${LAMBDA_IMG_CLUSTER}" \
      --attack_test1_lambda_text_shift "${LAMBDA_TEXT_SHIFT}" \
      --attack_test1_clip_temperature "${CLIP_TEMPERATURE}" \
      --attack_test2_lambda_clean "${TEST2_LAMBDA_CLEAN}" \
      --attack_test2_lambda_joint "${TEST2_LAMBDA_JOINT}" \
      --attack_test2_lambda_img "${TEST2_LAMBDA_IMG}" \
      --attack_test2_lambda_text "${TEST2_LAMBDA_TEXT}" \
      --attack_test2_lambda_aux "${TEST2_LAMBDA_AUX}" \
      --attack_test2_lambda_anti "${TEST2_LAMBDA_ANTI}" \
      --attack_test2_lambda_cons "${TEST2_LAMBDA_CONS}" \
      --attack_test2_lambda_logit_cons "${TEST2_LAMBDA_LOGIT_CONS}" \
      --attack_test2_lambda_align "${TEST2_LAMBDA_ALIGN}" \
      --attack_test2_lambda_margin "${TEST2_LAMBDA_MARGIN}" \
      --attack_test2_lambda_reg "${TEST2_LAMBDA_REG}" \
      --attack_test2_joint_margin "${TEST2_JOINT_MARGIN}" \
      --attack_test2_joint_boost_threshold "${TEST2_JOINT_BOOST_THRESHOLD}" \
      --attack_test2_dynamic_max "${TEST2_DYNAMIC_MAX}" \
      --attack_test2_poison_save_limit "${POISON_SAVE_LIMIT}" \
      --attack_test2_poison_save_interval "${POISON_SAVE_INTERVAL}" \
      --pmcmfl_regularization_start_epoch "${PMCMFL_REG_START}" \
      --pmcmfl_temperature "${PMCMFL_TEMPERATURE}" \
      --pmcmfl_img_loss_weight "${PMCMFL_IMG_WEIGHT}" \
      --pmcmfl_text_loss_weight "${PMCMFL_TEXT_WEIGHT}" \
      --pmcmfl_fusion_loss_weight "${PMCMFL_FUSION_WEIGHT}" \
      --pmcmfl_prototype_init_std "${PMCMFL_PROTO_INIT_STD}" \
      --pepsy_pool_size "${PEPSY_POOL_SIZE}" \
      --pepsy_top_k "${PEPSY_TOP_K}" \
      --pepsy_contrastive_weight "${PEPSY_CONTRASTIVE_WEIGHT}" \
      --pepsy_prompt_reg_weight "${PEPSY_PROMPT_REG_WEIGHT}" \
      --pepsy_temperature "${PEPSY_TEMPERATURE}" \
      --pepsy_prompt_dropout "${PEPSY_PROMPT_DROPOUT}" \
      "${clip_args[@]}" \
      > "${log_path}" 2>&1
    echo "[INFO] strategy=${strategy} q=${q} finished, log=${log_path}"
  done
done
