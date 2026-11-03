#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/hy-tmp/fed}"
PY="${PY:-/usr/local/miniconda3/envs/fed/bin/python}"
DATASET="${DATASET:-hateful_memes}"
STRATEGY="${STRATEGY:-zero}"
Q_VALUES_STRING="${Q_VALUES_STRING:-0.1 0.3 0.5 0.7 0.9}"
MALICIOUS_RATIO="${MALICIOUS_RATIO:-0.1}"
POISON_RATIO="${POISON_RATIO:-0.1}"
SEED="${SEED:-2024}"
ALPHA="${ALPHA:-5.0}"
SAMPLE_RATE="${SAMPLE_RATE:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HID_SIZE="${HID_SIZE:-128}"
TEST_FREQUENCY="${TEST_FREQUENCY:-1}"
TRAIN_INTENSITIES="${TRAIN_INTENSITIES:-0.3,1.0}"
EVAL_INTENSITY="${EVAL_INTENSITY:-1.0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/logs/titim"

case "${DATASET}" in
  crisis-mmd)
    PREP_MODULE="attacks.titim.prepare_crisis_mmd"
    TRAIN_FILE="experiment/crisis-mmd/train.py"
    TARGET_LABEL="${TARGET_LABEL:-0}"
    LR="${LR:-0.05}"
    EPOCHS="${EPOCHS:-200}"
    ;;
  hateful_memes)
    PREP_MODULE="attacks.titim.prepare_hateful_memes"
    TRAIN_FILE="experiment/hateful_memes/train.py"
    TARGET_LABEL="${TARGET_LABEL:-1}"
    LR="${LR:-0.05}"
    EPOCHS="${EPOCHS:-200}"
    ;;
  upmc_food101)
    PREP_MODULE="attacks.titim.prepare_upmc_food101"
    TRAIN_FILE="experiment/upmc_food101/train.py"
    TARGET_LABEL="${TARGET_LABEL:-0}"
    LR="${LR:-0.03}"
    EPOCHS="${EPOCHS:-300}"
    ;;
  *)
    echo "unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

case "${STRATEGY}" in
  zero)
    STRATEGY_ARGS=()
    ;;
  pepsy)
    STRATEGY_ARGS=(
      --pepsy_pool_size 10 --pepsy_top_k 3
      --pepsy_contrastive_weight 0.2 --pepsy_prompt_reg_weight 0.1
      --pepsy_temperature 0.2 --pepsy_prompt_dropout 0.0
    )
    ;;
  pmcmfl)
    STRATEGY_ARGS=(
      --pmcmfl_regularization_start_epoch 2 --pmcmfl_temperature 0.5
      --pmcmfl_img_loss_weight 1e-8 --pmcmfl_text_loss_weight 1e-8
      --pmcmfl_fusion_loss_weight 0.01 --pmcmfl_prototype_init_std 0.0
    )
    ;;
  *)
    echo "unsupported STRATEGY=${STRATEGY}" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"
cd "${ROOT}" || exit 1
export FED_SKIP_MOVIEPY=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

MAIN_LOG="${LOG_DIR}/launcher_titim_${DATASET}_${STRATEGY}_${RUN_TS}.main.log"
read -r -a Q_VALUES <<< "${Q_VALUES_STRING}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"
}

qtag() {
  printf "%02d" "$(awk -v q="$1" 'BEGIN { printf "%d", q * 10 }')"
}

ratio_tag() {
  printf "%s" "$1" | tr -d '.'
}

attack_root() {
  local q="$1"
  printf "%s/output/attacks/titim/%s/alpha%s/mm%s/mr%s_pr%s_tl%s_seed%s" \
    "${ROOT}" "${DATASET}" "$(ratio_tag "${ALPHA}")" "$(qtag "${q}")" \
    "$(ratio_tag "${MALICIOUS_RATIO}")" "$(ratio_tag "${POISON_RATIO}")" \
    "${TARGET_LABEL}" "${SEED}"
}

cleanup_q() {
  local q="$1" path
  path="$(attack_root "${q}")"
  case "${path}" in
    "${ROOT}/output/attacks/titim/${DATASET}/"alpha*/mm??/mr*_pr*_tl*_seed*) ;;
    *) log "refusing unexpected cleanup path: ${path}"; return 9 ;;
  esac
  if [ -d "${path}" ]; then
    rm -rf -- "${path}"
    log "q=${q} cleanup removed ${path}"
  fi
}

log "START TITIM-BadNet dataset=${DATASET} strategy=${STRATEGY} q=${Q_VALUES[*]}"
for q in "${Q_VALUES[@]}"; do
  tag="$(qtag "${q}")"
  qlog="${LOG_DIR}/titim_${DATASET}_${STRATEGY}_q${tag}_${RUN_TS}.log"
  cleanup_q "${q}" || exit $?
  log "q=${q} stage 1/3 save raw poisoned images, then reload and extract features"
  "${PY}" -u -m "${PREP_MODULE}" \
    --raw_data_dir "${ROOT}/data" --output_dir "${ROOT}/output" \
    --dataset "${DATASET}" --alpha "${ALPHA}" \
    --missing_modailty_rate "${q}" --titim_q_dependent_prepare \
    --bd_target_label "${TARGET_LABEL}" --bd_malicious_ratio "${MALICIOUS_RATIO}" \
    --bd_poison_ratio "${POISON_RATIO}" --bd_seed "${SEED}" \
    --titim_train_intensities "${TRAIN_INTENSITIES}" \
    --titim_eval_intensity "${EVAL_INTENSITY}" --titim_force_prepare \
    > "${qlog}" 2>&1 || { log "q=${q} prepare failed; retained artifacts; see ${qlog}"; exit 3; }

  log "q=${q} stage 2/3 verify raw-first image-only artifacts"
  "${PY}" - "$(attack_root "${q}")" "${q}" >> "${qlog}" 2>&1 <<'PY'
import json, sys
from pathlib import Path
root, q = Path(sys.argv[1]), float(sys.argv[2])
meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
assert meta["attack_type"] == "titim" and meta["image_only"] and not meta["text_modified"]
assert abs(float(meta["missing_rate"]) - q) < 1e-9
assert meta["raw_train_count"] > 0 and meta["raw_attack_test_count"] > 0
assert not (root / "poisoned_text").exists()
for path in [root / "raw_samples/train_manifest.jsonl", root / "raw_samples/attack_test_manifest.jsonl", root / "attack_img/test.pkl", root / "attack_text/test.pkl"]:
    assert path.is_file(), path
print("TITIM artifact verification passed", root)
PY
  [ "$?" -eq 0 ] || { log "q=${q} verification failed; retained artifacts"; exit 4; }

  log "q=${q} stage 3/3 federated training"
  "${PY}" -u "${TRAIN_FILE}" \
    --data_dir "${ROOT}/output" --dataset "${DATASET}" --alpha "${ALPHA}" \
    --sample_rate "${SAMPLE_RATE}" --learning_rate "${LR}" --hid_size "${HID_SIZE}" \
    --batch_size "${BATCH_SIZE}" --local_epochs "${LOCAL_EPOCHS}" \
    --num_epochs "${EPOCHS}" --test_frequency "${TEST_FREQUENCY}" \
    --en_missing_modality --missing_modailty_rate "${q}" --missing_strategy "${STRATEGY}" \
    --en_backdoor_attack --attack_type titim --bd_target_label "${TARGET_LABEL}" \
    --bd_malicious_ratio "${MALICIOUS_RATIO}" --bd_poison_ratio "${POISON_RATIO}" \
    --bd_seed "${SEED}" --titim_q_dependent_prepare --titim_attack_weight "${POISON_RATIO}" \
    "${STRATEGY_ARGS[@]}" >> "${qlog}" 2>&1 \
    || { log "q=${q} training failed; retained artifacts; see ${qlog}"; exit 5; }
  log "q=${q} completed successfully"
  cleanup_q "${q}" || exit $?
done
log "ALL DONE TITIM-BadNet dataset=${DATASET} strategy=${STRATEGY}"
