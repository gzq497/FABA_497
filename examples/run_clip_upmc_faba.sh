#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv-fed/bin/python}"
OUT="${OUT:-$ROOT/output_clip_upmc_faba}"
TARGET="${TARGET:-0}"
PHASE="${1:-help}"
QS=(0.1 0.3 0.5 0.7 0.9)
if [[ "$PHASE" == clean ]]; then
  "$PY" portability/run.py --architecture clip --module features.feature_processing.dual_encoder_features --architecture clip --dataset upmc_food101 --data-root "$OUT" --modality both --device cuda:0
elif [[ "$PHASE" == raw || "$PHASE" == extract || "$PHASE" == verify ]]; then
  if [[ "$PHASE" == extract ]]; then
    for Q in "${QS[@]}"; do
      TAG="$("$PY" -c 'import sys;print(str(int(round(float(sys.argv[1])*10))).zfill(2))' "$Q")"
      [[ -f "$OUT/attacks/attack_test2/upmc_food101/alpha50/mm$TAG/mr01_pr01_tl${TARGET}_ts24/metadata.raw.json" ]] || { echo "Complete all raw q first ($Q missing)"; exit 1; }
    done
  fi
  for Q in "${QS[@]}"; do
    "$PY" portability/run.py --architecture clip --module attacks.attack_test2.prepare_clip_food101 --stage "$PHASE" --output_dir "$OUT" --raw_data_dir "$ROOT/data" --missing_modailty_rate "$Q" --bd_target_label "$TARGET"
  done
elif [[ "$PHASE" == train ]]; then
  PY="$PY" OUT="$OUT" TARGET="$TARGET" bash "$0" verify
  for Q in "${QS[@]}"; do
    "$PY" portability/run.py --architecture clip experiment/upmc_food101/train_clip_faba.py --data_dir "$OUT" --raw_data_dir "$ROOT/data" --dataset upmc_food101 --alpha 5.0 --fed_alg fed_avg --optimizer sgd --learning_rate .05 --hid_size 128 --batch_size 16 --local_epochs 1 --sample_rate .1 --num_epochs 200 --num_folds 5 --test_frequency 1 --en_missing_modality --missing_modailty_rate "$Q" --missing_strategy zero --img_feat clip_vit_b32_img --text_feat clip_vit_b32_text --att_name no_att --backdoor_attack --eval_attack --attack_type attack_test2 --bd_malicious_ratio .1 --bd_poison_ratio .1 --bd_target_label "$TARGET" --bd_seed 2024 --bd_trigger_size 24 --no-attack_test2_raw_prepare --attack_test2_use_raw_trigger_features
  done
else
  echo 'Usage: run_clip_upmc_faba.sh clean|raw|extract|verify|train'
  [[ "$PHASE" == help ]] || exit 2
fi
