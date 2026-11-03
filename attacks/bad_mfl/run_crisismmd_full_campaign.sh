#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/logs/bad_mfl"
mkdir -p "${LOG_DIR}"

CAMPAIGN_LOG="${LOG_DIR}/bad_mfl_campaign_${RUN_TS}.log"

{
  echo "[$(date '+%F %T')] Bad-MFL full campaign start"
  echo "RUN_TS=${RUN_TS}"
  echo "ROOT_DIR=${ROOT_DIR}"

  RUN_CLEAN="${RUN_CLEAN:-1}" \
  RUN_ATTACK="${RUN_ATTACK:-1}" \
  FORCE_PREPARE="${FORCE_PREPARE:-1}" \
  RUN_TS="${RUN_TS}" \
  bash attacks/bad_mfl/run_crisismmd_campaign.sh

  RUN_TS="${RUN_TS}" \
  bash attacks/bad_mfl/run_crisismmd_missing_campaign.sh

  echo "[$(date '+%F %T')] Bad-MFL full campaign done"
} 2>&1 | tee "${CAMPAIGN_LOG}"
