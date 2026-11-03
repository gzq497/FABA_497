#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-both}"
PY="${PYTHON39:-python3.9}"
FLAVOR="${TORCH_FLAVOR:-cu117}"
[[ "$PROFILE" == fed || "$PROFILE" == align || "$PROFILE" == both ]] || { echo 'Usage: setup_env.sh [fed|align|both]'; exit 2; }
[[ "$FLAVOR" == cu117 || "$FLAVOR" == cpu ]] || { echo 'TORCH_FLAVOR must be cu117 or cpu'; exit 2; }
"$PY" -c 'import sys; assert sys.version_info[:2]==(3,9), "Use Python 3.9 (e.g. conda create -n fed-bootstrap python=3.9)"'
for p in fed align; do
 [[ "$PROFILE" == both || "$PROFILE" == "$p" ]] || continue
 V="$ROOT/.venv-$p"
 [[ -x "$V/bin/python" ]] || "$PY" -m venv "$V"
 "$V/bin/python" -m pip install 'pip==23.2.1' 'setuptools==68.0.0' 'wheel==0.38.4'
 "$V/bin/python" -m pip install "torch==1.13.0+$FLAVOR" "torchvision==0.14.0+$FLAVOR" "torchaudio==0.13.0+$FLAVOR" --extra-index-url "https://download.pytorch.org/whl/$FLAVOR"
 "$V/bin/python" -m pip install -r "$ROOT/portability/requirements-$p.txt"
 "$V/bin/python" -m pip check
done
"$PY" "$ROOT/portability/configure_paths.py"
echo "Ready. Run: $ROOT/.venv-fed/bin/python $ROOT/portability/smoke_test.py"
