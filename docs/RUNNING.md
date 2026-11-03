# Running experiments

All commands run from the repository root after environment setup and model
retrieval. `portability/run.py` selects the feature contract for the requested
architecture and exposes the local model caches. Generic `train.py` is launched
through this wrapper when selecting CLIP or ALIGN.

## Clean Hateful Memes example (CLIP)

Prepare the local dataset, then restore its reference protocol as described in
DATA.md into `output_clip_hateful`. Extract the frozen features:

```bash
.venv-fed/bin/python portability/run.py --architecture clip --module features.feature_processing.dual_encoder_features --architecture clip --dataset hateful_memes --data-root "$PWD/output_clip_hateful" --modality both --device cuda:0
for Q in 0.1 0.3 0.5 0.7 0.9; do
  .venv-fed/bin/python portability/run.py --architecture clip experiment/hateful_memes/train.py --data_dir "$PWD/output_clip_hateful" --dataset hateful_memes --alpha 5.0 --fed_alg fed_avg --optimizer sgd --learning_rate .05 --hid_size 128 --batch_size 16 --local_epochs 1 --sample_rate .1 --num_epochs 200 --num_folds 5 --test_frequency 1 --en_missing_modality --missing_modailty_rate "$Q" --missing_strategy zero
done
```

For ALIGN use `.venv-align/bin/python` and replace both architecture selections
with `align`. For MobileNet/MobileBERT, use the dataset's native image/text
extraction scripts under `features/feature_processing/` and select `mobilenet`
in the wrapper. Synthetic FL checks exercise all three strategies and all three
feature contracts without requiring external data or weights.

## Staged FABA example (CLIP + UPMC-Food101 + zero)

Restore the UPMC reference protocol to a new output directory, then run:

```bash
PY="$PWD/.venv-fed/bin/python" OUT="$PWD/output_clip_upmc_faba" bash examples/run_clip_upmc_faba.sh clean
PY="$PWD/.venv-fed/bin/python" OUT="$PWD/output_clip_upmc_faba" bash examples/run_clip_upmc_faba.sh raw
PY="$PWD/.venv-fed/bin/python" OUT="$PWD/output_clip_upmc_faba" bash examples/run_clip_upmc_faba.sh extract
PY="$PWD/.venv-fed/bin/python" OUT="$PWD/output_clip_upmc_faba" bash examples/run_clip_upmc_faba.sh verify
PY="$PWD/.venv-fed/bin/python" OUT="$PWD/output_clip_upmc_faba" bash examples/run_clip_upmc_faba.sh train
```

The raw phase completes **all five q** before extraction. Extraction reads saved
PNG/TXT bytes. Training verifies all five q first, disables in-training raw
preparation and enables raw-trigger feature loading. The example explicitly uses
target label 0; specify `TARGET` consistently in every phase to use another
target. It runs 200 rounds and five folds (seeds 8/16/24/32/40), alpha 5, malicious
and poison ratios .1, batch 16, hidden 128, local epochs 1, SGD .05, sample rate .1.
Do not reuse an output directory across changed settings.

Other retained preparation modules expose `--help`, including the date-stamped
dataset/architecture adapters. Their `common.py` contains the expected clean
feature source directory; prepare those clean features first. Select the native
entry point appropriate to the dataset, encoder and strategy instead of
substituting feature-level synthetic triggers into a saved-raw-input run.

Round records and summaries are written by the native trainers/auditors. Select
the maximum clean development Macro-F1 with the earliest tie, then take clean
and attack metrics from that same round. Report five-fold means and sample
standard deviations; preserve empty-update rounds. Never select by attack success
or test performance. The archive does not include historical results or models.
