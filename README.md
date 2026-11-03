# FABA

## A Filling-Aware Backdoor Attack on Multimodal Federated Learning with Missing Modalities

This repository contains the FABA implementation and an image–text federated
learning framework for studying backdoor attacks under missing modalities.
It provides dataset preprocessing, frozen feature encoders, missing-modality
handling, attack preparation, federated training, evaluation, and defenses.

FABA is implemented in [`attacks/attack_test2/`](attacks/attack_test2/).
The `attack_test2` name is retained in command-line arguments and output paths.

## Contents

- [Framework overview](#framework-overview)
- [Installation](#installation)
- [Datasets and reference protocol](#datasets-and-reference-protocol)
- [Pretrained encoders](#pretrained-encoders)
- [Quick start: FABA with CLIP](#quick-start-faba-with-clip)
- [Other experiments](#other-experiments)
- [Evaluation and reproducibility](#evaluation-and-reproducibility)
- [Repository structure](#repository-structure)
- [Acknowledgments](#acknowledgments)

## Framework overview

The framework separates frozen image/text feature extraction from downstream
federated optimization. Experiments cover paired, image-only, text-only, and
fully missing client modality profiles.

| Component | Available options |
| --- | --- |
| Datasets | CrisisMMD, Hateful Memes, UPMC-Food101 |
| Encoders | MobileNetV2 + MobileBERT, CLIP ViT-B/32, ALIGN |
| Missing-modality strategies | Zero filling, PEPSY, PMCMFL |
| Attacks | FABA, TITIM-BadNet, ToxicTextCLIP, BadCM, Bad-MFL |
| Additional attack variants | BadCM-3path, Bad-MFL-3path |
| Missing rates | `q = 0.1, 0.3, 0.5, 0.7, 0.9` |

The saved-raw-input experiment workflow is:

```text
Local dataset
  → client partitions and missing-modality masks
  → clean frozen features
  → poisoned PNG/TXT files for all five q values
  → frozen features extracted from the saved poisoned files
  → artifact verification
  → federated training and round-level evaluation
```

Here, `q` denotes the modality missing rate, not the malicious-client or poison
ratio. Select the preparation and training entry points matching the dataset,
encoder, attack, and missing-modality strategy.

## Installation

The installation scripts target **Linux with Python 3.9**. CUDA experiments
require a compatible NVIDIA driver.

```bash
git clone https://github.com/gzq497/FABA_497.git
cd FABA_497

# Verify the source snapshot before configuring machine-specific paths.
python3.9 portability/verify_package.py

# Create both environments and configure paths inside this checkout.
PYTHON39=python3.9 bash portability/setup_env.sh both
```

Two environments keep the encoder dependencies separate:

| Environment | Purpose | PyTorch | Transformers |
| --- | --- | --- | --- |
| `.venv-fed` | MobileNet/MobileBERT and CLIP | 1.13.0 | 4.24.0 |
| `.venv-align` | ALIGN | 1.13.0 | 4.30.2 |

Full dependency pins are in
[`portability/requirements-fed.txt`](portability/requirements-fed.txt) and
[`portability/requirements-align.txt`](portability/requirements-align.txt).
Use `fed` or `align` instead of `both` to install one environment.

For CPU smoke checks, install with:

```bash
TORCH_FLAVOR=cpu PYTHON39=python3.9 bash portability/setup_env.sh both
```

Setup relocates path literals within the checkout. Verify the manifest
**before** setup; configured files will have different hashes.

### Check the installation

```bash
.venv-fed/bin/python portability/smoke_test.py
.venv-fed/bin/python portability/smoke_fl.py
```

These checks use synthetic inputs for forward/backward passes and short
federated runs. They do not require external data or pretrained weights and do
not constitute full experiment results. Recorded release checks are in
[`VALIDATION.json`](VALIDATION.json).

## Datasets and reference protocol

Acquire the datasets from their providers and prepare them as described in
[`docs/DATA.md`](docs/DATA.md). Raw images, text annotations, pretrained weights,
generated features, poisoned samples, and trained models are not distributed
in this repository.

The retained reference protocol has the following sample counts:

| Dataset | Classes | Training | Development | Test | Training clients |
| --- | ---: | ---: | ---: | ---: | ---: |
| CrisisMMD | 8 | 13,608 | 2,237 | 2,237 | 100 |
| Hateful Memes | 2 | 6,732 | 396 | 814 | 40 |
| UPMC-Food101 | 101 | 58,578 | 6,507 | 21,683 | 100 |

[`reference_protocol/`](reference_protocol/) contains sample IDs, labels, client
membership, and five-q masks, without dataset images or text.
`portability/restore_protocol.py` joins this metadata to locally prepared
records and reconstructs the native partition and mask files.

For example, after preparing Hateful Memes locally:

```bash
.venv-fed/bin/python portability/restore_protocol.py \
  --dataset hateful_memes \
  --prepared-partition output_prepared/partition/hateful_memes/partition_alpha50.json \
  --output-dir output_clip_hateful
```

Use `--raw-data-root /path/to/data` when relocating records from another machine.
The helper checks sample IDs, labels, and image availability and refuses to
overwrite an existing protocol.

## Pretrained encoders

Download the encoder needed for the experiment:

```bash
.venv-fed/bin/python portability/fetch_models.py --architecture mobilenet
.venv-fed/bin/python portability/fetch_models.py --architecture clip
.venv-align/bin/python portability/fetch_models.py --architecture align
```

| Architecture | Image representation | Text representation |
| --- | --- | --- |
| MobileNetV2 + MobileBERT | 1,280-dimensional features | 512-dimensional token features |
| CLIP ViT-B/32 | 512-dimensional projected pooled features | 512-dimensional projected pooled features; 77-token limit |
| ALIGN | 640-dimensional projected pooled features | 640-dimensional projected pooled features; 64-token limit |

CLIP and ALIGN features are independently encoded, L2-normalized `float32`
vectors. CLIP uses `openai/clip-vit-base-patch32`; ALIGN uses
`kakaobrain/align-base`. Their pinned revisions and weight checksums are in
[`constants/vision_language.py`](constants/vision_language.py).

## Quick start: FABA with CLIP

The included example runs **CLIP + UPMC-Food101 + zero filling + FABA** for
all five missing rates. Acquire and arrange the local UPMC data first; consult
[`docs/DATA.md`](docs/DATA.md) and the native partition script for the input layout.

### 1. Prepare and restore the protocol

```bash
.venv-fed/bin/python features/data_partitioning/upmc_food101/data_partition.py \
  --raw_data_dir "$PWD/data" \
  --output_partition_path "$PWD/output_prepared" \
  --dataset upmc_food101 --alpha 5.0 --num_clients 100

.venv-fed/bin/python portability/restore_protocol.py \
  --dataset upmc_food101 \
  --prepared-partition output_prepared/partition/upmc_food101/partition_alpha50.json \
  --output-dir output_clip_upmc_faba
```

### 2. Run the stages in order

```bash
export PY="$PWD/.venv-fed/bin/python"
export OUT="$PWD/output_clip_upmc_faba"
export TARGET=0

# Extract clean features using the frozen CLIP encoders.
bash examples/run_clip_upmc_faba.sh clean

# Save poisoned samples for every q before extracting poisoned features.
bash examples/run_clip_upmc_faba.sh raw
bash examples/run_clip_upmc_faba.sh extract
bash examples/run_clip_upmc_faba.sh verify

# Verify every q again, then train with the saved-file features.
bash examples/run_clip_upmc_faba.sh train
```

Keep `OUT` and `TARGET` fixed across all stages. This example uses target label
`0`; choose the target explicitly before preparation. Start a new output
directory when changing the experiment definition.

| Example setting | Value |
| --- | --- |
| Missing rates | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Rounds / folds | 200 / 5 |
| Fold seeds | 8, 16, 24, 32, 40 |
| Dirichlet alpha | 5.0 |
| Client sampling ratio | 0.1 |
| Malicious-client / poison ratios | 0.1 / 0.1 |
| Optimizer / learning rate | SGD / 0.05 |
| Local epochs / batch size | 1 / 16 |
| Hidden dimension | 128 |
| Attack seed | 2024 |

Training disables automatic raw preparation and loads verified saved-file
features. The stage script and native entry points retain the full arguments.

## Other experiments

[`docs/RUNNING.md`](docs/RUNNING.md) provides a clean CLIP example and explains
how to select other encoders and strategies. Dataset-specific training entry
points are under `experiment/crisis-mmd/`, `experiment/hateful_memes/`, and
`experiment/upmc_food101/`.

When using generic `train.py` with CLIP or ALIGN, launch it through
`portability/run.py --architecture clip|align` so that feature names and
dimensions match the encoder. Use `.venv-align/bin/python` for ALIGN.

Native modules expose `--help`. See
[`attacks/THREE_PATH_README.md`](attacks/THREE_PATH_README.md) for three-path
variants. Date-stamped preparation adapters retain experiment-specific settings;
their `common.py` files identify required clean feature directories.

## Evaluation and reproducibility

The trainers and auditors retain round-level records and summaries. Evaluation
includes clean accuracy, clean Macro-F1, joint-trigger attack success, image-only
and text-only attack success where applicable, and modality-balance statistics.

For the documented selection protocol, select the round with the highest
**clean development Macro-F1**, breaking ties by the earliest round. Read all
clean and attack metrics from that same round. Report five-fold means and sample
standard deviations (`ddof=1`), retaining empty-update rounds in the records.

Keep client partitions, missing-modality masks, seeds, target labels, and training
settings consistent across experiments being compared. Feature manifests and
saved-file checksums support provenance verification during preparation.

## Repository structure

```text
attacks/                 Attack preparation, local training, and evaluation
  attack_test2/          FABA implementation
constants/               Dataset metadata and encoder contracts
dataloader/              Feature and missing-modality data loading
defenses/                Aggregation and post-training defenses
experiment/              Dataset-specific training entry points
features/                Partitioning, missing masks, and frozen extraction
methods/                 Federated learning method utilities
model/                   Encoders, fusion modules, and classifiers
trainers/                Server/client training and result auditing
reference_protocol/      Reference IDs, labels, client assignments, and masks
portability/             Environment setup, model retrieval, and smoke checks
examples/                Staged experiment scripts
docs/                    Data preparation and execution details
```

## Acknowledgments

This framework builds on existing multimodal federated learning and attack
implementations. Upstream attribution comments and bundled license notices are
retained. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the notices
in the corresponding method directories.
