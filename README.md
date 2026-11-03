# FABA multimodal federated learning framework

Source distribution for image/text federated learning with missing modalities.
The framework includes FABA (`attacks/attack_test2`), TITIM-BadNet, ToxicTextCLIP,
BadCM, Bad-MFL and three-path variants, with zero, PEPSY and PMCMFL strategies.
Frozen MobileNetV2 + MobileBERT, CLIP ViT-B/32 and ALIGN encoders are retained.

## Install (Linux)

Use Python 3.9 and a compatible NVIDIA driver for CUDA runs. Two environments
keep the original/CLIP and ALIGN Transformers versions separate.

```bash
python3.9 portability/verify_package.py
PYTHON39=python3.9 bash portability/setup_env.sh both
# CPU environment instead: TORCH_FLAVOR=cpu PYTHON39=python3.9 bash portability/setup_env.sh both
.venv-fed/bin/python portability/smoke_test.py
.venv-fed/bin/python portability/smoke_fl.py
```

`setup_env.sh` installs PyTorch 1.13.0 and the version-pinned dependencies,
then relocates path literals **inside this checkout**. Run it after extracting
or cloning; a relocated virtual environment should be recreated. Verify the
release manifest before relocation, since relocation changes file hashes.

## Data and encoders

Datasets, pretrained weights and generated experiment outputs are not included.
See [data preparation](docs/DATA.md) and [experiment commands](docs/RUNNING.md).
The source-owned BadCM static pattern is included as a runtime asset.

```bash
.venv-fed/bin/python portability/fetch_models.py --architecture clip
.venv-align/bin/python portability/fetch_models.py --architecture align
# Original MobileNet/MobileBERT encoders:
.venv-fed/bin/python portability/fetch_models.py --architecture mobilenet
```

CLIP uses independent L2-normalized 512-dimensional pooled features and 77-token
text input; ALIGN uses 640-dimensional features and 64-token text input.
ALIGN refers to the Kakao Brain `kakaobrain/align-base` checkpoint.
Checkpoint revisions and hashes are defined in `constants/vision_language.py`.

## Repository layout

| Path | Contents |
| --- | --- |
| `attacks/` | Attack training, raw-sample preparation and evaluation |
| `trainers/`, `model/`, `methods/` | Federated optimization, classifiers and missing-modality handling |
| `features/`, `dataloader/` | Dataset partitioning, offline encoders and loaders |
| `defenses/` | Aggregation and post-training defenses |
| `experiment/` | Dataset-specific experiment entry points |
| `reference_protocol/` | Sample IDs/labels, client membership and five-q masks; no sample text or images |
| `portability/` | Installation, relocation, model retrieval and synthetic checks |
| `examples/` | Explicit staged CLIP + UPMC-Food101 + FABA example |

Date-stamped modules in `attacks/` and `experiment/` retain experiment-specific
entry points and settings. Historical server monitoring, automatic sample deletion
and migration launchers are omitted. The commands in `docs/RUNNING.md` start fresh
experiments without historical server status files.

## Validation and reuse

Synthetic checks validate interfaces and short training runs; they are not formal
experiment results. Consult `VALIDATION.json` for checks actually run on this
release. Use a fresh output directory for each experiment definition.

Original attribution comments and bundled third-party notices are preserved;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This distribution has no
new blanket license grant. Anonymization and repository publication remain with
the submitting authors. [GitHub upload steps](docs/UPLOAD.md).
