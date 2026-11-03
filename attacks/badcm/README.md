# BadCM framework adaptation

This package integrates [xandery-geek/BadCM](https://github.com/xandery-geek/BadCM)
as an attack method parallel to `bad_mfl` and `attack_test2`.

The preparation pipeline is raw-sample first. It loads the original image and
text, applies the upstream BadCM visual pattern as a bounded pixel-space
perturbation and the direct text-pattern strategy, saves the resulting image
and text under `raw_samples`, reloads those files, and only then extracts
MobileNetV2 and MobileBERT features. Manifests preserve the original and
poisoned paths for audit.

The upstream repository does not ship a trained visual-generator checkpoint.
This package is therefore a raw-sample BadCM adaptation, not a claim that the
paper's learned generator checkpoint has been reproduced.

Prepare Hateful Memes artifacts:

```bash
/usr/local/miniconda3/envs/fed/bin/python -m attacks.badcm.prepare_hateful_memes \
  --output_dir /hy-tmp/fed/output \
  --dataset hateful_memes \
  --alpha 5.0 \
  --bd_malicious_ratio 0.1 \
  --bd_poison_ratio 0.1 \
  --bd_target_label 1 \
  --bd_seed 2024 \
  --missing_modailty_rate 0.1 \
  --badcm_pixel_eps 0.0156862745 \
  --badcm_text_trigger cf
```

Train with the same experiment entry point used by comparison methods:

```bash
/usr/local/miniconda3/envs/fed/bin/python experiment/hateful_memes/train.py \
  --dataset hateful_memes \
  --data_dir /hy-tmp/fed/output \
  --alpha 5.0 \
  --backdoor_attack \
  --attack_type badcm \
  --bd_malicious_ratio 0.1 \
  --bd_poison_ratio 0.1 \
  --bd_target_label 1 \
  --bd_seed 2024
```

For missing-modality experiments, pass `--missing_modality`,
`--missing_modailty_rate`, and the desired `--missing_strategy`. By default,
BadCM artifacts are q-dependent and live under `mm{q}`. Q-dependent
preparation excludes clients whose image and text modalities are both
completely unavailable, and uses deterministic stratified selection across
the remaining modality types.
