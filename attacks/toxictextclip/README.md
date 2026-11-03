# ToxicTextCLIP framework adaptation

This package integrates the official `xinyaocse/ToxicTextCLIP` method as an
attack type parallel to `bad_mfl`, `badcm`, and `attack_test2`.

ToxicTextCLIP is text-only. The image and image feature are never modified.
Preparation selects malicious clients that retain text under the requested
missing-rate simulation, ranks candidate non-target texts against a target-text
prototype, chooses a background-aligned target caption, augments and saves the
poisoned text, reloads the saved text, and extracts MobileBERT features before
federated training. Image records are copied only to synchronize the target
training label required by the framework's paired dataset loader.

The official repository exposes its selector/augmenter as offline research
scripts with manual paths rather than a reusable API. This implementation is a
framework-compatible adaptation of that pipeline and records the source commit
and preparation trace in `metadata.json`.

Only the text-only attack metric is defined:

- `TA` / `ASR_T`: `(zero image, poisoned text) -> target label`
- Joint ASR and image-only ASR are intentionally not constructed or reported.

Example preparation:

```bash
python -m attacks.toxictextclip.prepare_upmc_food101 \
  --output_dir /hy-tmp/fed/output \
  --dataset upmc_food101 \
  --alpha 5.0 \
  --missing_modailty_rate 0.1 \
  --bd_target_label 0 \
  --bd_malicious_ratio 0.1 \
  --bd_poison_ratio 0.1 \
  --toxictextclip_q_dependent_prepare \
  --toxictextclip_force_prepare
```

Then train with `--attack_type toxictextclip` and the same dataset/completion
settings used during preparation.
