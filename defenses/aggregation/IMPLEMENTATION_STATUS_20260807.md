# Multi-Krum and FLAME integration status

## Scope

- Project: `/hy-tmp/fed`
- Shared server path: `Server.save_train_updates()` -> `Server.average_weights()`
- Compatible attack dispatch: `attack_test2`, `bad_mfl`, `badcm`, `titim`,
  `toxictextclip`
- Supported completion models: zero-filled, PEPSY, PMCMFL
- Default behavior: `--robust_aggregation fedavg`

No attack trainer was modified. Robust aggregation receives the same client
state dictionaries that previously entered FedAvg. PMCMFL prototype uploads are
filtered using the model-update acceptance set.

## Verification completed

- Python compilation for all defense modules and three multimodal entry points.
- Argument parsing for Crisis-MMD, Hateful-Memes, and UPMC-Food101.
- Eight unit/integration tests, including exact default FedAvg weighting.
- Crisis-MMD zero-filled one-round five-fold Multi-Krum smoke run.
- Crisis-MMD zero-filled one-round five-fold FLAME smoke run.
- Crisis-MMD zero-filled one-round five-fold default FedAvg regression run.

Smoke logs:

```text
/hy-tmp/fed/logs/defenses/smoke_crisismmd_zero_multi_krum_q01.log
/hy-tmp/fed/logs/defenses/smoke_crisismmd_zero_flame_q01.log
/hy-tmp/fed/logs/defenses/smoke_crisismmd_zero_fedavg_q01.log
```

## Example flags

Append one of these to the training stage of an existing attack campaign:

```text
--robust_aggregation multi_krum --robust_byzantine_ratio 0.1
--robust_aggregation flame --flame_noise_multiplier 0.001
```

The attack preparation, saved poisoned samples, extracted features, and local
client selection remain unchanged.

## Operational notes

- `hdbscan==0.8.40` is installed in the `fed` environment.
- HDBSCAN is imported before CUDA/data-loader initialization to avoid a native
  OpenMP loading crash observed when importing it after local training.
- Multi-Krum requires `n >= 2f + 3` after skipped clients are removed.
- Robust aggregation with Scaffold is intentionally rejected.
- The pre-deployment files are stored at
  `/hy-tmp/fed/backup_robust_aggregation_pre_20260807_0701`.

