# Robust aggregation

This module adds server-side Multi-Krum and FLAME without changing any client
trainer. It therefore applies to `attack_test2`, `bad_mfl`, `badcm`, `titim`,
and `toxictextclip` through the common `Server.save_train_updates()` path.

## Selection

Use exactly one of:

```text
--robust_aggregation fedavg
--robust_aggregation multi_krum
--robust_aggregation flame
```

The default is `fedavg`, which retains the previous aggregation path. Robust
runs receive a distinct `_agg<method>` result suffix.

Multi-Krum uses trainable model deltas, squared Euclidean distances, the
closest `n-f-2` updates for each score, and an unweighted mean of the selected
updates. `f` is an assumed upper bound and never uses malicious client IDs.

FLAME clusters model vectors by cosine distance with HDBSCAN, keeps the largest
non-noise cluster, clips accepted model deltas to the median client-update norm,
averages the clipped deltas, and optionally adds Gaussian noise. The default
noise standard deviation is `0.001 * median_update_norm`, matching the image
classification setting reported by FLAME, and is always logged.

## Important constraints

Multi-Krum requires `n >= 2f + 3` in every round. With 42 Hateful-Memes clients
and `sample_rate=0.1`, only four clients are selected, so `f=1` is invalid. Use
at least five participating clients, and keep the same participation setting
for defended and undefended comparisons.

Robust aggregation filters PMCMFL prototype uploads with the same accepted
client indices. Non-trainable model buffers remain at their previous global
values. Scaffold is rejected because its control-variate aggregation would
otherwise be inconsistent with filtered model updates.

## Dependency

FLAME requires `hdbscan`, pinned in `requirements_defenses.txt`.

## References

- Blanchard et al., Machine Learning with Adversaries, NeurIPS 2017.
- Nguyen et al., FLAME: Taming Backdoors in Federated Learning, USENIX 2022.
