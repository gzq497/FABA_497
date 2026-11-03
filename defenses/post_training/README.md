# PAR and TSC post-training defenses

This package adds two server-side model purification defenses without changing
the default federated training path. Both defenses run once per fold after the
global model checkpoint with the best clean dev Macro-F1 has been selected.

## Shared protocol

- `--server_defense none` is the default and preserves the previous behavior.
- `--server_defense par` enables Perturb-and-Recover adaptation.
- `--server_defense tsc` enables two-stage curve purification.
- The server reference set is a class-stratified 5% sample of clean, complete
  training pairs (`--defense_clean_ratio 0.05`, seed 2024).
- Reference IDs, class counts, and SHA256 are written to `clean_reference.json`.
- The reference set is independent of missing-rate simulation and is not built
  from dev, test, or attack data.
- Attack data are evaluated only before and after purification, never for
  checkpoint or hyperparameter selection.
- PMCMFL prototypes are fixed during purification and stored with checkpoints.

Outputs are written below the normal result directory:

```
post_par/foldN/{pre_defense.pt,post_defense.pt,clean_reference.json,defense_result.json}
post_tsc/foldN/{pre_defense.pt,post_defense.pt,clean_reference.json,defense_result.json}
```

## PAR

PAR keeps a frozen copy of the selected global model and optimizes a trainable
copy on clean server data. The objective combines supervised clean
cross-entropy with bounded image/text embedding displacement. Defaults are 10
epochs, AdamW, peak learning rate `3e-5`, warmup start `3e-6`, final learning
rate `1e-9`, `tau=2.15`, and perturbation weight 1.0.

## TSC

TSC performs three purification rounds. Each round uses two quadratic Bezier
curves with fixed endpoints and a trainable bend. The safe permutation groups
are image projection, classifier hidden layer, and PEPSY projector/fuser hidden
layers. GRU state dimensions are intentionally not permuted. Every permutation
is rejected if the maximum clean-batch logit error exceeds `1e-5`.

Defaults are three rounds, two 200-epoch curve stages per round, `t=0.4`, curve
learning rate 0.02 with cosine decay, and a five-epoch clean fine-tune between
rounds.

## Example

Append one of these options to an existing attack command. The attack,
missing-modality strategy, q, and poison settings remain unchanged.

```
--server_defense par --defense_clean_ratio 0.05 --defense_clean_seed 2024
```

```
--server_defense tsc --defense_clean_ratio 0.05 --defense_clean_seed 2024
```

Use `--robust_aggregation fedavg` when reporting PAR/TSC as standalone defenses.
Multi-Krum or FLAME can be combined for ablation, but that is a different
experimental condition and should be labeled explicitly.
