# Data and partition preparation

Obtain CrisisMMD, Hateful Memes and UPMC-Food101 from their respective providers
and follow their licenses/access conditions. No raw data is redistributed here.
The expected local dataset directory names are `data/crisis-mmd`,
`data/hateful_memes` and `data/upmc_food101` (UPMC native preprocessing may also
read `data/UPMC-Food101`; inspect its partition script for the input format).

Native partition entry points:

```bash
.venv-fed/bin/python features/data_partitioning/hateful_memes/data_partition.py --raw_data_dir "$PWD/data" --output_partition_path "$PWD/output_prepared" --dataset hateful_memes --alpha 5.0 --num_clients 40
# Corresponding directories for the other datasets:
# features/data_partitioning/crisis-mmd/data_partition.py
# features/data_partitioning/upmc_food101/data_partition.py
```

Read each entry's `--help` and dataset loading code for the exact raw layout.
Native partitions have rows `[sample_id, local_image_path, class_label, text]`.
`system.cfg` defaults to repository-local data and output directories.

For the retained reference protocol, client assignment and row order can be
restored without distributing source text or images:

```bash
.venv-fed/bin/python portability/restore_protocol.py --dataset hateful_memes --prepared-partition output_prepared/partition/hateful_memes/partition_alpha50.json --output-dir output_clip_hateful
```

Use --raw-data-root /path/to/data when moving records with old absolute data paths. Only the data root changes; IDs, labels, text and assignments remain fixed.

The helper joins ID/label metadata to your locally prepared records, verifies
labels and image availability, restores the original client/holdout membership,
and reconstructs all five native mask files. It refuses missing IDs, conflicting
records, train/holdout overlaps and existing output files. It never silently
regenerates a split or substitutes unavailable samples. Different dataset
releases or unavailable images can therefore cause a deliberate failure.

Reference counts: CrisisMMD train/dev/test = 13608/2237/2237 (100 clients);
Hateful Memes = 6732/396/814 (40); UPMC-Food101 = 58578/6507/21683 (100).
Only IDs, class labels, client membership and masks are committed under
`reference_protocol/`; original text, absolute image paths and pixels are absent.
