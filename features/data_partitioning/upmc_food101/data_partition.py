import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.data_partitioning.partition_manager import PartitionManager


def _read_path_conf():
    path_conf = {}
    with open(str(PROJECT_ROOT.joinpath("system.cfg"))) as f:
        for line in f:
            key, val = line.strip().split("=")
            path_conf[key] = val.replace('"', "")
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(PROJECT_ROOT.joinpath("data"))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(PROJECT_ROOT.joinpath("output"))
    return path_conf


def _load_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _make_record(row, image_root, split):
    label_name = row["annotation"]
    image_path = Path(image_root).joinpath("images", split, label_name, row["id"])
    if not image_path.exists():
        return None
    text = row.get("text", "").strip()
    if len(text) == 0:
        text = label_name.replace("_", " ")
    return [
        row["id"],
        str(image_path),
        int(row["label"]),
        text,
    ]


def _split_train_dev(records, dev_ratio, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record[2])].append(record)

    rng = random.Random(seed)
    train_records, dev_records = [], []
    for label in sorted(grouped):
        rows = grouped[label]
        rng.shuffle(rows)
        dev_count = max(1, int(round(len(rows) * dev_ratio)))
        dev_records.extend(rows[:dev_count])
        train_records.extend(rows[dev_count:])

    train_records.sort(key=lambda x: (x[2], x[0]))
    dev_records.sort(key=lambda x: (x[2], x[0]))
    return train_records, dev_records


def data_partition(args):
    data_root = Path(args.raw_data_dir).joinpath("UPMC-Food101", "UPMC-Food-101")
    train_csv = data_root.joinpath("train.csv")
    test_csv = data_root.joinpath("test.csv")
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"UPMC-Food101 csv files not found under {data_root}")

    train_all, skipped_train = [], 0
    for row in _load_csv(train_csv):
        record = _make_record(row, data_root, "train")
        if record is None:
            skipped_train += 1
            continue
        train_all.append(record)

    test_records, skipped_test = [], 0
    for row in _load_csv(test_csv):
        record = _make_record(row, data_root, "test")
        if record is None:
            skipped_test += 1
            continue
        test_records.append(record)

    train_records, dev_records = _split_train_dev(
        train_all,
        dev_ratio=float(args.dev_ratio),
        seed=int(args.seed),
    )

    pm = PartitionManager(args)
    file_label_list = [record[2] for record in train_records]
    file_idx_clients = pm.direchlet_partition(
        file_label_list,
        min_sample_size=int(args.min_sample_size),
    )

    client_data_dict = {}
    for client_idx in range(args.num_clients):
        client_data_dict[client_idx] = [
            train_records[idx] for idx in file_idx_clients[client_idx]
        ]
    client_data_dict["dev"] = dev_records
    client_data_dict["test"] = test_records

    output_data_path = Path(args.output_partition_path).joinpath(
        "partition",
        args.dataset,
    )
    output_data_path.mkdir(parents=True, exist_ok=True)
    alpha_str = str(args.alpha).replace(".", "")
    output_file = output_data_path.joinpath(f"partition_alpha{alpha_str}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(client_data_dict, f, indent=4)

    print(f"Saved partition to {output_file}")
    print(
        "UPMC-Food101 records: "
        f"train_clients={len(train_records)}, dev={len(dev_records)}, "
        f"test={len(test_records)}, skipped_train={skipped_train}, skipped_test={skipped_test}"
    )


def parse_args():
    path_conf = _read_path_conf()
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default=path_conf["data_dir"])
    parser.add_argument("--output_partition_path", type=str, default=path_conf["output_dir"])
    parser.add_argument("--setup", type=str, default="federated")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="upmc_food101")
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--min_sample_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    data_partition(parse_args())
