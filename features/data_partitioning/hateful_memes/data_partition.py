# Author: Tiantian Feng
# USC SAIL lab, tiantiaf@usc.edu
import json
import sys, os
import re, pdb
import argparse
from pathlib import Path

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.data_partitioning.partition_manager import PartitionManager


def data_partition(args: dict):
    
    # Read arguments
    num_clients, alpha = args.num_clients, args.alpha
    
    # Define partition manager
    pm = PartitionManager(args)
    
    # Fetch all labels
    pm.fetch_label_dict()
    
    # read train, dev, and test dict
    train_data_dict, dev_data_dict, test_data_dict = dict(), dict(), dict()
    skipped_missing = {"train": 0, "dev": 0, "test": 0}

    def add_record(target_dict, split_name, line_data):
        img_rel = line_data['img']
        img_abs = Path(args.raw_data_dir).joinpath(args.dataset, img_rel)
        if Path.exists(img_abs) == False:
            skipped_missing[split_name] += 1
            return
        target_dict[img_rel] = [
            img_rel,
            str(img_abs),
            line_data['label'],
            line_data['text']
        ]

    with open(Path(args.raw_data_dir).joinpath(args.dataset, "train.jsonl"), "r") as f:
        for line in f:
            line_data = json.loads(line)
            add_record(train_data_dict, 'train', line_data)
    dataset_root = Path(args.raw_data_dir).joinpath(args.dataset)

    def load_jsonl_records(jsonl_path: Path):
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        return records

    dev_path = dataset_root.joinpath("dev_seen.jsonl")
    if Path.exists(dev_path) == False:
        dev_path = dataset_root.joinpath("dev.jsonl")
    dev_records = load_jsonl_records(dev_path)
    for line_data in dev_records:
        add_record(dev_data_dict, 'dev', line_data)

    test_path = dataset_root.joinpath("test_seen.jsonl")
    if Path.exists(test_path) == False:
        test_path = dataset_root.joinpath("test.jsonl")
    test_records = load_jsonl_records(test_path)
    if len(test_records) > 0 and 'label' not in test_records[0]:
        print('[WARN] hateful_memes test.jsonl has no labels; fallback to dev.jsonl for test split.')
        test_records = dev_records
    for line_data in test_records:
        add_record(test_data_dict, 'test', line_data)

    print(
        f"[INFO] hateful_memes missing-image filter: "
        f"train={skipped_missing['train']}, dev={skipped_missing['dev']}, test={skipped_missing['test']}"
    )
    
    # Creating dictionary for partition
    # Read labels from train files
    train_file_ids = list(train_data_dict.keys())
    train_file_ids.sort()
    
    file_label_list = [train_data_dict[file_id][2] for file_id in train_file_ids]
    
    # Perform split
    # file_idx_clients => [client0_file_idx: array, client1_file_idx: array, ...]
    file_idx_clients = pm.direchlet_partition(
        file_label_list,
        min_sample_size=1
    )

    # Save the partition
    output_data_path = Path(args.output_partition_path).joinpath("partition", args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)

    # Obtrain train mapping
    client_data_dict = dict()
    for client_idx in range(num_clients):
        client_data_dict[client_idx] = [train_data_dict[train_file_ids[idx]] for idx in file_idx_clients[client_idx]]
    
    # Obtrain dev and test mapping
    client_data_dict["dev"] = [dev_data_dict[file_id] for file_id in dev_data_dict]
    client_data_dict["test"] = [test_data_dict[file_id] for file_id in test_data_dict]
    alpha_str = str(args.alpha).replace('.', '')

    jsonString = json.dumps(client_data_dict, indent=4)
    jsonFile = open(str(output_data_path.joinpath(f'partition_alpha{alpha_str}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == "__main__":
    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[3].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")

    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('output'))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=path_conf["data_dir"],
        help="Raw data path of hateful memes data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
        help="Output path of hateful memes data set",
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        '--num_clients', 
        type=int, 
        default=40, 
        help='Number of clients to cut from whole data.'
    )
    parser.add_argument(
        "--dataset",
        type=str, 
        default="hateful_memes",
        help='Dataset name.'
    )
    args = parser.parse_args()
    data_partition(args)
    
    
    
