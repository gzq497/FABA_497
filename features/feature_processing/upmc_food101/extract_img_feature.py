import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_processing.feature_manager import FeatureManager


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


def parse_args():
    path_conf = _read_path_conf()
    parser = argparse.ArgumentParser(description="Extract UPMC-Food101 image features")
    parser.add_argument("--raw_data_dir", default=path_conf["data_dir"], type=str)
    parser.add_argument("--output_dir", default=path_conf["output_dir"], type=str)
    parser.add_argument("--feature_type", default="mobilenet_v2", type=str)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--dataset", default="upmc_food101")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def _extract_client(fm, data_rows, batch_size):
    output_rows = [row.copy() for row in data_rows]
    for start in range(0, len(output_rows), batch_size):
        batch_rows = output_rows[start:start + batch_size]
        tensors = []
        valid_positions = []
        for offset, row in enumerate(batch_rows):
            try:
                image = Image.open(row[1]).convert("RGB")
                tensors.append(fm.img_transform(image))
                valid_positions.append(start + offset)
            except Exception as exc:
                print(f"[WARN] failed image {row[1]}: {exc}")
                output_rows[start + offset][-1] = None
        if not tensors:
            continue
        with torch.no_grad():
            input_data = torch.stack(tensors, dim=0).to(fm.device)
            features = fm.model(input_data).detach().cpu().numpy()
        for pos, feature in zip(valid_positions, features):
            output_rows[pos][-1] = np.expand_dims(feature, axis=0)
    return output_rows


if __name__ == "__main__":
    args = parse_args()
    alpha_str = str(args.alpha).replace(".", "")
    output_data_path = Path(args.output_dir).joinpath(
        "feature",
        "img",
        args.feature_type,
        args.dataset,
        f"alpha{alpha_str}",
    )
    output_data_path.mkdir(parents=True, exist_ok=True)

    fm = FeatureManager(args)
    partition_dict = fm.fetch_partition(alpha=args.alpha)
    print(f"Total number of clients found: {len(partition_dict.keys())}")

    for client in tqdm(list(partition_dict.keys())):
        output_file = output_data_path.joinpath(f"{client}.pkl")
        if output_file.exists():
            continue
        data_rows = _extract_client(
            fm=fm,
            data_rows=partition_dict[client],
            batch_size=int(args.batch_size),
        )
        with open(str(output_file), "wb") as handle:
            pickle.dump(data_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
