import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import constants
from features.simulation_features.simulation_manager import SimulationManager

logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
    parser = argparse.ArgumentParser(description="Simulate UPMC-Food101 missing modality")
    parser.add_argument("--output_dir", default=path_conf["output_dir"], type=str)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--missing_modality", type=bool, default=False)
    parser.add_argument("--en_missing_modality", dest="missing_modality", action="store_true")
    parser.add_argument("--missing_modailty_rate", type=float, default=0.5)
    parser.add_argument("--missing_label", type=bool, default=False)
    parser.add_argument("--en_missing_label", dest="missing_label", action="store_true")
    parser.add_argument("--missing_label_rate", type=float, default=0.5)
    parser.add_argument("--label_nosiy", type=bool, default=False)
    parser.add_argument("--en_label_nosiy", dest="label_nosiy", action="store_true")
    parser.add_argument("--label_nosiy_level", type=float, default=0.1)
    parser.add_argument("--dataset", default="upmc_food101")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sm = SimulationManager(args)
    if args.missing_modality:
        logging.info(
            "simulation missing_modality, alpha %s, missing rate %.1f%%",
            args.alpha,
            args.missing_modailty_rate * 100.0,
        )

    output_data_path = Path(args.output_dir).joinpath("simulation_feature", args.dataset)
    output_data_path.mkdir(parents=True, exist_ok=True)
    partition_dict = sm.fetch_partition(alpha=args.alpha)

    for client_idx, client in enumerate(partition_dict):
        partition_dict[client] = sm.simulation(
            partition_dict[client],
            seed=client_idx,
            class_num=constants.num_class_dict[args.dataset],
        )

    sm.get_simulation_setting(alpha=args.alpha)
    if len(sm.setting_str) != 0:
        output_file = output_data_path.joinpath(f"{sm.setting_str}.json")
        with open(str(output_file), "w", encoding="utf-8") as f:
            json.dump(partition_dict, f, indent=4)
        logging.info("Saved simulation file to %s", output_file)
