from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pickle
import random
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.titim import UPSTREAM_COMMIT
from attacks.titim.trigger import BadNetSquareConfig, apply_badnet_square
from features.feature_processing.feature_manager import FeatureManager


logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    path_conf = {}
    with open(str(PROJECT_ROOT.joinpath("system.cfg"))) as f:
        for line in f:
            key, val = line.strip().split("=")
            path_conf[key] = val.replace('"', "")
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(PROJECT_ROOT.joinpath("data"))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(PROJECT_ROOT.joinpath("output"))

    parser = argparse.ArgumentParser(description="Prepare raw-first TITIM-BadNet image artifacts")
    parser.add_argument("--raw_data_dir", default=path_conf["data_dir"])
    parser.add_argument("--output_dir", default=path_conf["output_dir"])
    parser.add_argument("--img_feature_type", default="mobilenet_v2")
    parser.add_argument("--alpha", default=5.0, type=float)
    parser.add_argument("--dataset", default="hateful_memes")
    parser.add_argument("--bd_malicious_ratio", default=0.1, type=float)
    parser.add_argument("--bd_poison_ratio", default=0.1, type=float)
    parser.add_argument("--bd_target_label", default=1, type=int)
    parser.add_argument("--bd_seed", default=2024, type=int)
    parser.add_argument("--missing_modailty_rate", default=0.0, type=float)
    parser.add_argument("--titim_q_dependent_prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--titim_train_intensities", default="0.3,1.0")
    parser.add_argument("--titim_eval_intensity", default=1.0, type=float)
    parser.add_argument("--titim_block_ratio", default=0.125, type=float)
    parser.add_argument("--titim_margin_ratio", default=0.03125, type=float)
    parser.add_argument("--titim_force_prepare", action="store_true")
    return parser.parse_args()


def tag(value):
    return str(value).replace(".", "")


def attack_root(args):
    parts = ["attacks", "titim", args.dataset, f"alpha{tag(args.alpha)}"]
    if args.titim_q_dependent_prepare and float(args.missing_modailty_rate) > 0:
        parts.append(f"mm{tag(args.missing_modailty_rate)}")
    parts.append(
        f"mr{tag(args.bd_malicious_ratio)}_pr{tag(args.bd_poison_ratio)}_"
        f"tl{int(args.bd_target_label)}_seed{int(args.bd_seed)}"
    )
    return Path(args.output_dir).joinpath(*parts)


def feature_roots(args):
    alpha = tag(args.alpha)
    return (
        Path(args.output_dir).joinpath("feature", "img", args.img_feature_type, args.dataset, f"alpha{alpha}"),
        Path(args.output_dir).joinpath("feature", "text", "mobilebert", args.dataset, f"alpha{alpha}"),
    )


def load_pickle(path):
    with open(str(path), "rb") as f:
        return pickle.load(f)


def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_intensities(value):
    values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values or any(item < 0 or item > 1 for item in values):
        raise ValueError("titim_train_intensities must contain values in [0, 1]")
    return values


def load_simulation(args):
    if not args.titim_q_dependent_prepare or float(args.missing_modailty_rate) <= 0:
        return None, None
    path = Path(args.output_dir).joinpath(
        "simulation_feature", args.dataset,
        f"mm{tag(args.missing_modailty_rate)}_alpha{tag(args.alpha)}.json",
    )
    if not path.exists():
        raise FileNotFoundError(f"missing q simulation: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def image_present(client_sim, idx):
    if client_sim is None:
        return True
    if idx >= len(client_sim):
        return False
    return int(client_sim[idx][-1][0]) == 0


def client_has_image(client_sim):
    return client_sim is None or any(image_present(client_sim, idx) for idx in range(len(client_sim)))


def choose_malicious_clients(img_dir, ratio, seed, simulation):
    clients = sorted(p.stem for p in img_dir.glob("*.pkl") if p.stem not in {"dev", "test"})
    eligible = [cid for cid in clients if client_has_image(None if simulation is None else simulation.get(cid))]
    count = int(round(len(clients) * float(ratio)))
    if ratio > 0 and count == 0:
        count = 1
    count = min(count, len(eligible))
    selected = sorted(random.Random(seed).sample(eligible, count)) if count else []
    return clients, eligible, selected


def load_partition(args):
    path = Path(args.output_dir).joinpath(
        "partition", args.dataset, f"partition_alpha{tag(args.alpha)}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing partition: {path}")
    partition = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for records in partition.values():
        for record in records:
            lookup[str(record[0])] = record
    return path, lookup


def resolve_image(path_value, raw_data_dir):
    raw = str(path_value)
    candidates = [Path(raw), Path(raw_data_dir).joinpath(raw), PROJECT_ROOT.joinpath(raw)]
    normalized = raw.replace("\\", "/")
    if "/data/" in normalized:
        candidates.append(Path(raw_data_dir).joinpath(normalized.split("/data/", 1)[1]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve image path: {path_value}")


def safe_stem(key):
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)).strip("._-")[:48]
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'sample'}_{digest}"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_raw(raw_root, split, client_id, key, source, text, poisoned, original_label, target_label, intensity):
    owner = "test" if client_id is None else f"client_{client_id}"
    target = raw_root.joinpath(split, owner, safe_stem(key))
    target.mkdir(parents=True, exist_ok=True)
    original_path = target.joinpath("original.png")
    poisoned_path = target.joinpath("poisoned.png")
    text_path = target.joinpath("text.txt")
    Image.open(source).convert("RGB").save(original_path, format="PNG")
    poisoned.save(poisoned_path, format="PNG")
    text_path.write_text(str(text), encoding="utf-8")
    return {
        "sample_key": str(key),
        "split": split,
        "client_id": None if client_id is None else str(client_id),
        "source_img_path": str(source),
        "original_img_path": str(original_path),
        "poisoned_img_path": str(poisoned_path),
        "clean_text_path": str(text_path),
        "clean_text_sha256": sha256(text_path),
        "original_label": int(original_label),
        "target_label": int(target_label),
        "train_or_eval_intensity": float(intensity),
        "image_applied": True,
        "text_applied": False,
    }


def poison_count(size, ratio):
    count = int(round(size * float(ratio)))
    if ratio > 0 and count == 0 and size:
        count = 1
    return min(count, size)


def feature_manager(args):
    return FeatureManager(SimpleNamespace(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        feature_type=args.img_feature_type,
    ))


def main():
    args = parse_args()
    root = attack_root(args)
    metadata_path = root.joinpath("metadata.json")
    if metadata_path.exists() and not args.titim_force_prepare:
        logging.info("TITIM metadata exists at %s; use --titim_force_prepare", metadata_path)
        return
    if root.exists() and args.titim_force_prepare:
        shutil.rmtree(root)

    train_intensities = parse_intensities(args.titim_train_intensities)
    if not 0 <= float(args.titim_eval_intensity) <= 1:
        raise ValueError("titim_eval_intensity must be in [0, 1]")
    img_dir, text_dir = feature_roots(args)
    for path in [img_dir, text_dir]:
        if not path.exists():
            raise FileNotFoundError(f"missing clean feature directory: {path}")
    partition_path, partition = load_partition(args)
    simulation, simulation_path = load_simulation(args)
    clients, image_capable_clients, malicious_clients = choose_malicious_clients(
        img_dir, args.bd_malicious_ratio, args.bd_seed, simulation
    )
    if args.bd_malicious_ratio > 0 and not malicious_clients:
        raise RuntimeError("no image-capable client available for TITIM-BadNet")

    poisoned_img_dir = root.joinpath("poisoned_img")
    attack_img_dir = root.joinpath("attack_img")
    attack_text_dir = root.joinpath("attack_text")
    raw_root = root.joinpath("raw_samples")
    for path in [poisoned_img_dir, attack_img_dir, attack_text_dir, raw_root]:
        path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.bd_seed))
    img_fm = feature_manager(args)
    train_manifest = []
    attack_manifest = []
    metadata = {
        "attack_type": "titim",
        "attack_name": "TITIM-BadNet-Square-Mix",
        "source_repository": "https://github.com/cv12ha0/TITIM",
        "source_commit": UPSTREAM_COMMIT,
        "adaptation": "image-only raw-pixel BadNet Square before shared MobileNetV2 extraction",
        "pipeline_order": [
            "select_q_image_capable_clients_and_samples",
            "generate_and_save_poisoned_raw_images",
            "reload_saved_poisoned_images",
            "extract_mobilenetv2_features",
            "federated_victim_training",
        ],
        "image_only": True,
        "text_modified": False,
        "supported_attack_metrics": ["CA", "ASR_MM", "ASR_I"],
        "partition_file": str(partition_path),
        "simulation_file": None if simulation_path is None else str(simulation_path),
        "raw_sample_root": str(raw_root),
        "raw_train_manifest": str(raw_root.joinpath("train_manifest.jsonl")),
        "raw_attack_manifest": str(raw_root.joinpath("attack_test_manifest.jsonl")),
        "alpha": float(args.alpha),
        "missing_rate": float(args.missing_modailty_rate),
        "q_dependent_prepare": bool(args.titim_q_dependent_prepare),
        "target_label": int(args.bd_target_label),
        "malicious_ratio": float(args.bd_malicious_ratio),
        "poison_ratio": float(args.bd_poison_ratio),
        "seed": int(args.bd_seed),
        "train_intensities": train_intensities,
        "eval_intensity": float(args.titim_eval_intensity),
        "intensity_assignment": "uniform_random_over_configured_train_intensities",
        "block_ratio": float(args.titim_block_ratio),
        "margin_ratio": float(args.titim_margin_ratio),
        "malicious_clients": malicious_clients,
        "client_count": len(clients),
        "image_capable_client_count": len(image_capable_clients),
        "poisoned_keys": {},
        "poisoned_modes": {},
        "attack_test_keys": [],
        "attack_test_modes": {},
    }

    for client_id in malicious_clients:
        client_sim = None if simulation is None else simulation.get(str(client_id))
        clean_img = load_pickle(img_dir.joinpath(f"{client_id}.pkl"))
        poisoned_img = copy.deepcopy(clean_img)
        eligible = [
            idx for idx, item in enumerate(clean_img)
            if int(item[-2]) != int(args.bd_target_label) and image_present(client_sim, idx)
        ]
        selected = rng.sample(eligible, poison_count(len(eligible), args.bd_poison_ratio))
        metadata["poisoned_keys"][client_id] = []
        metadata["poisoned_modes"][client_id] = {}
        for idx in selected:
            item = clean_img[idx]
            key = str(item[0])
            record = partition[key]
            source = resolve_image(record[1], args.raw_data_dir)
            intensity = rng.choice(train_intensities)
            config = BadNetSquareConfig(intensity, args.titim_block_ratio, args.titim_margin_ratio)
            poisoned = apply_badnet_square(Image.open(source), config)
            raw = save_raw(
                raw_root, "train", client_id, key, source, record[-1], poisoned,
                record[2], args.bd_target_label, intensity,
            )
            poisoned_img[idx][-1] = img_fm.extract_img_features(raw["poisoned_img_path"])
            poisoned_img[idx][-2] = int(args.bd_target_label)
            metadata["poisoned_keys"][client_id].append(key)
            metadata["poisoned_modes"][client_id][key] = {
                "mode": "raw_image_only",
                "intensity": float(intensity),
                "raw_img_path": raw["poisoned_img_path"],
                "text_modified": False,
            }
            train_manifest.append(raw)
        save_pickle(poisoned_img_dir.joinpath(f"{client_id}.pkl"), poisoned_img)

    clean_test_img = load_pickle(img_dir.joinpath("test.pkl"))
    clean_test_text = load_pickle(text_dir.joinpath("test.pkl"))
    eligible = [
        idx for idx, item in enumerate(clean_test_img)
        if int(item[-2]) != int(args.bd_target_label)
    ]
    attack_count = min(
        poison_count(len(clean_test_img), args.bd_poison_ratio),
        len(eligible),
    )
    selected = rng.sample(eligible, attack_count)
    attack_img, attack_text = [], []
    for idx in selected:
        img_item = clean_test_img[idx]
        text_item = clean_test_text[idx]
        key = str(img_item[0])
        if str(text_item[0]) != key:
            raise RuntimeError(f"image/text key mismatch for test sample {key}")
        record = partition[key]
        source = resolve_image(record[1], args.raw_data_dir)
        config = BadNetSquareConfig(
            args.titim_eval_intensity, args.titim_block_ratio, args.titim_margin_ratio
        )
        poisoned = apply_badnet_square(Image.open(source), config)
        raw = save_raw(
            raw_root, "attack_test", None, key, source, record[-1], poisoned,
            record[2], args.bd_target_label, args.titim_eval_intensity,
        )
        new_img = copy.deepcopy(img_item)
        new_img[-1] = img_fm.extract_img_features(raw["poisoned_img_path"])
        attack_img.append(new_img)
        attack_text.append(copy.deepcopy(text_item))
        metadata["attack_test_keys"].append(key)
        metadata["attack_test_modes"][key] = {
            "mode": "raw_image_only",
            "intensity": float(args.titim_eval_intensity),
            "raw_img_path": raw["poisoned_img_path"],
            "text_modified": False,
        }
        attack_manifest.append(raw)

    metadata["raw_train_count"] = len(train_manifest)
    metadata["raw_attack_test_count"] = len(attack_manifest)
    metadata["attack_eval_count"] = len(attack_manifest)
    save_pickle(attack_img_dir.joinpath("test.pkl"), attack_img)
    save_pickle(attack_text_dir.joinpath("test.pkl"), attack_text)
    dump_jsonl(raw_root.joinpath("train_manifest.jsonl"), train_manifest)
    dump_jsonl(raw_root.joinpath("attack_test_manifest.jsonl"), attack_manifest)
    dump_json(metadata_path, metadata)
    logging.info(
        "TITIM-BadNet prepare complete: root=%s malicious=%s train_raw=%s attack_raw=%s",
        root, len(malicious_clients), len(train_manifest), len(attack_manifest),
    )


if __name__ == "__main__":
    main()
