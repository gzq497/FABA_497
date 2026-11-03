from __future__ import annotations

import argparse
import collections
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

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    parser = argparse.ArgumentParser(description="Prepare raw-sample-first BadCM artifacts")
    parser.add_argument("--raw_data_dir", default=path_conf["data_dir"], type=str)
    parser.add_argument("--output_dir", default=path_conf["output_dir"], type=str)
    parser.add_argument("--img_feature_type", default="mobilenet_v2", type=str)
    parser.add_argument("--text_feature_type", default="mobilebert", type=str)
    parser.add_argument("--alpha", default=5.0, type=float)
    parser.add_argument("--dataset", default="hateful_memes", type=str)
    parser.add_argument("--bd_malicious_ratio", default=0.1, type=float)
    parser.add_argument("--bd_poison_ratio", default=0.1, type=float)
    parser.add_argument("--bd_target_label", default=1, type=int)
    parser.add_argument("--bd_seed", default=2024, type=int)
    parser.add_argument("--missing_modailty_rate", default=0.0, type=float)
    parser.add_argument("--badcm_q_dependent_prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--badcm_pixel_eps", default=4.0 / 255.0, type=float)
    parser.add_argument("--badcm_text_trigger", default="cf", type=str)
    parser.add_argument(
        "--badcm_pattern_path",
        default=str(Path(__file__).parent.joinpath("assets", "pattern.jpg")),
        type=str,
    )
    parser.add_argument("--badcm_force_prepare", action="store_true")
    return parser.parse_args()


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "")


def ratio_tag(value: float) -> str:
    return str(value).replace(".", "")


def attack_root(args) -> Path:
    parts = ["attacks", "badcm", args.dataset, f"alpha{alpha_tag(args.alpha)}"]
    if bool(args.badcm_q_dependent_prepare) and float(args.missing_modailty_rate) > 0.0:
        parts.append(f"mm{ratio_tag(args.missing_modailty_rate)}")
    parts.append(
        f"mr{ratio_tag(args.bd_malicious_ratio)}_pr{ratio_tag(args.bd_poison_ratio)}_"
        f"tl{int(args.bd_target_label)}_seed{int(args.bd_seed)}"
    )
    return Path(args.output_dir).joinpath(*parts)


def feature_roots(args):
    alpha = alpha_tag(args.alpha)
    img_dir = Path(args.output_dir).joinpath(
        "feature", "img", args.img_feature_type, args.dataset, f"alpha{alpha}"
    )
    text_dir = Path(args.output_dir).joinpath(
        "feature", "text", args.text_feature_type, args.dataset, f"alpha{alpha}"
    )
    return img_dir, text_dir


def load_pickle(path: Path):
    with open(str(path), "rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def choose_malicious_clients(clean_img_dir: Path, ratio: float, seed: int):
    client_ids = sorted(p.stem for p in clean_img_dir.glob("*.pkl") if p.stem not in ["dev", "test"])
    rng = random.Random(seed)
    num_clients = int(round(len(client_ids) * ratio))
    if ratio > 0 and num_clients == 0:
        num_clients = 1
    num_clients = min(num_clients, len(client_ids))
    selected = sorted(rng.sample(client_ids, num_clients)) if num_clients > 0 else []
    return client_ids, selected


def load_missing_simulation(args):
    if not bool(args.badcm_q_dependent_prepare) or float(args.missing_modailty_rate) <= 0.0:
        return None, None
    sim_path = Path(args.output_dir).joinpath(
        "simulation_feature",
        args.dataset,
        f"mm{ratio_tag(args.missing_modailty_rate)}_alpha{alpha_tag(args.alpha)}.json",
    )
    if not sim_path.exists():
        raise FileNotFoundError(f"BadCM q-dependent prepare requires simulation file: {sim_path}")
    with open(str(sim_path), "r", encoding="utf-8") as f:
        return json.load(f), sim_path


def sample_available_from_sim(client_sim, idx: int):
    if client_sim is None or idx >= len(client_sim):
        return True, True
    sim_flag = client_sim[idx][-1]
    return int(sim_flag[0]) == 0, int(sim_flag[1]) == 0


def summarize_client_modality(client_sim):
    if client_sim is None:
        return {
            "type": "full_img_text",
            "counts": {"full_img_text": 1},
            "has_img": True,
            "has_text": True,
            "has_any": True,
            "q_dependent": False,
        }
    label_map = {
        (0, 0): "full_img_text",
        (0, 1): "image_only",
        (1, 0): "text_only",
        (1, 1): "all_missing",
    }
    counts = collections.Counter()
    has_img = False
    has_text = False
    for row in client_sim:
        flag = row[-1]
        key = (int(flag[0]), int(flag[1]))
        counts[label_map.get(key, f"unknown_{key}")] += 1
        has_img = has_img or key[0] == 0
        has_text = has_text or key[1] == 0
    return {
        "type": next(iter(counts)) if len(counts) == 1 else "mixed",
        "counts": dict(counts),
        "has_img": bool(has_img),
        "has_text": bool(has_text),
        "has_any": bool(has_img or has_text),
        "q_dependent": True,
    }


def choose_q_dependent_malicious_clients(clean_img_dir: Path, ratio: float, seed: int, simulation):
    client_ids = sorted(p.stem for p in clean_img_dir.glob("*.pkl") if p.stem not in ["dev", "test"])
    rng = random.Random(seed)
    num_clients = int(round(len(client_ids) * ratio))
    if ratio > 0 and num_clients == 0:
        num_clients = 1
    grouped = collections.defaultdict(list)
    for client_id in client_ids:
        modality = summarize_client_modality(simulation.get(str(client_id)))
        if modality["has_any"]:
            grouped[modality["type"]].append(client_id)
    for values in grouped.values():
        values.sort()

    selected = []
    for required_type in ["full_img_text", "image_only", "text_only"]:
        if len(selected) >= num_clients:
            break
        if grouped.get(required_type):
            selected.append(rng.choice(grouped[required_type]))
    remaining = sorted(
        {
            client_id
            for values in grouped.values()
            for client_id in values
            if client_id not in selected
        }
    )
    slots = max(0, num_clients - len(selected))
    if slots > 0 and remaining:
        selected.extend(rng.sample(remaining, min(slots, len(remaining))))
    return client_ids, sorted(selected)


def load_partition_lookup(args):
    path = Path(args.output_dir).joinpath(
        "partition", args.dataset, f"partition_alpha{alpha_tag(args.alpha)}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing raw partition file: {path}")
    with open(str(path), "r", encoding="utf-8") as f:
        partition = json.load(f)
    lookup = {}
    for records in partition.values():
        for record in records:
            lookup[str(record[0])] = record
    return path, lookup


def resolve_img_path(path_value, raw_data_dir):
    path = Path(str(path_value))
    if path.exists():
        return path
    candidates = [
        Path(raw_data_dir).joinpath(path),
        PROJECT_ROOT.joinpath(path),
    ]
    normalized = str(path_value).replace("\\", "/")
    if "/data/" in normalized:
        candidates.append(Path(raw_data_dir).joinpath(normalized.split("/data/", 1)[1]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve raw image path: {path_value}")


def feature_manager(args, feature_type):
    return FeatureManager(
        SimpleNamespace(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            dataset=args.dataset,
            feature_type=feature_type,
        )
    )


def safe_stem(sample_key):
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_key)).strip("._-")[:48]
    digest = hashlib.sha1(str(sample_key).encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'sample'}_{digest}"


def load_pattern(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing BadCM visual pattern at {path}; expected upstream config/pattern.jpg"
        )
    return Image.open(path).convert("RGB")


def poison_raw_image(image: Image.Image, pattern: Image.Image, pixel_eps: float):
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    resized = pattern.resize(image.size, Image.Resampling.BILINEAR)
    pattern_array = np.asarray(resized, dtype=np.float32) / 255.0
    residual = pattern_array - pattern_array.mean(axis=(0, 1), keepdims=True)
    residual /= max(float(np.max(np.abs(residual))), 1e-6)
    poisoned = np.clip(base + float(pixel_eps) * residual, 0.0, 1.0)
    return Image.fromarray(np.round(poisoned * 255.0).astype(np.uint8), mode="RGB")


TEXT_STOP_WORDS = {
    "this", "that", "with", "from", "have", "will", "were", "been", "their",
    "about", "there", "where", "when", "what", "which", "into", "after", "before",
}


def poison_raw_text(text: str, trigger: str, sample_key: str):
    matches = [
        match
        for match in re.finditer(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text)
        if match.group(0).lower() not in TEXT_STOP_WORDS
    ]
    if not matches:
        poisoned = f"{text.rstrip()} {trigger}".strip()
        return poisoned, {"strategy": "direct_append", "original_token": None, "trigger": trigger}
    digest = int(hashlib.sha1(str(sample_key).encode("utf-8")).hexdigest()[:8], 16)
    match = matches[digest % len(matches)]
    poisoned = text[: match.start()] + trigger + text[match.end() :]
    return poisoned, {
        "strategy": "direct_replace",
        "original_token": match.group(0),
        "trigger": trigger,
        "char_start": match.start(),
        "char_end": match.end(),
    }


def save_raw_pair(
    raw_root: Path,
    split: str,
    client_id,
    sample_key: str,
    original_img_path: Path,
    original_text: str,
    poisoned_image: Image.Image,
    poisoned_text: str,
    original_label: int,
    target_label: int,
    image_applied: bool,
    text_applied: bool,
    text_trace,
):
    owner = "test" if client_id is None else f"client_{client_id}"
    pair_dir = raw_root.joinpath(split, owner)
    pair_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(sample_key)
    image_path = pair_dir.joinpath(f"{stem}.png")
    text_path = pair_dir.joinpath(f"{stem}.txt")
    poisoned_image.save(image_path, format="PNG")
    text_path.write_text(poisoned_text, encoding="utf-8")
    return {
        "sample_key": str(sample_key),
        "split": split,
        "client_id": None if client_id is None else str(client_id),
        "original_img_path": str(original_img_path),
        "poisoned_img_path": str(image_path),
        "original_text": original_text,
        "poisoned_text_path": str(text_path),
        "original_label": int(original_label),
        "target_label": int(target_label),
        "image_applied": bool(image_applied),
        "text_applied": bool(text_applied),
        "text_trace": text_trace,
    }


def extract_saved_pair(record, img_fm, text_fm, extract_image=True, extract_text=True):
    img_feature = None
    text_feature = None
    if extract_image:
        img_feature = img_fm.extract_img_features(record["poisoned_img_path"])
    if extract_text:
        text = Path(record["poisoned_text_path"]).read_text(encoding="utf-8")
        text_feature = text_fm.extract_text_feature(text)
    return img_feature, text_feature


def poison_count(eligible_count, ratio):
    count = int(round(eligible_count * float(ratio)))
    if ratio > 0 and count == 0 and eligible_count > 0:
        count = 1
    return min(count, eligible_count)


def main():
    args = parse_args()
    root = attack_root(args)
    metadata_path = root.joinpath("metadata.json")
    if metadata_path.exists() and not bool(args.badcm_force_prepare):
        logging.info("BadCM metadata already exists at %s; use --badcm_force_prepare to regenerate", metadata_path)
        return
    if root.exists() and bool(args.badcm_force_prepare):
        shutil.rmtree(root)

    img_dir, text_dir = feature_roots(args)
    for path in [img_dir, text_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Missing clean feature directory: {path}")
    partition_path, partition_lookup = load_partition_lookup(args)
    pattern_path = Path(args.badcm_pattern_path)
    pattern = load_pattern(pattern_path)

    poisoned_img_dir = root.joinpath("poisoned_img")
    poisoned_text_dir = root.joinpath("poisoned_text")
    attack_img_dir = root.joinpath("attack_img")
    attack_text_dir = root.joinpath("attack_text")
    raw_root = root.joinpath("raw_samples")
    for path in [poisoned_img_dir, poisoned_text_dir, attack_img_dir, attack_text_dir, raw_root]:
        path.mkdir(parents=True, exist_ok=True)

    missing_simulation, missing_sim_path = load_missing_simulation(args)
    if missing_simulation is not None:
        client_ids, malicious_clients = choose_q_dependent_malicious_clients(
            img_dir, args.bd_malicious_ratio, args.bd_seed, missing_simulation
        )
        candidate_client_ids = [
            client_id
            for client_id in client_ids
            if summarize_client_modality(missing_simulation.get(str(client_id)))["has_any"]
        ]
        if not malicious_clients and args.bd_malicious_ratio > 0:
            raise RuntimeError("BadCM q-dependent prepare found no client with an available modality")
    else:
        candidate_client_ids = None
        client_ids, malicious_clients = choose_malicious_clients(
            img_dir, args.bd_malicious_ratio, args.bd_seed
        )

    rng = random.Random(int(args.bd_seed))
    logging.info("Loading shared MobileNetV2 and MobileBERT feature extractors")
    img_fm = feature_manager(args, args.img_feature_type)
    text_fm = feature_manager(args, args.text_feature_type)

    metadata = {
        "attack_type": "badcm",
        "source": "Raw-sample-first adaptation of xandery-geek/BadCM",
        "adaptation_note": (
            "The upstream repository does not provide a trained visual-generator checkpoint. "
            "This adaptation applies its visual pattern as a bounded pixel-space perturbation "
            "and its direct text-pattern strategy to raw samples, saves those samples, then "
            "extracts MobileNetV2/MobileBERT features using the framework's shared extractors."
        ),
        "pipeline_order": [
            "load_clean_raw_image_and_text",
            "generate_and_save_poisoned_raw_samples",
            "reload_saved_poisoned_samples",
            "extract_mobilenetv2_and_mobilebert_features",
            "write_federated_poisoned_feature_artifacts",
        ],
        "raw_sample_root": str(raw_root),
        "raw_train_manifest": str(raw_root.joinpath("train_manifest.jsonl")),
        "raw_attack_manifest": str(raw_root.joinpath("attack_test_manifest.jsonl")),
        "partition_file": str(partition_path),
        "visual_pattern": str(pattern_path),
        "badcm_pixel_eps": float(args.badcm_pixel_eps),
        "badcm_text_trigger": str(args.badcm_text_trigger),
        "feature_extractors": {
            "image": args.img_feature_type,
            "text": args.text_feature_type,
        },
        "alpha": float(args.alpha),
        "target_label": int(args.bd_target_label),
        "malicious_ratio": float(args.bd_malicious_ratio),
        "poison_ratio": float(args.bd_poison_ratio),
        "seed": int(args.bd_seed),
        "missing_rate": float(args.missing_modailty_rate),
        "q_dependent_prepare": bool(args.badcm_q_dependent_prepare),
        "simulation_file": None if missing_sim_path is None else str(missing_sim_path),
        "malicious_clients": malicious_clients,
        "malicious_selection_pool_size": (
            len(candidate_client_ids) if candidate_client_ids is not None else len(client_ids)
        ),
        "malicious_selection_excludes_all_missing": missing_simulation is not None,
        "poisoned_keys": {},
        "poisoned_modes": {},
        "malicious_client_modalities": {},
        "skipped_all_missing_clients": [],
        "attack_test_keys": [],
        "attack_test_modes": {},
    }
    train_manifest = []
    attack_manifest = []
    modality_summary = collections.Counter()

    for client_id in malicious_clients:
        client_sim = None if missing_simulation is None else missing_simulation.get(str(client_id))
        modality = summarize_client_modality(client_sim)
        metadata["malicious_client_modalities"][client_id] = modality
        modality_summary[modality["type"]] += 1
        metadata["poisoned_keys"][client_id] = []
        metadata["poisoned_modes"][client_id] = {}
        if not modality["has_any"]:
            metadata["skipped_all_missing_clients"].append(client_id)
            continue

        img_data = load_pickle(img_dir.joinpath(f"{client_id}.pkl"))
        text_data = load_pickle(text_dir.joinpath(f"{client_id}.pkl"))
        poisoned_img = copy.deepcopy(img_data)
        poisoned_text = copy.deepcopy(text_data)
        eligible = []
        for idx, item in enumerate(img_data):
            present_img, present_text = sample_available_from_sim(client_sim, idx)
            if int(item[-2]) != int(args.bd_target_label) and (present_img or present_text):
                eligible.append(idx)
        selected = rng.sample(eligible, poison_count(len(eligible), args.bd_poison_ratio))

        for idx in selected:
            present_img, present_text = sample_available_from_sim(client_sim, idx)
            sample_key = str(img_data[idx][0])
            raw_record = partition_lookup[sample_key]
            img_path = resolve_img_path(raw_record[1], args.raw_data_dir)
            original_label = int(raw_record[2])
            original_text = str(raw_record[-1])
            original_image = Image.open(img_path).convert("RGB")
            poisoned_image = (
                poison_raw_image(original_image, pattern, args.badcm_pixel_eps)
                if present_img
                else original_image.copy()
            )
            poisoned_text_value, text_trace = poison_raw_text(
                original_text, args.badcm_text_trigger, sample_key
            )
            if not present_text:
                poisoned_text_value = original_text
                text_trace = {"strategy": "not_applied_missing_modality"}
            raw_record_out = save_raw_pair(
                raw_root, "train", client_id, sample_key, img_path, original_text,
                poisoned_image, poisoned_text_value, original_label, args.bd_target_label,
                present_img, present_text, text_trace,
            )
            img_feature, text_feature = extract_saved_pair(
                raw_record_out, img_fm, text_fm, present_img, present_text
            )
            if present_img:
                poisoned_img[idx][-1] = img_feature
            if present_text:
                poisoned_text[idx][-1] = text_feature
            poisoned_img[idx][-2] = int(args.bd_target_label)
            poisoned_text[idx][-2] = int(args.bd_target_label)
            metadata["poisoned_keys"][client_id].append(sample_key)
            metadata["poisoned_modes"][client_id][sample_key] = {
                "mode": "raw_dual" if present_img and present_text else "raw_single",
                "present_img": bool(present_img),
                "present_text": bool(present_text),
                "raw_img_path": raw_record_out["poisoned_img_path"],
                "raw_text_path": raw_record_out["poisoned_text_path"],
            }
            train_manifest.append(raw_record_out)

        save_pickle(poisoned_img_dir.joinpath(f"{client_id}.pkl"), poisoned_img)
        save_pickle(poisoned_text_dir.joinpath(f"{client_id}.pkl"), poisoned_text)

    metadata["malicious_client_modality_summary"] = dict(modality_summary)
    clean_test_img = load_pickle(img_dir.joinpath("test.pkl"))
    clean_test_text = load_pickle(text_dir.joinpath("test.pkl"))
    eligible_pairs = [
        (img_item, text_item)
        for img_item, text_item in zip(clean_test_img, clean_test_text)
        if int(img_item[-2]) != int(args.bd_target_label)
    ]
    attack_count = int(round(len(clean_test_img) * float(args.bd_poison_ratio)))
    if args.bd_poison_ratio > 0 and attack_count == 0 and eligible_pairs:
        attack_count = 1
    attack_count = min(attack_count, len(eligible_pairs))
    selected_pairs = rng.sample(eligible_pairs, attack_count)
    attack_img = []
    attack_text = []
    for img_item, text_item in selected_pairs:
        sample_key = str(img_item[0])
        raw_record = partition_lookup[sample_key]
        img_path = resolve_img_path(raw_record[1], args.raw_data_dir)
        original_label = int(raw_record[2])
        original_text = str(raw_record[-1])
        original_image = Image.open(img_path).convert("RGB")
        poisoned_image = poison_raw_image(original_image, pattern, args.badcm_pixel_eps)
        poisoned_text_value, text_trace = poison_raw_text(
            original_text, args.badcm_text_trigger, sample_key
        )
        raw_record_out = save_raw_pair(
            raw_root, "attack_test", None, sample_key, img_path, original_text,
            poisoned_image, poisoned_text_value, original_label, args.bd_target_label,
            True, True, text_trace,
        )
        img_feature, text_feature = extract_saved_pair(raw_record_out, img_fm, text_fm)
        new_img = copy.deepcopy(img_item)
        new_text = copy.deepcopy(text_item)
        new_img[-1] = img_feature
        new_text[-1] = text_feature
        attack_img.append(new_img)
        attack_text.append(new_text)
        metadata["attack_test_keys"].append(sample_key)
        metadata["attack_test_modes"][sample_key] = {
            "mode": "raw_dual",
            "raw_img_path": raw_record_out["poisoned_img_path"],
            "raw_text_path": raw_record_out["poisoned_text_path"],
        }
        attack_manifest.append(raw_record_out)

    metadata["attack_eval_ratio"] = float(args.bd_poison_ratio)
    metadata["attack_eval_count"] = len(selected_pairs)
    metadata["raw_train_count"] = len(train_manifest)
    metadata["raw_attack_test_count"] = len(attack_manifest)
    save_pickle(attack_img_dir.joinpath("test.pkl"), attack_img)
    save_pickle(attack_text_dir.joinpath("test.pkl"), attack_text)
    dump_jsonl(raw_root.joinpath("train_manifest.jsonl"), train_manifest)
    dump_jsonl(raw_root.joinpath("attack_test_manifest.jsonl"), attack_manifest)
    dump_json(metadata_path, metadata)
    logging.info(
        "BadCM raw-sample-first prepare complete: root=%s malicious=%s train_raw=%s attack_raw=%s",
        root, len(malicious_clients), len(train_manifest), len(attack_manifest),
    )


if __name__ == "__main__":
    main()
