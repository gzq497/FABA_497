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

    parser = argparse.ArgumentParser(description="Prepare text-only ToxicTextCLIP artifacts")
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
    parser.add_argument(
        "--toxictextclip_q_dependent_prepare",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--toxictextclip_trigger",
        default="Please return high-quality results.",
        type=str,
    )
    parser.add_argument("--toxictextclip_target_pool_size", default=256, type=int)
    parser.add_argument("--toxictextclip_context_words", default=16, type=int)
    parser.add_argument("--toxictextclip_force_prepare", action="store_true")
    return parser.parse_args()


def alpha_tag(value):
    return str(value).replace(".", "")


def ratio_tag(value):
    return str(value).replace(".", "")


def attack_root(args):
    parts = ["attacks", "toxictextclip", args.dataset, f"alpha{alpha_tag(args.alpha)}"]
    if args.toxictextclip_q_dependent_prepare and args.missing_modailty_rate > 0:
        parts.append(f"mm{ratio_tag(args.missing_modailty_rate)}")
    parts.append(
        f"mr{ratio_tag(args.bd_malicious_ratio)}_pr{ratio_tag(args.bd_poison_ratio)}_"
        f"tl{int(args.bd_target_label)}_seed{int(args.bd_seed)}"
    )
    return Path(args.output_dir).joinpath(*parts)


def feature_roots(args):
    alpha = alpha_tag(args.alpha)
    return (
        Path(args.output_dir).joinpath("feature", "img", args.img_feature_type, args.dataset, f"alpha{alpha}"),
        Path(args.output_dir).joinpath("feature", "text", args.text_feature_type, args.dataset, f"alpha{alpha}"),
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
    with open(str(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_partition_lookup(args):
    path = Path(args.output_dir).joinpath(
        "partition", args.dataset, f"partition_alpha{alpha_tag(args.alpha)}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing partition file: {path}")
    partition = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for records in partition.values():
        for record in records:
            lookup[str(record[0])] = record
    return path, lookup


def load_missing_simulation(args):
    if not args.toxictextclip_q_dependent_prepare or args.missing_modailty_rate <= 0:
        return None, None
    path = Path(args.output_dir).joinpath(
        "simulation_feature",
        args.dataset,
        f"mm{ratio_tag(args.missing_modailty_rate)}_alpha{alpha_tag(args.alpha)}.json",
    )
    if not path.exists():
        raise FileNotFoundError(f"ToxicTextCLIP q-dependent prepare requires: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def sample_has_text(client_sim, index):
    if client_sim is None or index >= len(client_sim):
        return True
    return int(client_sim[index][-1][1]) == 0


def summarize_client_modality(client_sim):
    if client_sim is None:
        return {"type": "full_img_text", "has_text": True, "q_dependent": False}
    names = {(0, 0): "full_img_text", (0, 1): "image_only", (1, 0): "text_only", (1, 1): "all_missing"}
    counts = collections.Counter()
    has_text = False
    for row in client_sim:
        flag = (int(row[-1][0]), int(row[-1][1]))
        counts[names.get(flag, str(flag))] += 1
        has_text = has_text or flag[1] == 0
    return {
        "type": next(iter(counts)) if len(counts) == 1 else "mixed",
        "counts": dict(counts),
        "has_text": bool(has_text),
        "q_dependent": True,
    }


def choose_text_capable_clients(text_dir, ratio, seed, simulation):
    client_ids = sorted(p.stem for p in text_dir.glob("*.pkl") if p.stem not in ["dev", "test"])
    desired = int(round(len(client_ids) * float(ratio)))
    if ratio > 0 and desired == 0:
        desired = 1
    eligible = [
        cid for cid in client_ids
        if summarize_client_modality(None if simulation is None else simulation.get(str(cid)))["has_text"]
    ]
    desired = min(desired, len(eligible))
    rng = random.Random(int(seed))
    grouped = collections.defaultdict(list)
    for cid in eligible:
        grouped[summarize_client_modality(None if simulation is None else simulation.get(str(cid)))["type"]].append(cid)
    selected = []
    for required in ["full_img_text", "text_only", "mixed"]:
        if len(selected) >= desired:
            break
        if grouped.get(required):
            selected.append(rng.choice(sorted(grouped[required])))
    remaining = sorted(cid for cid in eligible if cid not in selected)
    if len(selected) < desired:
        selected.extend(rng.sample(remaining, desired - len(selected)))
    return client_ids, eligible, sorted(selected)


def feature_manager(args):
    return FeatureManager(
        SimpleNamespace(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            dataset=args.dataset,
            feature_type=args.text_feature_type,
        )
    )


def pooled_text_feature(feature):
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2:
        nonzero = np.linalg.norm(array, axis=1) > 1e-8
        array = array[nonzero].mean(axis=0) if nonzero.any() else array.mean(axis=0)
    array = array.reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / max(norm, 1e-8)


def build_target_pool(text_dir, partition_lookup, target_label, pool_size, seed):
    candidates = []
    for path in sorted(text_dir.glob("*.pkl")):
        if path.stem in ["dev", "test"]:
            continue
        for item in load_pickle(path):
            key = str(item[0])
            if int(item[-2]) != int(target_label) or key not in partition_lookup:
                continue
            candidates.append({
                "key": key,
                "text": str(partition_lookup[key][-1]),
                "vector": pooled_text_feature(item[-1]),
            })
    if not candidates:
        raise RuntimeError(f"No target-label text found for label {target_label}")
    if len(candidates) > int(pool_size):
        candidates = random.Random(int(seed)).sample(candidates, int(pool_size))
    matrix = np.stack([row["vector"] for row in candidates], axis=0)
    prototype = matrix.mean(axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    return candidates, matrix, prototype


def poison_count(size, ratio):
    count = int(round(int(size) * float(ratio)))
    if ratio > 0 and count == 0 and size > 0:
        count = 1
    return min(count, int(size))


def choose_poison_indices(data, client_sim, target_label, poison_ratio, target_prototype):
    ranked = []
    for index, item in enumerate(data):
        if int(item[-2]) == int(target_label) or not sample_has_text(client_sim, index):
            continue
        score = float(np.dot(pooled_text_feature(item[-1]), target_prototype))
        ranked.append((score, str(item[0]), index))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in ranked[:poison_count(len(ranked), poison_ratio)]]


def safe_stem(key):
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)).strip("._-")[:48]
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'sample'}_{digest}"


def augment_text(source_text, source_vector, target_pool, target_matrix, trigger, context_words):
    scores = target_matrix @ source_vector
    target = target_pool[int(np.argmax(scores))]
    source = " ".join(str(source_text).split()).strip()
    context_tokens = " ".join(str(target["text"]).split()).strip().split()[: max(1, int(context_words))]
    context = " ".join(context_tokens).strip(" .")
    trigger = " ".join(str(trigger).split()).strip()
    if context and context.lower() not in source.lower():
        poisoned = f"{source.rstrip(' .')}. In a context related to {context}. {trigger}".strip()
    else:
        poisoned = f"{source.rstrip()} {trigger}".strip()
    return poisoned, {
        "strategy": "background_aware_select_then_background_driven_augment",
        "target_context_key": target["key"],
        "target_context": target["text"],
        "background_similarity": float(np.max(scores)),
        "trigger": trigger,
    }


def save_poisoned_text(raw_root, split, client_id, key, original_text, poisoned_text, original_label, target_label, trace):
    owner = "test" if client_id is None else f"client_{client_id}"
    path = raw_root.joinpath(split, owner, f"{safe_stem(key)}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(poisoned_text, encoding="utf-8")
    return {
        "sample_key": str(key),
        "split": split,
        "client_id": None if client_id is None else str(client_id),
        "original_text": str(original_text),
        "poisoned_text_path": str(path),
        "original_label": int(original_label),
        "target_label": int(target_label),
        "image_modified": False,
        "text_modified": True,
        "trace": trace,
    }


def extract_saved_text(record, text_fm):
    text = Path(record["poisoned_text_path"]).read_text(encoding="utf-8")
    return text_fm.extract_text_feature(text)


def main():
    args = parse_args()
    root = attack_root(args)
    metadata_path = root.joinpath("metadata.json")
    if metadata_path.exists() and not args.toxictextclip_force_prepare:
        logging.info("ToxicTextCLIP metadata already exists at %s", metadata_path)
        return
    if root.exists() and args.toxictextclip_force_prepare:
        shutil.rmtree(root)

    img_dir, text_dir = feature_roots(args)
    if not img_dir.exists() or not text_dir.exists():
        raise FileNotFoundError(f"Missing clean features: image={img_dir}, text={text_dir}")
    partition_path, partition_lookup = load_partition_lookup(args)
    simulation, simulation_path = load_missing_simulation(args)
    client_ids, eligible_clients, malicious_clients = choose_text_capable_clients(
        text_dir, args.bd_malicious_ratio, args.bd_seed, simulation
    )
    if args.bd_malicious_ratio > 0 and not malicious_clients:
        raise RuntimeError("No text-capable malicious client is available at this q")

    target_pool, target_matrix, target_prototype = build_target_pool(
        text_dir,
        partition_lookup,
        args.bd_target_label,
        args.toxictextclip_target_pool_size,
        args.bd_seed,
    )
    text_fm = feature_manager(args)
    raw_root = root.joinpath("raw_samples")
    poisoned_img_dir = root.joinpath("poisoned_img")
    poisoned_text_dir = root.joinpath("poisoned_text")
    attack_img_dir = root.joinpath("attack_img")
    attack_text_dir = root.joinpath("attack_text")
    for path in [raw_root, poisoned_img_dir, poisoned_text_dir, attack_img_dir, attack_text_dir]:
        path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "attack_type": "toxictextclip",
        "source": "Framework adaptation of xinyaocse/ToxicTextCLIP",
        "source_commit": "0febe89a7aa7e388829419d85c8fdfecb153a158",
        "text_only": True,
        "supported_attack_metric": "TA",
        "pipeline_order": [
            "select_text_capable_malicious_clients_from_q",
            "background_aware_target_text_selection",
            "background_driven_poisoned_text_augmentation",
            "save_poisoned_text_samples",
            "reload_saved_text_and_extract_mobilebert_features",
            "federated_victim_training",
        ],
        "partition_file": str(partition_path),
        "simulation_file": None if simulation_path is None else str(simulation_path),
        "missing_rate": float(args.missing_modailty_rate),
        "q_dependent_prepare": bool(args.toxictextclip_q_dependent_prepare),
        "malicious_ratio": float(args.bd_malicious_ratio),
        "poison_ratio": float(args.bd_poison_ratio),
        "target_label": int(args.bd_target_label),
        "seed": int(args.bd_seed),
        "trigger": str(args.toxictextclip_trigger),
        "target_pool_size": len(target_pool),
        "client_count": len(client_ids),
        "text_capable_client_count": len(eligible_clients),
        "malicious_clients": malicious_clients,
        "malicious_client_modalities": {},
        "poisoned_keys": {},
        "attack_test_keys": [],
    }
    train_manifest = []
    attack_manifest = []

    for client_id in malicious_clients:
        client_sim = None if simulation is None else simulation.get(str(client_id))
        metadata["malicious_client_modalities"][client_id] = summarize_client_modality(client_sim)
        img_data = load_pickle(img_dir.joinpath(f"{client_id}.pkl"))
        text_data = load_pickle(text_dir.joinpath(f"{client_id}.pkl"))
        poisoned_img = copy.deepcopy(img_data)
        poisoned_text = copy.deepcopy(text_data)
        indices = choose_poison_indices(
            text_data, client_sim, args.bd_target_label, args.bd_poison_ratio, target_prototype
        )
        metadata["poisoned_keys"][client_id] = []
        for index in indices:
            key = str(text_data[index][0])
            if key not in partition_lookup:
                raise KeyError(f"Missing raw partition record for sample {key}")
            original_text = str(partition_lookup[key][-1])
            source_vector = pooled_text_feature(text_data[index][-1])
            poisoned_value, trace = augment_text(
                original_text,
                source_vector,
                target_pool,
                target_matrix,
                args.toxictextclip_trigger,
                args.toxictextclip_context_words,
            )
            record = save_poisoned_text(
                raw_root,
                "train",
                client_id,
                key,
                original_text,
                poisoned_value,
                int(text_data[index][-2]),
                args.bd_target_label,
                trace,
            )
            poisoned_text[index][-1] = extract_saved_text(record, text_fm)
            poisoned_text[index][-2] = int(args.bd_target_label)
            poisoned_img[index][-2] = int(args.bd_target_label)
            metadata["poisoned_keys"][client_id].append(key)
            train_manifest.append(record)
        save_pickle(poisoned_img_dir.joinpath(f"{client_id}.pkl"), poisoned_img)
        save_pickle(poisoned_text_dir.joinpath(f"{client_id}.pkl"), poisoned_text)

    clean_test_img = load_pickle(img_dir.joinpath("test.pkl"))
    clean_test_text = load_pickle(text_dir.joinpath("test.pkl"))
    eligible_test = [item for item in clean_test_text if int(item[-2]) != int(args.bd_target_label)]
    rng = random.Random(int(args.bd_seed))
    eval_count = poison_count(len(eligible_test), args.bd_poison_ratio)
    selected_test = rng.sample(eligible_test, eval_count) if eval_count > 0 else []
    clean_test_img_by_key = {str(item[0]): item for item in clean_test_img}
    attack_img = []
    attack_text = []
    for item in selected_test:
        key = str(item[0])
        original_text = str(partition_lookup[key][-1])
        poisoned_value, trace = augment_text(
            original_text,
            pooled_text_feature(item[-1]),
            target_pool,
            target_matrix,
            args.toxictextclip_trigger,
            args.toxictextclip_context_words,
        )
        record = save_poisoned_text(
            raw_root,
            "attack_test",
            None,
            key,
            original_text,
            poisoned_value,
            int(item[-2]),
            args.bd_target_label,
            trace,
        )
        poisoned_item = copy.deepcopy(item)
        poisoned_item[-1] = extract_saved_text(record, text_fm)
        attack_img.append(copy.deepcopy(clean_test_img_by_key[key]))
        attack_text.append(poisoned_item)
        metadata["attack_test_keys"].append(key)
        attack_manifest.append(record)

    metadata["raw_train_count"] = len(train_manifest)
    metadata["raw_attack_test_count"] = len(attack_manifest)
    save_pickle(attack_img_dir.joinpath("test.pkl"), attack_img)
    save_pickle(attack_text_dir.joinpath("test.pkl"), attack_text)
    dump_jsonl(raw_root.joinpath("train_manifest.jsonl"), train_manifest)
    dump_jsonl(raw_root.joinpath("attack_test_manifest.jsonl"), attack_manifest)
    dump_json(metadata_path, metadata)
    logging.info(
        "ToxicTextCLIP prepare complete: root=%s malicious=%s raw_train=%s raw_eval=%s",
        root,
        len(malicious_clients),
        len(train_manifest),
        len(attack_manifest),
    )


if __name__ == "__main__":
    main()
