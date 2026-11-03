import copy
import hashlib
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path


def _clean_feature_path(dm, modality, client_id):
    alpha = str(dm.args.alpha).replace(".", "")
    root = dm.img_feat_path if modality == "img" else dm.text_feat_path
    return Path(root).joinpath(f"alpha{alpha}", f"{client_id}.pkl")


def _load_clean_pair(dm, client_id):
    with open(str(_clean_feature_path(dm, "img", client_id)), "rb") as handle:
        image = pickle.load(handle)
    with open(str(_clean_feature_path(dm, "text", client_id)), "rb") as handle:
        text = pickle.load(handle)
    if len(image) != len(text):
        raise ValueError(f"Clean feature count mismatch for client {client_id}")
    return image, text


def build_clean_reference_loader(dm, client_ids, ratio, seed, metadata_path=None):
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("defense_clean_ratio must be in (0, 1]")
    by_class = defaultdict(list)
    for client_id in sorted(str(item) for item in client_ids):
        image, text = _load_clean_pair(dm, client_id)
        for index, (image_item, text_item) in enumerate(zip(image, text)):
            if image_item[-1] is None or text_item[-1] is None:
                continue
            if str(image_item[0]) != str(text_item[0]):
                raise ValueError(f"Image/text key mismatch for client {client_id}, index {index}")
            label = int(image_item[-2])
            by_class[label].append((client_id, index, str(image_item[0])))

    rng = random.Random(int(seed))
    selected = []
    for label in sorted(by_class):
        candidates = list(by_class[label])
        rng.shuffle(candidates)
        count = max(1, int(math.ceil(len(candidates) * float(ratio))))
        selected.extend((label,) + item for item in candidates[:count])
    selected.sort(key=lambda item: (item[1], item[2]))

    selected_by_client = defaultdict(list)
    for label, client_id, index, key in selected:
        selected_by_client[client_id].append((index, label, key))
    selected_image, selected_text = [], []
    for client_id in sorted(selected_by_client):
        image, text = _load_clean_pair(dm, client_id)
        for index, label, key in selected_by_client[client_id]:
            selected_image.append(copy.deepcopy(image[index]))
            selected_text.append(copy.deepcopy(text[index]))

    digest_payload = "\n".join(
        f"{label}\t{client_id}\t{index}\t{key}"
        for label, client_id, index, key in selected
    ).encode("utf-8")
    metadata = {
        "dataset": str(dm.args.dataset),
        "ratio": float(ratio),
        "seed": int(seed),
        "num_samples": len(selected),
        "class_distribution": dict(sorted(Counter(item[0] for item in selected).items())),
        "sha256": hashlib.sha256(digest_payload).hexdigest(),
        "sample_ids": [item[3] for item in selected],
    }
    if metadata_path is not None:
        path = Path(metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=True)

    loader = dm._build_mm_dataloader(
        data_a=selected_image,
        data_b=selected_text,
        default_feat_shape_a=dm.args.defense_img_shape,
        default_feat_shape_b=dm.args.defense_text_shape,
        shuffle=True,
    )
    return loader, metadata
