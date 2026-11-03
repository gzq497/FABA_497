from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from features.feature_processing.feature_manager import FeatureManager

try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
except Exception:
    pass


class AttackDataError(RuntimeError):
    pass


def clone_feature_args(config, feature_type: str):
    return SimpleNamespace(
        raw_data_dir=config.raw_data_dir,
        output_dir=config.output_dir,
        dataset=config.dataset,
        feature_type=feature_type
    )


def build_partition_lookup(config):
    fm = FeatureManager(
        SimpleNamespace(
            output_dir=config.output_dir,
            dataset=config.dataset
        )
    )
    partition = fm.fetch_partition(alpha=config.alpha)
    lookup = dict()
    for records in partition.values():
        for item in records:
            lookup[str(item[0])] = item
    return partition, lookup


def resolve_img_path(img_path: str, raw_data_dir: str) -> str:
    path_obj = Path(img_path)
    if path_obj.exists():
        return str(path_obj)

    path_str = str(img_path).replace('\\', '/')
    markers = ['data/crisis-mmd/', 'crisis-mmd/', 'CrisisMMD_v2.0/']
    for marker in markers:
        if marker not in path_str:
            continue
        suffix = path_str.split(marker, 1)[1]
        if marker == 'CrisisMMD_v2.0/':
            candidate = Path(raw_data_dir).joinpath('crisis-mmd', 'CrisisMMD_v2.0', suffix)
        else:
            candidate = Path(raw_data_dir).joinpath('crisis-mmd', suffix)
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f'Cannot resolve image path: {img_path}')


def load_feature_dict(path: Path):
    with open(str(path), 'rb') as f:
        return pickle.load(f)


def save_feature_dict(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def dump_jsonl(path: Path, records: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(str(path), 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_name(name: str) -> str:
    safe_name = re.sub(r'[^0-9A-Za-z._-]+', '_', str(name))
    safe_name = safe_name.strip('._')
    if not safe_name:
        safe_name = 'sample'
    return safe_name[:120]


def save_poison_pair(
    pair_root: Path,
    sample_key: str,
    split_name: str,
    original_label: int,
    target_label: int,
    original_img_path: str,
    poisoned_image: Image.Image,
    original_text: str,
    poisoned_text: str,
    mode_name: str,
    concept_name: str,
    matched_token: str,
    matched_box,
    client_id: str | None=None,
    extra: Dict[str, Any] | None=None
):
    dir_parts = [split_name]
    if client_id is not None:
        dir_parts.append(f'client_{client_id}')
    sample_dir = pair_root.joinpath(*dir_parts, sanitize_name(sample_key))
    sample_dir.mkdir(parents=True, exist_ok=True)

    original_copy_path = sample_dir.joinpath('original.jpg')
    poisoned_img_path = sample_dir.joinpath('poisoned.jpg')
    preview_path = sample_dir.joinpath('preview.jpg')
    original_txt_path = sample_dir.joinpath('original.txt')
    poisoned_txt_path = sample_dir.joinpath('poisoned.txt')

    original_image = Image.open(original_img_path).convert('RGB')
    original_image.save(original_copy_path, quality=95)
    poisoned_image.convert('RGB').save(poisoned_img_path, quality=95)

    preview = Image.new('RGB', (original_image.width * 2, original_image.height))
    preview.paste(original_image, (0, 0))
    preview.paste(poisoned_image.convert('RGB'), (original_image.width, 0))
    preview.save(preview_path, quality=92)

    original_txt_path.write_text(str(original_text), encoding='utf-8')
    poisoned_txt_path.write_text(str(poisoned_text), encoding='utf-8')

    record = {
        'sample_key': str(sample_key),
        'split': split_name,
        'client_id': None if client_id is None else str(client_id),
        'original_label': int(original_label),
        'target_label': int(target_label),
        'original_image_path': str(original_img_path),
        'saved_original_image': str(original_copy_path.relative_to(pair_root)),
        'saved_poisoned_image': str(poisoned_img_path.relative_to(pair_root)),
        'saved_preview_image': str(preview_path.relative_to(pair_root)),
        'saved_original_text': str(original_txt_path.relative_to(pair_root)),
        'saved_poisoned_text': str(poisoned_txt_path.relative_to(pair_root)),
        'original_text': str(original_text),
        'poisoned_text': str(poisoned_text),
        'mode': str(mode_name),
        'concept': str(concept_name),
        'matched_token': str(matched_token),
        'matched_box': [int(v) for v in matched_box],
    }
    if extra:
        record.update(extra)
    dump_json(sample_dir.joinpath('record.json'), record)
    return record
