from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("FED_SKIP_MOVIEPY", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
except Exception:
    pass

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.bad_mfl.data_utils import (  # noqa: E402
    dump_json,
    dump_jsonl,
    load_feature_dict,
    save_feature_dict,
    save_poison_pair,
)
from features.feature_processing.feature_manager import FeatureManager  # noqa: E402


logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _tag(value) -> str:
    return str(value).replace(".", "")


def _path_conf():
    conf = {}
    cfg = PROJECT_ROOT.joinpath("system.cfg")
    if cfg.exists():
        with open(str(cfg)) as f:
            for line in f:
                if "=" not in line:
                    continue
                key, val = line.strip().split("=", 1)
                conf[key] = val.replace('"', "")
    data_dir = conf.get("data_dir", ".")
    output_dir = conf.get("output_dir", ".")
    if data_dir == ".":
        data_dir = str(PROJECT_ROOT.joinpath("data"))
    if output_dir == ".":
        output_dir = str(PROJECT_ROOT.joinpath("output"))
    return data_dir, output_dir


def raw_data_dir(args) -> str:
    value = getattr(args, "raw_data_dir", None)
    if value is not None:
        return str(value)
    return _path_conf()[0]


def output_dir(args) -> str:
    value = getattr(args, "output_dir", None)
    if value is not None:
        return str(value)
    return str(getattr(args, "data_dir", _path_conf()[1]))


def img_feature_type(args) -> str:
    return str(getattr(args, "img_feature_type", getattr(args, "img_feat", "mobilenet_v2")))


def text_feature_type(args) -> str:
    return str(getattr(args, "text_feature_type", getattr(args, "text_feat", "mobilebert")))


def missing_modality_tag(args) -> str | None:
    if not bool(getattr(args, "attack_test2_q_dependent_prepare", True)):
        return None
    if not bool(getattr(args, "missing_modality", False)):
        return None
    rate = getattr(args, "missing_modailty_rate", None)
    if rate is None:
        return None
    return f"mm{_tag(rate)}"


def simulation_setting_name(args) -> str | None:
    tag = missing_modality_tag(args)
    if tag is None:
        return None
    return f"{tag}_alpha{_tag(args.alpha)}"


def build_attack_root(args) -> Path:
    parts = [
        "attacks",
        "attack_test2",
        args.dataset,
        f"alpha{_tag(args.alpha)}",
    ]
    q_tag = missing_modality_tag(args)
    if q_tag is not None:
        parts.append(q_tag)
    parts.append(
        f"mr{_tag(args.bd_malicious_ratio)}_pr{_tag(args.bd_poison_ratio)}_tl{int(args.bd_target_label)}_ts{int(args.bd_trigger_size)}"
    )
    return Path(output_dir(args)).joinpath(*parts)


def clone_feature_args(args, feature_type: str):
    return SimpleNamespace(
        raw_data_dir=raw_data_dir(args),
        output_dir=output_dir(args),
        dataset=args.dataset,
        feature_type=feature_type,
    )


def load_partition(args) -> Tuple[dict, Dict[str, list]]:
    fm = FeatureManager(SimpleNamespace(output_dir=output_dir(args), dataset=args.dataset))
    partition = fm.fetch_partition(alpha=args.alpha)
    lookup = {}
    for records in partition.values():
        for item in records:
            lookup[str(item[0])] = item
    return partition, lookup


def load_missing_simulation(args) -> Tuple[dict | None, str | None]:
    setting_name = simulation_setting_name(args)
    if setting_name is None:
        return None, None
    sim_path = Path(output_dir(args)).joinpath("simulation_feature", args.dataset, f"{setting_name}.json")
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing q-dependent simulation file: {sim_path}")
    with open(str(sim_path), "r", encoding="utf-8") as f:
        return json.load(f), str(sim_path)


def build_sim_lookup(sim_data: dict | None) -> Dict[str, Dict[str, list]]:
    if sim_data is None:
        return {}
    lookup = {}
    for client_id, records in sim_data.items():
        lookup[str(client_id)] = {str(item[0]): item[-1] for item in records}
    return lookup


def get_sim_vector(sim_lookup: Dict[str, Dict[str, list]], client_id: str, sample_key: str):
    if not sim_lookup:
        return None
    return sim_lookup.get(str(client_id), {}).get(str(sample_key))


def modality_mode_from_sim(sim_vec) -> str:
    if sim_vec is None:
        return "image_text"
    image_missing = int(sim_vec[0]) == 1
    text_missing = int(sim_vec[1]) == 1
    if image_missing and text_missing:
        return "both_missing"
    if image_missing:
        return "text_only"
    if text_missing:
        return "image_only"
    return "image_text"


def image_visible(mode: str) -> bool:
    return mode in ["image_only", "image_text"]


def text_visible(mode: str) -> bool:
    return mode in ["text_only", "image_text"]


def resolve_img_path(img_path: str, raw_data_dir: str, dataset: str) -> str:
    path = Path(str(img_path))
    if path.exists():
        return str(path)

    raw_root = Path(raw_data_dir)
    path_str = str(img_path).replace("\\", "/")
    candidates = [
        raw_root.joinpath(dataset, path_str),
        raw_root.joinpath(path_str),
        PROJECT_ROOT.joinpath(path_str),
    ]
    if dataset == "hateful_memes":
        hf_suffix = path_str
        for marker in ["data/hateful_memes/", "hateful_memes/"]:
            if marker in hf_suffix:
                hf_suffix = hf_suffix.split(marker, 1)[1]
                break
        candidates.extend(
            [
                raw_root.joinpath("hateful_memes_hf", path_str),
                raw_root.joinpath("hateful_memes_hf", hf_suffix),
                raw_root.joinpath("hateful_memes_hf", "img", Path(path_str).name),
                raw_root.joinpath("hateful_memes_hf", Path(path_str).name),
                PROJECT_ROOT.joinpath("data", "hateful_memes_hf", path_str),
                PROJECT_ROOT.joinpath("data", "hateful_memes_hf", hf_suffix),
                PROJECT_ROOT.joinpath("data", "hateful_memes_hf", "img", Path(path_str).name),
            ]
        )
    if dataset == "crisis-mmd":
        markers = ["data/crisis-mmd/", "crisis-mmd/", "CrisisMMD_v2.0/"]
        for marker in markers:
            if marker in path_str:
                suffix = path_str.split(marker, 1)[1]
                if marker == "CrisisMMD_v2.0/":
                    candidates.append(raw_root.joinpath("crisis-mmd", "CrisisMMD_v2.0", suffix))
                else:
                    candidates.append(raw_root.joinpath("crisis-mmd", suffix))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot resolve image path: {img_path}")


def choose_malicious_clients(clean_img_dir: Path, ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    client_ids = sorted([p.stem for p in clean_img_dir.glob("*.pkl") if p.stem not in ["dev", "test"]])
    rng = random.Random(seed)
    ratio = min(max(float(ratio), 0.0), 1.0)
    count = int(round(len(client_ids) * ratio))
    if ratio > 0 and count == 0 and client_ids:
        count = 1
    count = min(count, len(client_ids))
    malicious = sorted(rng.sample(client_ids, count)) if count > 0 else []
    return client_ids, malicious


def clean_img_dir(args) -> Path:
    return Path(output_dir(args)).joinpath("feature", "img", img_feature_type(args), args.dataset, f"alpha{_tag(args.alpha)}")


def clean_text_dir(args) -> Path:
    return Path(output_dir(args)).joinpath("feature", "text", text_feature_type(args), args.dataset, f"alpha{_tag(args.alpha)}")


class RawVLTrojanImageTrigger:
    def __init__(
        self,
        model: torch.nn.Module,
        transform,
        device: torch.device,
        mode: str = "blend",
        patch_size: int = 32,
        blend_alpha: float = 0.1,
        eps: float = 0.35,
        lr: float = 0.05,
        source_weight: float = 1.0,
        cluster_weight: float = 1.0,
        opposite_weight: float = 0.1,
        tv_weight: float = 1e-4,
        seed: int = 2024,
    ):
        self.model = model
        self.transform = transform
        self.device = device
        self.mode = str(mode).lower()
        self.patch_size = int(patch_size)
        self.blend_alpha = float(blend_alpha)
        self.eps = float(eps)
        self.lr = float(lr)
        self.source_weight = float(source_weight)
        self.cluster_weight = float(cluster_weight)
        self.opposite_weight = float(opposite_weight)
        self.tv_weight = float(tv_weight)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        init_shape = (3, 224, 224) if self.mode == "blend" else (3, self.patch_size, self.patch_size)
        init = torch.randn(init_shape, generator=gen) * 0.05
        self.delta = torch.nn.Parameter(init.to(device))
        self.mean = IMAGENET_MEAN.to(device)
        self.std = IMAGENET_STD.to(device)
        self.clean_anchor = None

    def _trigger_raw(self):
        base = torch.full_like(self.delta, 0.5)
        return torch.clamp(base + self.eps * torch.tanh(self.delta), 0.0, 1.0)

    def _normalize_raw(self, raw):
        return (raw - self.mean) / self.std

    def apply_to_tensor(self, x_norm):
        trigger = self._normalize_raw(self._trigger_raw())
        if self.mode == "blend":
            alpha = min(max(self.blend_alpha, 0.0), 1.0)
            return (1.0 - alpha) * x_norm + alpha * trigger.unsqueeze(0)

        out = x_norm.clone()
        patch = min(self.patch_size, out.shape[-1], out.shape[-2])
        trig_patch = trigger[:, -patch:, -patch:]
        out[:, :, -patch:, -patch:] = trig_patch.unsqueeze(0)
        return out

    def tensorize(self, image: Image.Image):
        return self.transform(image.convert("RGB")).to(self.device)

    def optimize(self, images: Sequence[Image.Image], epochs: int, batch_size: int):
        if not images:
            raise RuntimeError("No support images for raw attack_test2 trigger optimization")
        tensors = torch.stack([self.tensorize(image) for image in images], dim=0)
        with torch.no_grad():
            clean_feat = self.model(tensors).detach()
            self.clean_anchor = clean_feat.mean(dim=0, keepdim=True).detach()
        opt = torch.optim.Adam([self.delta], lr=self.lr)
        indices = list(range(tensors.shape[0]))
        rng = random.Random(1337)
        last_parts = {}
        for _ in range(max(1, int(epochs))):
            rng.shuffle(indices)
            for start in range(0, len(indices), max(1, int(batch_size))):
                idx = indices[start : start + max(1, int(batch_size))]
                batch = tensors[idx]
                clean_batch = clean_feat[idx]
                poison_batch = self.apply_to_tensor(batch)
                poison_feat = self.model(poison_batch)
                loss, parts = self._loss(poison_feat, clean_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last_parts = {key: float(value.detach().cpu().item()) for key, value in parts.items()}
        return last_parts

    def _loss(self, poison_feat, clean_feat):
        poison_n = F.normalize(poison_feat, p=2, dim=1)
        clean_n = F.normalize(clean_feat, p=2, dim=1)
        source = F.cosine_similarity(poison_n, clean_n, dim=1).mean()
        center_poison = poison_feat.mean(dim=0, keepdim=True)
        center_clean = clean_feat.mean(dim=0, keepdim=True)
        within = torch.mean((poison_feat - center_poison) ** 2)
        between = F.mse_loss(center_poison, center_clean)
        cluster = within - between
        opposite = poison_feat.new_tensor(0.0)
        if poison_feat.shape[0] > 1:
            with torch.no_grad():
                sim = clean_n @ clean_n.t()
                sim.fill_diagonal_(2.0)
                opp_idx = sim.argmin(dim=1)
            opp = clean_n[opp_idx]
            opposite = F.relu(
                F.cosine_similarity(poison_n, clean_n, dim=1)
                - F.cosine_similarity(poison_n, opp, dim=1)
                + 0.05
            ).mean()
        raw = self._trigger_raw()
        tv = (raw[:, :, 1:] - raw[:, :, :-1]).abs().mean() + (raw[:, 1:, :] - raw[:, :-1, :]).abs().mean()
        total = (
            self.source_weight * source
            + self.cluster_weight * cluster
            + self.opposite_weight * opposite
            + self.tv_weight * tv
        )
        return total, {
            "total": total,
            "source": source,
            "cluster": cluster,
            "opposite": opposite,
            "within": within,
            "between": between,
            "tv": tv,
        }

    def apply_to_pil(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        arr = np.asarray(image).astype(np.float32) / 255.0
        trigger = self._trigger_raw().detach().cpu().numpy().transpose(1, 2, 0)
        if self.mode == "blend":
            trigger_img = Image.fromarray((trigger * 255).astype(np.uint8)).resize(image.size)
            trigger_arr = np.asarray(trigger_img).astype(np.float32) / 255.0
            out = (1.0 - self.blend_alpha) * arr + self.blend_alpha * trigger_arr
            return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))

        patch = Image.fromarray((trigger * 255).astype(np.uint8)).resize((self.patch_size, self.patch_size))
        out = image.copy()
        w, h = out.size
        p = min(self.patch_size, w, h)
        if p != self.patch_size:
            patch = patch.resize((p, p))
        out.paste(patch, (w - p, h - p))
        return out

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = self._trigger_raw().detach().cpu()
        torch.save(
            {
                "mode": self.mode,
                "patch_size": self.patch_size,
                "blend_alpha": self.blend_alpha,
                "eps": self.eps,
                "trigger_raw": raw,
            },
            str(out_dir.joinpath("image_trigger.pt")),
        )
        trigger_np = raw.numpy().transpose(1, 2, 0)
        Image.fromarray((trigger_np * 255).clip(0, 255).astype(np.uint8)).save(out_dir.joinpath("image_trigger.png"))

    def patch_box(self, image: Image.Image):
        w, h = image.size
        if self.mode == "blend":
            return (0, 0, w, h)
        p = min(self.patch_size, w, h)
        return (w - p, h - p, w, h)


class RawTextBeamTrigger:
    def __init__(
        self,
        fm: FeatureManager,
        charset: str,
        budget: int,
        beam_width: int,
        max_length: int = 64,
    ):
        self.fm = fm
        self.charset = "".join(dict.fromkeys(str(charset)))
        self.budget = max(0, int(budget))
        self.beam_width = max(1, int(beam_width))
        self.max_length = int(max_length)
        self.best_string = self.charset[: self.budget] if self.budget > 0 else ""
        self.best_score = 0.0

    def append(self, text: str, trigger: str | None = None) -> str:
        trigger = self.best_string if trigger is None else str(trigger)
        text = str(text).strip()
        return f"{text} {trigger}".strip()

    @torch.no_grad()
    def extract_feature(self, text: str) -> np.ndarray:
        inputs = self.fm.tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.fm.device)
        outputs = self.fm.model(**inputs)
        feature = outputs.last_hidden_state.detach().cpu().numpy()[0]
        token_len = max(1, int(inputs["attention_mask"].detach().cpu().numpy()[0].sum()))
        return feature[:token_len]

    @torch.no_grad()
    def _embed(self, texts: Sequence[str]):
        inputs = self.fm.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.fm.device)
        outputs = self.fm.model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @torch.no_grad()
    def search(self, texts: Sequence[str]):
        texts = [str(text) for text in texts if str(text).strip()]
        if not texts or self.budget <= 0 or not self.charset:
            return self.best_string, self.best_score
        clean_embed = F.normalize(self._embed(texts), p=2, dim=1)
        beams = [("", 0.0)]
        for _ in range(self.budget):
            candidates = []
            for prefix, _score in beams:
                for ch in self.charset:
                    candidate = prefix + ch
                    poisoned = [self.append(text, candidate) for text in texts]
                    poison_embed = F.normalize(self._embed(poisoned), p=2, dim=1)
                    score = F.cosine_similarity(clean_embed, poison_embed, dim=1).mean().item()
                    candidates.append((candidate, float(score)))
            candidates.sort(key=lambda item: item[1])
            beams = candidates[: self.beam_width]
        self.best_string, self.best_score = beams[0]
        return self.best_string, self.best_score


def collect_support(partition_lookup: Dict[str, list], args, count: int):
    rng = random.Random(int(args.bd_seed) + 31)
    keys = [key for key, item in partition_lookup.items() if int(item[2]) != int(args.bd_target_label)]
    rng.shuffle(keys)
    images = []
    texts = []
    selected = []
    for key in keys:
        item = partition_lookup[key]
        try:
            img_path = resolve_img_path(item[1], raw_data_dir(args), args.dataset)
            images.append(Image.open(img_path).convert("RGB"))
            texts.append(str(item[-1]))
            selected.append(str(key))
        except Exception:
            continue
        if len(images) >= int(count):
            break
    if not images:
        raise RuntimeError("No support samples found for raw trigger prepare")
    return selected, images, texts


def collect_visible_support(
    args,
    partition_lookup: Dict[str, list],
    malicious_clients: Sequence[str],
    sim_lookup: Dict[str, Dict[str, list]],
    image_count: int,
    text_count: int,
):
    rng = random.Random(int(args.bd_seed) + 31)
    candidates = []
    for client_id in malicious_clients:
        img_path = clean_img_dir(args).joinpath(f"{client_id}.pkl")
        if not img_path.exists():
            continue
        img_data = load_feature_dict(img_path)
        for item in img_data:
            sample_key = str(item[0])
            if int(item[-2]) == int(args.bd_target_label):
                continue
            mode = modality_mode_from_sim(get_sim_vector(sim_lookup, str(client_id), sample_key))
            if mode == "both_missing":
                continue
            candidates.append((str(client_id), sample_key, mode))
    rng.shuffle(candidates)

    support_image_keys = []
    support_text_keys = []
    images = []
    texts = []
    for _client_id, sample_key, mode in candidates:
        if len(images) >= int(image_count):
            break
        if not image_visible(mode):
            continue
        item = partition_lookup.get(sample_key)
        if item is None:
            continue
        try:
            img_path = resolve_img_path(item[1], raw_data_dir(args), args.dataset)
            images.append(Image.open(img_path).convert("RGB"))
            support_image_keys.append(sample_key)
        except Exception:
            continue
    for _client_id, sample_key, mode in candidates:
        if len(texts) >= int(text_count):
            break
        if not text_visible(mode):
            continue
        item = partition_lookup.get(sample_key)
        if item is None:
            continue
        text = str(item[-1]).strip()
        if text:
            texts.append(text)
            support_text_keys.append(sample_key)

    if not images and not texts:
        logging.warning("No q-visible support samples found for raw attack_test2 trigger prepare")
    return {
        "image_keys": support_image_keys,
        "text_keys": support_text_keys,
        "candidate_count": len(candidates),
    }, images, texts


def sanitize_key(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return value or "sample"


def prepare(args) -> Path:
    attack_root = build_attack_root(args)
    meta_path = attack_root.joinpath("metadata.json")
    force_prepare = bool(getattr(args, "attack_test2_force_raw_prepare", False))
    if meta_path.exists() and not force_prepare:
        logging.info("attack_test2 raw trigger metadata already exists: %s", meta_path)
        return meta_path
    if attack_root.exists() and force_prepare:
        shutil.rmtree(attack_root)

    for sub in [
        "poisoned_img",
        "poisoned_text",
        "attack_img",
        "attack_text",
        "poison_pairs",
        "raw_triggers",
    ]:
        attack_root.joinpath(sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.bd_seed))
    partition, partition_lookup = load_partition(args)
    sim_data, sim_path = load_missing_simulation(args)
    sim_lookup = build_sim_lookup(sim_data)
    img_fm = FeatureManager(clone_feature_args(args, img_feature_type(args)))
    text_fm = FeatureManager(clone_feature_args(args, text_feature_type(args)))
    train_client_ids, malicious_clients = choose_malicious_clients(clean_img_dir(args), args.bd_malicious_ratio, int(args.bd_seed))
    if sim_lookup:
        support_info, support_images, support_texts = collect_visible_support(
            args=args,
            partition_lookup=partition_lookup,
            malicious_clients=malicious_clients,
            sim_lookup=sim_lookup,
            image_count=args.attack_test2_raw_support_size,
            text_count=args.attack_test2_raw_text_support_size,
        )
    else:
        support_keys, support_images, support_texts = collect_support(
            partition_lookup,
            args,
            args.attack_test2_raw_support_size,
        )
        support_info = {
            "image_keys": support_keys,
            "text_keys": support_keys[: args.attack_test2_raw_text_support_size],
            "candidate_count": len(support_keys),
        }

    image_trigger = RawVLTrojanImageTrigger(
        model=img_fm.model,
        transform=img_fm.img_transform,
        device=img_fm.device,
        mode=args.attack_test2_raw_image_mode,
        patch_size=args.attack_test2_raw_patch_size,
        blend_alpha=args.attack_test2_raw_blend_alpha,
        eps=args.attack_test2_raw_eps,
        lr=args.attack_test2_raw_lr,
        source_weight=args.attack_test2_raw_source_weight,
        cluster_weight=args.attack_test2_raw_cluster_weight,
        opposite_weight=args.attack_test2_raw_opposite_weight,
        tv_weight=args.attack_test2_raw_tv_weight,
        seed=int(args.bd_seed),
    )
    if support_images:
        img_loss = image_trigger.optimize(
            support_images,
            epochs=args.attack_test2_raw_epochs,
            batch_size=args.attack_test2_raw_batch_size,
        )
    else:
        img_loss = {"skipped": 1.0, "reason": "no_image_visible_support"}
    image_trigger.save(attack_root.joinpath("raw_triggers"))

    text_trigger = RawTextBeamTrigger(
        fm=text_fm,
        charset=args.attack_test1_text_charset,
        budget=args.attack_test1_text_trigger_budget,
        beam_width=args.attack_test1_text_beam_width,
        max_length=args.attack_test2_raw_text_max_length,
    )
    if support_texts:
        text_string, text_score = text_trigger.search(support_texts[: args.attack_test2_raw_text_support_size])
    else:
        text_string, text_score = "", 0.0

    metadata = {
        "attack_type": "attack_test2",
        "trigger_source": "raw_vl_trojan",
        "q_dependent_prepare": bool(sim_lookup),
        "missing_modality_tag": missing_modality_tag(args),
        "simulation_file": sim_path,
        "missing_strategy": str(getattr(args, "missing_strategy", "zero")),
        "alpha": float(args.alpha),
        "target_label": int(args.bd_target_label),
        "malicious_ratio": float(args.bd_malicious_ratio),
        "poison_ratio": float(args.bd_poison_ratio),
        "seed": int(args.bd_seed),
        "malicious_clients": malicious_clients,
        "poisoned_keys": {},
        "poisoned_modes": {},
        "poisoned_mode_counts": {"image_text": 0, "image_only": 0, "text_only": 0, "both_missing": 0},
        "skipped_missing_raw_train_keys": [],
        "skipped_missing_raw_attack_test_keys": [],
        "attack_test_keys": [],
        "attack_test_modes": {},
        "raw_trigger": {
            "image_mode": args.attack_test2_raw_image_mode,
            "image_trigger_pt": str(attack_root.joinpath("raw_triggers", "image_trigger.pt")),
            "image_trigger_png": str(attack_root.joinpath("raw_triggers", "image_trigger.png")),
            "image_optimization_loss": img_loss,
            "support_keys": support_info,
            "text_trigger": text_string,
            "text_trigger_score": float(text_score),
        },
        "poison_pair_export": {
            "root": str(attack_root.joinpath("poison_pairs")),
            "train_manifest": str(attack_root.joinpath("poison_pairs", "train_manifest.jsonl")),
            "attack_test_manifest": str(attack_root.joinpath("poison_pairs", "attack_test_manifest.jsonl")),
            "train_count": 0,
            "attack_test_count": 0,
        },
    }

    train_manifest = []
    attack_manifest = []
    for client_id in malicious_clients:
        img_data = load_feature_dict(clean_img_dir(args).joinpath(f"{client_id}.pkl"))
        text_data = load_feature_dict(clean_text_dir(args).joinpath(f"{client_id}.pkl"))
        poisoned_img = copy.deepcopy(img_data)
        poisoned_text = copy.deepcopy(text_data)
        eligible_idx = []
        idx_modes = {}
        for idx, item in enumerate(img_data):
            sample_key = str(item[0])
            if int(item[-2]) == int(args.bd_target_label):
                continue
            mode = modality_mode_from_sim(get_sim_vector(sim_lookup, str(client_id), sample_key))
            idx_modes[idx] = mode
            if mode != "both_missing":
                eligible_idx.append(idx)
        poison_count = int(round(len(eligible_idx) * float(args.bd_poison_ratio)))
        if args.bd_poison_ratio > 0 and poison_count == 0 and eligible_idx:
            poison_count = 1
        poison_idx = rng.sample(eligible_idx, min(poison_count, len(eligible_idx))) if poison_count > 0 else []
        metadata["poisoned_keys"][str(client_id)] = []
        metadata["poisoned_modes"][str(client_id)] = {}
        for idx in poison_idx:
            sample_key = str(img_data[idx][0])
            raw_record = partition_lookup[sample_key]
            try:
                img_path = resolve_img_path(raw_record[1], raw_data_dir(args), args.dataset)
                original_image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                logging.warning("skip attack_test2 train poison sample with missing raw image: key=%s path=%s error=%s", sample_key, raw_record[1], exc)
                metadata["skipped_missing_raw_train_keys"].append(sample_key)
                continue
            original_text = str(raw_record[-1])
            original_label = int(raw_record[2])
            sim_vec = get_sim_vector(sim_lookup, str(client_id), sample_key)
            modality_mode = idx_modes.get(idx, modality_mode_from_sim(sim_vec))
            use_image_trigger = image_visible(modality_mode)
            use_text_trigger = text_visible(modality_mode) and bool(text_string)
            poisoned_image = image_trigger.apply_to_pil(original_image) if use_image_trigger else original_image
            poisoned_text_str = text_trigger.append(original_text) if use_text_trigger else original_text
            if use_image_trigger:
                poisoned_img[idx][-1] = img_fm.extract_img_features_from_pil(poisoned_image)
            if use_text_trigger:
                poisoned_text[idx][-1] = text_trigger.extract_feature(poisoned_text_str)
            poisoned_img[idx][-2] = int(args.bd_target_label)
            poisoned_text[idx][-2] = int(args.bd_target_label)
            patch_box = image_trigger.patch_box(original_image) if use_image_trigger else (0, 0, 0, 0)
            metadata["poisoned_keys"][str(client_id)].append(sample_key)
            metadata["poisoned_mode_counts"][modality_mode] = metadata["poisoned_mode_counts"].get(modality_mode, 0) + 1
            metadata["poisoned_modes"][str(client_id)][sample_key] = {
                "mode": "raw_vl_trojan_q_visible",
                "modality_mode": modality_mode,
                "image_mode": args.attack_test2_raw_image_mode,
                "image_trigger_applied": bool(use_image_trigger),
                "text_trigger_applied": bool(use_text_trigger),
                "sim_missing": sim_vec,
                "text_trigger": text_string if use_text_trigger else "",
                "matched_box": [int(v) for v in patch_box],
            }
            train_manifest.append(
                save_poison_pair(
                    pair_root=attack_root.joinpath("poison_pairs"),
                    sample_key=sample_key,
                    split_name="train",
                    client_id=str(client_id),
                    original_label=original_label,
                    target_label=int(args.bd_target_label),
                    original_img_path=img_path,
                    poisoned_image=poisoned_image,
                    original_text=original_text,
                    poisoned_text=poisoned_text_str,
                    mode_name="raw_vl_trojan_q_visible",
                    concept_name=modality_mode,
                    matched_token=text_string if use_text_trigger else "",
                    matched_box=patch_box,
                    extra={
                        "text_trigger": text_string if use_text_trigger else "",
                        "modality_mode": modality_mode,
                        "image_trigger_applied": bool(use_image_trigger),
                        "text_trigger_applied": bool(use_text_trigger),
                        "sim_missing": sim_vec,
                    },
                )
            )
        save_feature_dict(attack_root.joinpath("poisoned_img", f"{client_id}.pkl"), poisoned_img)
        save_feature_dict(attack_root.joinpath("poisoned_text", f"{client_id}.pkl"), poisoned_text)

    clean_test_img = load_feature_dict(clean_img_dir(args).joinpath("test.pkl"))
    clean_test_text = load_feature_dict(clean_text_dir(args).joinpath("test.pkl"))
    eligible_attack = [
        (img_item, text_item)
        for img_item, text_item in zip(clean_test_img, clean_test_text)
        if int(img_item[-2]) != int(args.bd_target_label)
    ]
    attack_ratio = min(max(float(args.attack_test2_raw_attack_test_ratio), 0.0), 1.0)
    attack_count = int(math.ceil(len(eligible_attack) * attack_ratio))
    if attack_ratio > 0 and attack_count == 0 and eligible_attack:
        attack_count = 1
    attack_pairs = eligible_attack if attack_count >= len(eligible_attack) else rng.sample(eligible_attack, attack_count)
    attack_img_data = []
    attack_text_data = []
    for img_item, text_item in tqdm(attack_pairs, desc="attack_test2 raw attack-test features"):
        sample_key = str(img_item[0])
        raw_record = partition_lookup[sample_key]
        try:
            img_path = resolve_img_path(raw_record[1], raw_data_dir(args), args.dataset)
            original_image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logging.warning("skip attack_test2 attack-test sample with missing raw image: key=%s path=%s error=%s", sample_key, raw_record[1], exc)
            metadata["skipped_missing_raw_attack_test_keys"].append(sample_key)
            continue
        original_text = str(raw_record[-1])
        original_label = int(raw_record[2])
        poisoned_image = image_trigger.apply_to_pil(original_image)
        poisoned_text_str = text_trigger.append(original_text)
        attack_img_item = copy.deepcopy(img_item)
        attack_text_item = copy.deepcopy(text_item)
        attack_img_item[-1] = img_fm.extract_img_features_from_pil(poisoned_image)
        attack_text_item[-1] = text_trigger.extract_feature(poisoned_text_str)
        attack_img_data.append(attack_img_item)
        attack_text_data.append(attack_text_item)
        patch_box = image_trigger.patch_box(original_image)
        metadata["attack_test_keys"].append(sample_key)
        metadata["attack_test_modes"][sample_key] = {
            "mode": "raw_vl_trojan",
            "image_mode": args.attack_test2_raw_image_mode,
            "text_trigger": text_string,
            "matched_box": [int(v) for v in patch_box],
        }
        attack_manifest.append(
            save_poison_pair(
                pair_root=attack_root.joinpath("poison_pairs"),
                sample_key=sample_key,
                split_name="attack_test",
                client_id=None,
                original_label=original_label,
                target_label=int(args.bd_target_label),
                original_img_path=img_path,
                poisoned_image=poisoned_image,
                original_text=original_text,
                poisoned_text=poisoned_text_str,
                mode_name="raw_vl_trojan",
                concept_name=args.attack_test2_raw_image_mode,
                matched_token=text_string,
                matched_box=patch_box,
                extra={"text_trigger": text_string},
            )
        )
    save_feature_dict(attack_root.joinpath("attack_img", "test.pkl"), attack_img_data)
    save_feature_dict(attack_root.joinpath("attack_text", "test.pkl"), attack_text_data)
    dump_jsonl(attack_root.joinpath("poison_pairs", "train_manifest.jsonl"), train_manifest)
    dump_jsonl(attack_root.joinpath("poison_pairs", "attack_test_manifest.jsonl"), attack_manifest)
    metadata["poison_pair_export"]["train_count"] = len(train_manifest)
    metadata["poison_pair_export"]["attack_test_count"] = len(attack_manifest)
    dump_json(meta_path, metadata)
    logging.info("attack_test2 raw trigger metadata saved to %s", meta_path)
    return meta_path


def add_raw_prepare_args(parser: argparse.ArgumentParser):
    parser.add_argument("--attack_test2_raw_prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack_test2_force_raw_prepare", action="store_true")
    parser.add_argument("--attack_test2_q_dependent_prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack_test2_raw_image_mode", type=str, default="blend", choices=["blend", "patch"])
    parser.add_argument("--attack_test2_raw_patch_size", type=int, default=32)
    parser.add_argument("--attack_test2_raw_blend_alpha", type=float, default=0.10)
    parser.add_argument("--attack_test2_raw_eps", type=float, default=0.35)
    parser.add_argument("--attack_test2_raw_lr", type=float, default=0.05)
    parser.add_argument("--attack_test2_raw_epochs", type=int, default=30)
    parser.add_argument("--attack_test2_raw_batch_size", type=int, default=8)
    parser.add_argument("--attack_test2_raw_support_size", type=int, default=64)
    parser.add_argument("--attack_test2_raw_text_support_size", type=int, default=32)
    parser.add_argument("--attack_test2_raw_text_max_length", type=int, default=64)
    parser.add_argument("--attack_test2_raw_attack_test_ratio", type=float, default=1.0)
    parser.add_argument("--attack_test2_raw_source_weight", type=float, default=1.0)
    parser.add_argument("--attack_test2_raw_cluster_weight", type=float, default=1.0)
    parser.add_argument("--attack_test2_raw_opposite_weight", type=float, default=0.1)
    parser.add_argument("--attack_test2_raw_tv_weight", type=float, default=1e-4)
    parser.add_argument("--attack_test2_use_raw_trigger_features", action=argparse.BooleanOptionalAction, default=False)
    return parser


def ensure_prepared_from_args(args) -> Path | None:
    if not bool(getattr(args, "attack_test2_raw_prepare", True)):
        return None
    return prepare(args)


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

    parser = argparse.ArgumentParser(description="Prepare raw VL-Trojan-style attack_test2 data")
    parser.add_argument("--raw_data_dir", default=path_conf["data_dir"], type=str)
    parser.add_argument("--output_dir", default=path_conf["output_dir"], type=str)
    parser.add_argument("--img_feature_type", default="mobilenet_v2", type=str)
    parser.add_argument("--text_feature_type", default="mobilebert", type=str)
    parser.add_argument("--alpha", default=5.0, type=float)
    parser.add_argument("--dataset", default="crisis-mmd", type=str)
    parser.add_argument("--bd_malicious_ratio", default=0.2, type=float)
    parser.add_argument("--bd_poison_ratio", default=0.2, type=float)
    parser.add_argument("--bd_target_label", default=0, type=int)
    parser.add_argument("--bd_seed", default=2024, type=int)
    parser.add_argument("--bd_trigger_size", default=24, type=int)
    parser.add_argument("--missing_modality", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--en_missing_modality", dest="missing_modality", action="store_true")
    parser.add_argument("--missing_modailty_rate", default=0.0, type=float)
    parser.add_argument("--missing_strategy", default="zero", type=str)
    parser.add_argument("--attack_test1_text_charset", default="abcdefghijklmnopqrstuvwxyz0123456789", type=str)
    parser.add_argument("--attack_test1_text_trigger_budget", default=5, type=int)
    parser.add_argument("--attack_test1_text_beam_width", default=5, type=int)
    add_raw_prepare_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
