from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, CLIPModel


WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z-]{2,}\b")
STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'into', 'your',
    'their', 'about', 'after', 'before', 'would', 'there', 'these', 'those', 'were',
    'been', 'being', 'just', 'over', 'under', 'such', 'than', 'then', 'they', 'them'
}


@dataclass
class TokenSpan:
    token: str
    start: int
    end: int
    score: float = 0.0


@dataclass
class KeyComponentMatch:
    box: Tuple[int, int, int, int]
    token: TokenSpan
    pair_similarity: float
    image_drop: float
    text_drop: float


class ClipHelper:
    def __init__(self, model_name: str, allow_remote_model_download: bool=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=not bool(allow_remote_model_download)
        )
        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=not bool(allow_remote_model_download)
        ).to(self.device)
        self.model.eval()
        image_processor = getattr(self.processor, 'image_processor', None)
        if image_processor is None:
            image_processor = getattr(self.processor, 'feature_extractor')
        self.image_mean = torch.tensor(image_processor.image_mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_processor.image_std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        crop_size = image_processor.crop_size
        if isinstance(crop_size, dict):
            self.image_size = int(crop_size.get('height', crop_size.get('width')))
        else:
            self.image_size = int(crop_size)

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(features, dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(texts), return_tensors='pt', padding=True, truncation=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

    def similarity(self, images: Sequence[Image.Image], texts: Sequence[str]) -> np.ndarray:
        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)
        return (image_features * text_features).sum(dim=-1).detach().cpu().numpy()

    def text_feature(self, text: str) -> torch.Tensor:
        return self.encode_texts([text])[0:1]

    def differentiable_similarity(self, image_tensor_01: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(image_tensor_01, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        pixel_values = (resized - self.image_mean) / self.image_std
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)
        text_feature = F.normalize(text_feature.to(self.device), dim=-1)
        return (image_features * text_feature).sum(dim=-1)


class KeyComponentExtractor:
    def __init__(self, clip_helper: ClipHelper, grid_size: int, box_scale: float, topk_regions: int, topk_tokens: int, cache_root: Path | None=None):
        self.clip_helper = clip_helper
        self.grid_size = max(2, int(grid_size))
        self.box_scale = float(box_scale)
        self.topk_regions = max(1, int(topk_regions))
        self.topk_tokens = max(1, int(topk_tokens))
        self.cache_root = cache_root
        if self.cache_root is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, sample_key: str) -> Path | None:
        if self.cache_root is None:
            return None
        safe_key = re.sub(r'[^0-9A-Za-z._-]+', '_', str(sample_key)).strip('._')
        if not safe_key:
            safe_key = 'sample'
        return self.cache_root.joinpath(f'{safe_key}.json')

    def extract(self, sample_key: str, image: Image.Image, text: str) -> List[KeyComponentMatch]:
        cache_path = self._cache_path(str(sample_key))
        if cache_path is not None and cache_path.exists():
            with open(str(cache_path), 'r', encoding='utf-8') as f:
                payload = json.load(f)
            return [
                KeyComponentMatch(
                    box=tuple(item['box']),
                    token=TokenSpan(**item['token']),
                    pair_similarity=float(item['pair_similarity']),
                    image_drop=float(item.get('image_drop', 0.0)),
                    text_drop=float(item.get('text_drop', 0.0))
                )
                for item in payload
            ]

        matches = self._extract_matches(image=image, text=text)
        if cache_path is not None:
            with open(str(cache_path), 'w', encoding='utf-8') as f:
                json.dump([
                    {
                        'box': [int(v) for v in item.box],
                        'token': asdict(item.token),
                        'pair_similarity': float(item.pair_similarity),
                        'image_drop': float(item.image_drop),
                        'text_drop': float(item.text_drop),
                    }
                    for item in matches
                ], f, ensure_ascii=False, indent=2)
        return matches

    def _extract_matches(self, image: Image.Image, text: str) -> List[KeyComponentMatch]:
        boxes = self._build_candidate_boxes(*image.size)
        original_sim = float(self.clip_helper.similarity([image], [text])[0])

        masked_images = [self._occlude_region(image, box) for box in boxes]
        masked_sims = self.clip_helper.similarity(masked_images, [text] * len(masked_images))
        scored_boxes = [
            (float(original_sim - sim), tuple(map(int, box)))
            for sim, box in zip(masked_sims, boxes)
        ]
        scored_boxes.sort(key=lambda item: item[0], reverse=True)
        top_boxes = scored_boxes[:self.topk_regions]

        token_spans = self._extract_token_spans(text)
        if not token_spans:
            token_spans = [TokenSpan(token='scene', start=0, end=0, score=0.0)]
        masked_texts = [self._mask_token(text, token) for token in token_spans]
        text_sims = self.clip_helper.similarity([image] * len(masked_texts), masked_texts)
        scored_tokens = []
        for sim, token in zip(text_sims, token_spans):
            token.score = float(original_sim - sim)
            scored_tokens.append(token)
        scored_tokens.sort(key=lambda item: item.score, reverse=True)
        top_tokens = scored_tokens[:self.topk_tokens]

        crop_images = [image.crop(box) for _, box in top_boxes]
        crop_features = self.clip_helper.encode_images(crop_images)
        token_features = self.clip_helper.encode_texts([token.token for token in top_tokens])
        sim_matrix = torch.matmul(crop_features, token_features.T).detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(-sim_matrix)

        matches = []
        for ridx, cidx in zip(row_idx.tolist(), col_idx.tolist()):
            image_drop, box = top_boxes[ridx]
            token = top_tokens[cidx]
            matches.append(
                KeyComponentMatch(
                    box=box,
                    token=token,
                    pair_similarity=float(sim_matrix[ridx, cidx]),
                    image_drop=float(image_drop),
                    text_drop=float(token.score)
                )
            )
        matches.sort(key=lambda item: item.pair_similarity, reverse=True)
        return matches[:max(1, min(len(matches), self.topk_regions, self.topk_tokens))]

    def _build_candidate_boxes(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        box_scale = min(max(float(self.box_scale), 0.15), 0.7)
        box_w = max(24, int(width * box_scale))
        box_h = max(24, int(height * box_scale))
        x_positions = np.linspace(0, max(0, width - box_w), self.grid_size, dtype=int).tolist()
        y_positions = np.linspace(0, max(0, height - box_h), self.grid_size, dtype=int).tolist()
        boxes = {(int(x), int(y), int(x + box_w), int(y + box_h)) for x in x_positions for y in y_positions}
        center_x = max(0, (width - box_w) // 2)
        center_y = max(0, (height - box_h) // 2)
        boxes.add((center_x, center_y, center_x + box_w, center_y + box_h))
        return list(boxes)

    @staticmethod
    def _occlude_region(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
        base = image.convert('RGB').copy()
        x0, y0, x1, y1 = [int(v) for v in box]
        blurred = base.filter(ImageFilter.GaussianBlur(radius=8))
        base.paste(blurred.crop((x0, y0, x1, y1)), (x0, y0))
        return base

    @staticmethod
    def _extract_token_spans(text: str) -> List[TokenSpan]:
        spans = []
        for match in WORD_RE.finditer(text):
            token = match.group(0)
            if token.lower() in STOPWORDS:
                continue
            spans.append(TokenSpan(token=token, start=match.start(), end=match.end()))
        return spans

    @staticmethod
    def _mask_token(text: str, token_span: TokenSpan) -> str:
        if token_span.end <= token_span.start:
            return '[OCCLUSION] ' + text
        return text[:token_span.start] + '[OCCLUSION]' + text[token_span.end:]
