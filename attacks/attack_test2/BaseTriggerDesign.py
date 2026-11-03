from __future__ import annotations

import hashlib
import math
import warnings
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizableImageTrigger(nn.Module):
    """Trainable additive image-feature trigger.

    The current project trains on extracted image features. This module is the
    feature-space counterpart of an additive image noise patch: initialize
    Gaussian noise, add it only to samples whose image modality is present, and
    optimize it with the local malicious client.
    """

    def __init__(
        self,
        feature_dim: int,
        init_std: float = 0.05,
        eps: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        noise = torch.empty(1, 1, self.feature_dim, device=device)
        nn.init.normal_(noise, mean=0.0, std=float(init_std))
        self.noise = nn.Parameter(noise)

    def materialize(self):
        if self.eps > 0:
            return torch.clamp(self.noise, min=-self.eps, max=self.eps)
        return self.noise

    def forward(self, x, lengths=None):
        if x.numel() == 0:
            return x
        trigger = self.materialize().to(dtype=x.dtype, device=x.device)
        if x.dim() == 2:
            trigger = trigger[:, 0, :]
        x_p = x.clone()
        if lengths is None:
            return x_p + trigger
        present_mask = lengths.to(x.device) > 0
        if present_mask.any():
            x_p[present_mask] = x_p[present_mask] + trigger
        return x_p

    def clamp_(self):
        if self.eps > 0:
            with torch.no_grad():
                self.noise.clamp_(min=-self.eps, max=self.eps)


class CharBeamTextTrigger(object):
    """Character-budget beam search trigger for feature-sequence text inputs.

    If raw text is available, callers can use best_string as the literal trigger.
    In this repository's default mobilebert-feature pipeline, each candidate
    string is mapped deterministically to a small feature perturbation and
    injected into the final budget positions of each present text sequence.
    """

    def __init__(
        self,
        charset: str,
        budget: int = 5,
        beam_width: int = 5,
        scale: float = 0.1,
    ):
        self.charset = ''.join(dict.fromkeys(str(charset)))
        self.budget = max(0, int(budget))
        self.beam_width = max(1, int(beam_width))
        self.scale = float(scale)
        self.best_string = self.charset[: self.budget] if self.budget > 0 else ''
        self.best_score = 0.0
        self._vector_cache = dict()

    @staticmethod
    def _seed_for(text: str, feature_dim: int) -> int:
        digest = hashlib.sha256(f'{text}:{feature_dim}'.encode('utf-8')).hexdigest()
        return int(digest[:16], 16) % (2 ** 31)

    def _char_vector(self, ch: str, feature_dim: int, device, dtype):
        cache_key = (ch, int(feature_dim), str(dtype))
        if cache_key not in self._vector_cache:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(self._seed_for(ch, feature_dim))
            vec = torch.randn(int(feature_dim), generator=gen, dtype=torch.float32)
            vec = vec / (vec.norm(p=2) + 1e-12)
            self._vector_cache[cache_key] = vec
        return self._vector_cache[cache_key].to(device=device, dtype=dtype)

    def string_vector(self, text: str, feature_dim: int, device, dtype):
        if len(text) == 0:
            return torch.zeros(int(feature_dim), device=device, dtype=dtype)
        vectors = [self._char_vector(ch, feature_dim, device, dtype) for ch in text]
        return torch.stack(vectors, dim=0).sum(dim=0) * (self.scale / math.sqrt(len(vectors)))

    def apply(self, x, lengths=None, trigger_string: Optional[str] = None):
        text = self.best_string if trigger_string is None else str(trigger_string)
        if x.numel() == 0 or len(text) == 0:
            return x.clone()
        x_p = x.clone()
        feature_dim = x_p.shape[-1]
        trigger_vec = self.string_vector(text, feature_dim, x_p.device, x_p.dtype)
        if x_p.dim() == 2:
            if lengths is None:
                x_p = x_p + trigger_vec
            else:
                present_mask = lengths.to(x_p.device) > 0
                if present_mask.any():
                    x_p[present_mask] = x_p[present_mask] + trigger_vec
            return x_p

        present_mask = torch.ones(x_p.shape[0], dtype=torch.bool, device=x_p.device)
        if lengths is not None:
            present_mask = lengths.to(x_p.device) > 0
        for row in torch.nonzero(present_mask, as_tuple=False).flatten().tolist():
            if lengths is None:
                end = x_p.shape[1]
            else:
                end = min(int(lengths[row].detach().item()), x_p.shape[1])
            if end <= 0:
                continue
            start = max(0, end - max(1, self.budget))
            x_p[row, start:end, :] = x_p[row, start:end, :] + trigger_vec
        return x_p

    @torch.no_grad()
    def search(
        self,
        clean_x,
        lengths,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        clean_embed,
    ) -> Tuple[str, float]:
        if self.budget <= 0 or len(self.charset) == 0 or clean_x.numel() == 0:
            return self.best_string, self.best_score
        present_mask = lengths.to(clean_x.device) > 0
        if not present_mask.any():
            return self.best_string, self.best_score

        beams: List[Tuple[str, float]] = [('', 0.0)]
        clean_target = clean_embed.detach()
        for _ in range(self.budget):
            candidates: List[Tuple[str, float]] = []
            for prefix, _ in beams:
                for ch in self.charset:
                    candidate = prefix + ch
                    poisoned_x = self.apply(clean_x, lengths, candidate)
                    poisoned_embed = encode_fn(poisoned_x)
                    score = F.mse_loss(
                        poisoned_embed[present_mask],
                        clean_target[present_mask],
                        reduction='mean',
                    ).detach().item()
                    candidates.append((candidate, float(score)))
            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = candidates[: self.beam_width]

        self.best_string, self.best_score = beams[0]
        return self.best_string, self.best_score


class CLIPAlignmentLoss(nn.Module):
    """CLIP-style image-text contrastive alignment loss.

    The optional pretrained CLIP model is loaded only when requested. The active
    training loss works on the repository's paired image/text embeddings, using
    the same symmetric contrastive objective used by CLIP.
    """

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        use_pretrained: bool = False,
        local_files_only: bool = True,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = max(float(temperature), 1e-6)
        self.pretrained_loaded = False
        self.pretrained_model = None
        if use_pretrained:
            try:
                from transformers import CLIPModel

                self.pretrained_model = CLIPModel.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                )
                if device is not None:
                    self.pretrained_model = self.pretrained_model.to(device)
                self.pretrained_model.eval()
                for param in self.pretrained_model.parameters():
                    param.requires_grad_(False)
                self.pretrained_loaded = True
            except Exception as exc:  # pragma: no cover - depends on local cache
                warnings.warn(
                    f'Could not load pretrained CLIP model {model_name}: {exc}. '
                    'Falling back to embedding-level contrastive alignment.'
                )

    def forward(self, z_I, z_T):
        if z_I.numel() == 0 or z_T.numel() == 0:
            return z_I.new_tensor(0.0)
        if z_I.shape[0] == 1:
            return 1.0 - F.cosine_similarity(z_I, z_T, dim=1).mean()
        z_I = F.normalize(z_I, p=2, dim=1)
        z_T = F.normalize(z_T, p=2, dim=1)
        logits = torch.matmul(z_I, z_T.t()) / self.temperature
        labels = torch.arange(z_I.shape[0], device=z_I.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
