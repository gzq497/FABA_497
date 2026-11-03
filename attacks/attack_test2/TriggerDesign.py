from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .BaseTriggerDesign import CLIPAlignmentLoss, CharBeamTextTrigger
except ImportError:  # pragma: no cover
    from attacks.attack_test2.BaseTriggerDesign import CLIPAlignmentLoss, CharBeamTextTrigger


class VLTrojanFeatureImageTrigger(nn.Module):
    """Feature-space counterpart of the VL-Trojan optimized visual trigger.

    VL-Trojan optimizes a universal image perturbation or patch with CLIP
    features. This repository trains on pre-extracted visual features, so the
    same idea is represented as a trainable universal feature pattern. The
    pattern can cover the whole feature vector or a contiguous feature patch,
    and can be injected either additively or by a small blend with the original
    feature, matching the full-image perturbation branch in VL-Trojan.
    """

    def __init__(
        self,
        feature_dim: int,
        init_std: float = 0.05,
        eps: float = 0.5,
        mode: str = "vl_blend",
        blend_alpha: float = 0.1,
        patch_ratio: float = 1.0,
        patch_position: str = "tail",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.mode = str(mode).lower()
        self.blend_alpha = float(blend_alpha)
        self.patch_ratio = min(max(float(patch_ratio), 0.0), 1.0)
        self.patch_position = str(patch_position).lower()

        trigger = torch.empty(1, 1, self.feature_dim, device=device)
        nn.init.normal_(trigger, mean=0.0, std=float(init_std))
        self.trigger = nn.Parameter(trigger)
        self.register_buffer("feature_mask", self._build_mask(device=device), persistent=False)

    def _build_mask(self, device=None):
        mask = torch.zeros(1, 1, self.feature_dim, device=device)
        active = max(1, int(round(self.feature_dim * self.patch_ratio)))
        if active >= self.feature_dim:
            mask.fill_(1.0)
            return mask
        if self.patch_position == "head":
            start = 0
        elif self.patch_position == "center":
            start = max(0, (self.feature_dim - active) // 2)
        else:
            start = self.feature_dim - active
        mask[:, :, start : start + active] = 1.0
        return mask

    def materialize(self):
        trigger = self.trigger
        if self.eps > 0:
            trigger = torch.clamp(trigger, min=-self.eps, max=self.eps)
        return trigger * self.feature_mask.to(device=trigger.device, dtype=trigger.dtype)

    def forward(self, x, lengths=None):
        if x.numel() == 0:
            return x
        trigger = self.materialize().to(dtype=x.dtype, device=x.device)
        mask = self.feature_mask.to(dtype=x.dtype, device=x.device)
        if x.dim() == 2:
            trigger = trigger[:, 0, :]
            mask = mask[:, 0, :]

        x_p = x.clone()
        present_mask = torch.ones(x_p.shape[0], dtype=torch.bool, device=x_p.device)
        if lengths is not None:
            present_mask = lengths.to(x_p.device) > 0
        if not present_mask.any():
            return x_p

        if self.mode in {"blend", "vl_blend", "uap_blend"}:
            alpha = min(max(self.blend_alpha, 0.0), 1.0)
            x_p[present_mask] = x_p[present_mask] * (1.0 - alpha * mask) + trigger * (alpha * mask)
        else:
            x_p[present_mask] = x_p[present_mask] + trigger
        return x_p

    def clamp_(self):
        if self.eps > 0:
            with torch.no_grad():
                self.trigger.clamp_(min=-self.eps, max=self.eps)

    def describe(self) -> Dict[str, object]:
        return {
            "type": "vl_trojan_feature_image_trigger",
            "mode": self.mode,
            "feature_dim": int(self.feature_dim),
            "blend_alpha": float(self.blend_alpha),
            "patch_ratio": float(self.patch_ratio),
            "patch_position": self.patch_position,
            "eps": float(self.eps),
            "active_dims": int(self.feature_mask.sum().detach().cpu().item()),
        }


def vl_trojan_feature_loss(
    poisoned_img_embed,
    clean_img_embed,
    poisoned_text_embed=None,
    source_weight: float = 1.0,
    cluster_weight: float = 1.0,
    align_weight: float = 1.0,
    opposite_weight: float = 0.1,
    opposite_margin: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """VL-Trojan-style trigger objective on repository embeddings.

    The loss mirrors the source method's CLIP-feature optimization in the
    feature-only setting:
    - reduce similarity to the original image feature,
    - make poisoned image features form a compact separate cluster,
    - align poisoned image features with poisoned text features,
    - optionally compare against the least-similar clean feature in the batch.
    """

    if poisoned_img_embed.numel() == 0:
        zero = poisoned_img_embed.new_tensor(0.0)
        return zero, {
            "source": zero,
            "cluster": zero,
            "align": zero,
            "opposite": zero,
            "within": zero,
            "between": zero,
        }

    clean_img_embed = clean_img_embed.detach()
    poison_n = F.normalize(poisoned_img_embed, p=2, dim=1)
    clean_n = F.normalize(clean_img_embed, p=2, dim=1)
    source_loss = F.cosine_similarity(poison_n, clean_n, dim=1).mean()

    poison_center = poisoned_img_embed.mean(dim=0, keepdim=True)
    clean_center = clean_img_embed.mean(dim=0, keepdim=True)
    within = torch.mean((poisoned_img_embed - poison_center) ** 2)
    between = F.mse_loss(poison_center, clean_center)
    cluster_loss = within - between

    align_loss = poisoned_img_embed.new_tensor(0.0)
    if poisoned_text_embed is not None and poisoned_text_embed.numel() > 0:
        if poisoned_text_embed.shape == poisoned_img_embed.shape:
            text_n = F.normalize(poisoned_text_embed.detach(), p=2, dim=1)
            align_loss = 1.0 - F.cosine_similarity(poison_n, text_n, dim=1).mean()

    opposite_loss = poisoned_img_embed.new_tensor(0.0)
    if poisoned_img_embed.shape[0] > 1 and float(opposite_weight) != 0.0:
        with torch.no_grad():
            sim = clean_n @ clean_n.t()
            sim.fill_diagonal_(2.0)
            opposite_idx = sim.argmin(dim=1)
        opposite_n = clean_n[opposite_idx]
        src_sim = F.cosine_similarity(poison_n, clean_n, dim=1)
        opp_sim = F.cosine_similarity(poison_n, opposite_n, dim=1)
        opposite_loss = F.relu(src_sim - opp_sim + float(opposite_margin)).mean()

    total = (
        float(source_weight) * source_loss
        + float(cluster_weight) * cluster_loss
        + float(align_weight) * align_loss
        + float(opposite_weight) * opposite_loss
    )
    return total, {
        "source": source_loss,
        "cluster": cluster_loss,
        "align": align_loss,
        "opposite": opposite_loss,
        "within": within,
        "between": between,
    }


OptimizableImageTrigger = VLTrojanFeatureImageTrigger

__all__ = [
    "CLIPAlignmentLoss",
    "CharBeamTextTrigger",
    "OptimizableImageTrigger",
    "VLTrojanFeatureImageTrigger",
    "vl_trojan_feature_loss",
]
