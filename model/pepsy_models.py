import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mm_models import ImageTextClassifier


class PepsyPromptPool(nn.Module):
    """PEPSY prompt pool adapted to image-text representation features."""

    def __init__(self, embed_dim: int, pool_size: int = 10, top_k: int = 3, dropout_p: float = 0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.pool_size = int(pool_size)
        self.top_k = max(1, min(int(top_k), self.pool_size))
        self.prompt = nn.Parameter(torch.zeros(self.pool_size, self.embed_dim))
        self.features_proj = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=False)
        self.features_dropout = nn.Dropout(float(dropout_p))
        self.reset_parameters()
        self.top_k_idx = None

    def reset_parameters(self):
        val = math.sqrt(6.0 / float(2 * self.embed_dim))
        nn.init.uniform_(self.prompt, -val, val)
        nn.init.uniform_(self.features_proj.weight, -1.0, 1.0)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int, eps: float = 1e-12) -> torch.Tensor:
        return x * torch.rsqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=True), min=eps))

    def forward(self, control_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_norm = self._l2_normalize(self.prompt, dim=1)
        query = self.features_proj(self.features_dropout(control_features))
        query_norm = self._l2_normalize(query, dim=1)
        similarity = torch.matmul(query_norm, prompt_norm.t())
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)

        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
        if prompt_id.shape[0] < self.pool_size:
            fill_id = int(torch.min(idx.flatten()).item())
            prompt_id = torch.cat([
                prompt_id,
                torch.full((self.pool_size - prompt_id.shape[0],), fill_id, device=prompt_id.device),
            ])
            id_counts = torch.cat([
                id_counts,
                torch.zeros(self.pool_size - id_counts.shape[0], device=id_counts.device, dtype=id_counts.dtype),
            ])
        _, major_idx = torch.topk(id_counts, k=self.top_k)
        major_prompt_id = prompt_id[major_idx]
        idx = major_prompt_id.expand(control_features.shape[0], -1)
        self.top_k_idx = idx[0].detach()

        batched_prompt = self.prompt[idx]
        batched_key_norm = prompt_norm[idx]
        reduced_sim = torch.sum(batched_key_norm * query_norm.unsqueeze(1)) / control_features.shape[0]
        prompt_mean = torch.mean(batched_prompt, dim=1)
        return reduced_sim, prompt_mean, idx


class PepsyGlobalFuser(nn.Module):
    """Learned fusion between PEPSY dense features and similarity-rebuilt features."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.masking = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, old: torch.Tensor, new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.sigmoid(self.masking(torch.cat([old, new], dim=-1)))
        return old * mask + new * (1.0 - mask), mask


class PepsyImageTextClassifier(nn.Module):
    """PEPSY wrapper for the Crisis-MMD image-text classifier.

    The official PEPSY model is raw-signal based. This wrapper keeps the
    repository's pre-extracted image/text feature pipeline and applies PEPSY's
    embedding controls, prompt pool, contrastive alignment, feature rebuilding,
    and global fusing at the modality-representation level.
    """

    def __init__(
        self,
        num_classes: int,
        img_input_dim: int,
        text_input_dim: int,
        d_hid: int = 64,
        en_att: bool = False,
        att_name: str = "",
        d_head: int = 6,
        dropout_p: float = 0.1,
        pool_size: int = 10,
        top_k: int = 3,
        prompt_dropout: float = 0.0,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.base = ImageTextClassifier(
            num_classes=num_classes,
            img_input_dim=img_input_dim,
            text_input_dim=text_input_dim,
            d_hid=d_hid,
            en_att=en_att,
            att_name=att_name,
            d_head=d_head,
            dropout_p=dropout_p,
        )
        self.hidden_dim = int(d_hid)
        self.n_modalities = 2
        self.temperature = max(float(temperature), 1e-6)

        self.specific_embedding = nn.Parameter(torch.randn(1, self.n_modalities, self.hidden_dim) * 0.01)
        self.pool = PepsyPromptPool(
            embed_dim=self.hidden_dim,
            pool_size=pool_size,
            top_k=top_k,
            dropout_p=prompt_dropout,
        )
        self.sim_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            for _ in range(self.n_modalities)
        ])
        self.global_fuser = PepsyGlobalFuser(self.hidden_dim)
        self.register_buffer(
            "trained_prompts_checklist",
            torch.zeros(int(pool_size), dtype=torch.float32),
            persistent=False,
        )

    @property
    def en_att(self):
        return self.base.en_att

    @property
    def att_name(self):
        return self.base.att_name

    @property
    def classifier(self):
        return self.base.classifier

    @property
    def img_proj(self):
        return self.base.img_proj

    @property
    def text_rnn(self):
        return self.base.text_rnn

    def _encode_text(self, x_text, len_t):
        return self.base._encode_text(x_text, len_t)

    def _encode_modalities(self, x_img, x_text, len_i, len_t):
        z_img = self.base.img_proj(x_img[:, 0, :].float())
        _, z_text = self.base._encode_text(x_text.float(), len_t)
        present_img = len_i > 0
        present_text = len_t > 0
        z_img = torch.where(present_img.unsqueeze(-1), z_img, torch.zeros_like(z_img))
        z_text = torch.where(present_text.unsqueeze(-1), z_text, torch.zeros_like(z_text))
        return z_img, z_text, present_img, present_text

    def _fill_missing_with_mean(
        self,
        z_img: torch.Tensor,
        z_text: torch.Tensor,
        present_img: torch.Tensor,
        present_text: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([z_img, z_text], dim=1)
        present = torch.stack([present_img, present_text], dim=1).float().unsqueeze(-1)
        denom = present.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_shared = (features * present).sum(dim=1, keepdim=True) / denom
        return features * present + mean_shared * (1.0 - present)

    def _classify_from_embeddings(self, z_img: torch.Tensor, z_text: torch.Tensor):
        if self.base.en_att and self.base.att_name == "fuse_base" and hasattr(self.base, "fuse_att"):
            x_mm = torch.stack([z_img, z_text], dim=1)
            len_i = torch.ones(z_img.shape[0], dtype=torch.long, device=z_img.device)
            len_t = torch.ones(z_text.shape[0], dtype=torch.long, device=z_text.device)
            fused = self.base.fuse_att(x_mm, len_i, len_t, 1)
        else:
            fused = torch.cat([z_img, z_text], dim=1)
        return self.base.classifier(fused), fused

    def _contrastive_loss(self, z_img: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
        if z_img.shape[0] <= 1:
            return z_img.new_tensor(0.0)
        img = F.normalize(z_img, p=2, dim=1)
        text = F.normalize(z_text, p=2, dim=1)
        logits = torch.matmul(img, text.t()) / self.temperature
        labels = torch.arange(z_img.shape[0], device=z_img.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def _rebuild_features(self, dense_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_modalities, hidden_dim = dense_features.shape
        flat = dense_features.permute(1, 0, 2).reshape(batch_size * n_modalities, hidden_dim)
        norm_flat = F.normalize(flat, p=2, dim=1)
        similarity = torch.matmul(norm_flat, norm_flat.t()) / self.temperature
        similarity = similarity - torch.eye(similarity.shape[0], device=similarity.device) * 1e12
        weights = torch.softmax(similarity, dim=-1)
        rebuilt = torch.matmul(weights, flat)
        rebuilt = rebuilt.reshape(n_modalities, batch_size, hidden_dim).permute(1, 0, 2).contiguous()
        return rebuilt, weights

    def reset_trained_prompts_checklist(self):
        self.trained_prompts_checklist = torch.zeros_like(self.trained_prompts_checklist)

    def _mark_trained_prompts(self):
        if self.pool.top_k_idx is None:
            return
        if self.trained_prompts_checklist.device != self.pool.top_k_idx.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        with torch.no_grad():
            self.trained_prompts_checklist[self.pool.top_k_idx] += 1.0

    def reconfigure_embeddings(
        self,
        z_img: torch.Tensor,
        z_text: torch.Tensor,
        present_img: Optional[torch.Tensor] = None,
        present_text: Optional[torch.Tensor] = None,
        return_pepsy: bool = False,
    ):
        if present_img is None:
            present_img = torch.ones(z_img.shape[0], dtype=torch.bool, device=z_img.device)
        if present_text is None:
            present_text = torch.ones(z_text.shape[0], dtype=torch.bool, device=z_text.device)
        filled_features = self._fill_missing_with_mean(z_img, z_text, present_img, present_text)

        specific = self.specific_embedding.expand(filled_features.shape[0], -1, -1)
        dense_features = []
        reduced_sims = filled_features.new_tensor(0.0)
        prompt_indices = []
        for modality_idx in range(self.n_modalities):
            control_input = torch.cat([specific[:, modality_idx, :], filled_features[:, modality_idx, :]], dim=-1)
            reduced_sim, prompt_mean, idx = self.pool(control_input)
            reduced_sims = reduced_sims + reduced_sim
            prompt_indices.append(idx)
            projector_input = torch.cat(
                [specific[:, modality_idx, :], filled_features[:, modality_idx, :], prompt_mean],
                dim=-1,
            )
            dense_features.append(self.sim_projectors[modality_idx](projector_input))
        dense_features = torch.stack(dense_features, dim=1)
        reduced_sims = reduced_sims / float(self.n_modalities)

        contrastive_loss = self._contrastive_loss(dense_features[:, 0, :], dense_features[:, 1, :])
        rebuilt_features, rebuild_weights = self._rebuild_features(dense_features)
        fused_features, alpha_old = self.global_fuser(dense_features, rebuilt_features)
        logits, fused_embedding = self._classify_from_embeddings(fused_features[:, 0, :], fused_features[:, 1, :])

        if self.training:
            self._mark_trained_prompts()

        if return_pepsy:
            aux: Dict[str, torch.Tensor] = {
                "contrastive_loss": contrastive_loss,
                "prompt_loss": reduced_sims,
                "present_img": present_img.float().mean(),
                "present_text": present_text.float().mean(),
                "alpha_old": alpha_old.detach().mean(),
                "rebuild_weight_mean": rebuild_weights.detach().mean(),
            }
            return logits, fused_embedding, aux
        return logits, fused_embedding

    def forward(self, x_img, x_text, len_i, len_t, return_pepsy: bool = False):
        z_img, z_text, present_img, present_text = self._encode_modalities(x_img, x_text, len_i, len_t)
        return self.reconfigure_embeddings(
            z_img,
            z_text,
            present_img=present_img,
            present_text=present_text,
            return_pepsy=return_pepsy,
        )
