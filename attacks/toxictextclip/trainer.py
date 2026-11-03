from __future__ import annotations

import torch

from attacks.bad_mfl.trainer import ClientBadMFL


class ClientToxicTextCLIP(ClientBadMFL):
    """Text-only ToxicTextCLIP client with completion-strategy-aware training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attack_weight = float(getattr(self.args, "toxictextclip_attack_weight", -1.0))
        if attack_weight > 0:
            self.poison_weight = min(max(attack_weight, 0.0), 1.0)

    def _combine_bad_mfl_losses(self, log_probs, y, poison_mask):
        poison_count = int(poison_mask.sum().item())
        clean_mask = ~poison_mask
        clean_count = int(clean_mask.sum().item())

        poison_loss = (
            self.criterion(log_probs[poison_mask], y[poison_mask])
            if poison_count > 0
            else torch.tensor(0.0, device=self.device)
        )
        clean_loss = (
            self.criterion(log_probs[clean_mask], y[clean_mask])
            if clean_count > 0
            else torch.tensor(0.0, device=self.device)
        )

        if poison_count > 0 and clean_count > 0:
            weighted_loss = self.poison_weight * poison_loss + (1.0 - self.poison_weight) * clean_loss
        elif poison_count > 0:
            weighted_loss = poison_loss
        else:
            weighted_loss = clean_loss

        return weighted_loss, {
            "toxictextclip_poison_loss": float(poison_loss.detach().item()),
            "toxictextclip_clean_loss": float(clean_loss.detach().item()),
            "toxictextclip_weighted_loss": float(weighted_loss.detach().item()),
            "toxictextclip_poison_samples": poison_count,
            "toxictextclip_poison_weight": float(self.poison_weight),
        }
