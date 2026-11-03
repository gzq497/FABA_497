from __future__ import annotations

import torch

from attacks.bad_mfl.trainer import ClientBadMFL


class ClientBadCM(ClientBadMFL):
    """BadCM local client for the federated feature-space framework.

    BadCM's original implementation generates invisible image perturbations and
    semantically close text substitutions before model training. In this
    framework those poisoned image/text features are prepared offline under
    attacks/badcm, so the client-side trainer mirrors Bad-MFL's supervised
    poisoned-key weighting while reporting BadCM-specific stats.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        badcm_weight = float(getattr(self.args, "badcm_attack_weight", -1.0))
        if badcm_weight > 0:
            self.poison_weight = min(max(badcm_weight, 0.0), 1.0)

    def _combine_bad_mfl_losses(self, log_probs, y, poison_mask):
        poison_count = int(poison_mask.sum().item())
        clean_mask = ~poison_mask
        clean_count = int(clean_mask.sum().item())

        if poison_count > 0:
            poison_loss = self.criterion(log_probs[poison_mask], y[poison_mask])
        else:
            poison_loss = torch.tensor(0.0, device=self.device)
        if clean_count > 0:
            clean_loss = self.criterion(log_probs[clean_mask], y[clean_mask])
        else:
            clean_loss = torch.tensor(0.0, device=self.device)

        if poison_count > 0 and clean_count > 0:
            weighted_loss = self.poison_weight * poison_loss + (1.0 - self.poison_weight) * clean_loss
        elif poison_count > 0:
            weighted_loss = poison_loss
        else:
            weighted_loss = clean_loss

        return weighted_loss, {
            "badcm_poison_loss": float(poison_loss.detach().item()),
            "badcm_clean_loss": float(clean_loss.detach().item()),
            "badcm_weighted_loss": float(weighted_loss.detach().item()),
            "badcm_poison_samples": poison_count,
            "badcm_poison_weight": float(self.poison_weight),
        }
