from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(fake_image: torch.Tensor, reference_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((fake_image - reference_image) * mask))


def semantic_preservation_loss(fake_image: torch.Tensor, clean_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    preserve_mask = 1.0 - mask
    return torch.mean(torch.abs((fake_image - clean_image) * preserve_mask))


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_loss = torch.mean(F.relu(1.0 - real_logits))
    fake_loss = torch.mean(F.relu(1.0 + fake_logits))
    return real_loss + fake_loss


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -torch.mean(fake_logits)
