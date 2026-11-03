import copy
import math

import torch
import torch.nn.functional as F


def unpack_batch(batch, device):
    if len(batch) < 5:
        raise ValueError("Post-training defenses require multimodal batches")
    x_i, x_t, len_i, len_t, y = batch[:5]
    return (
        x_i.to(device).float(),
        x_t.to(device).float(),
        len_i.to(device),
        len_t.to(device),
        y.to(device).long(),
    )


def forward_logits(model, batch, device):
    x_i, x_t, len_i, len_t, y = unpack_batch(batch, device)
    output = model(x_i, x_t, len_i, len_t)
    return output[0], y


def forward_with_embeddings(model, batch, device):
    x_i, x_t, len_i, len_t, y = unpack_batch(batch, device)
    if hasattr(model, "base") and hasattr(model, "_encode_modalities"):
        z_i, z_t, present_i, present_t = model._encode_modalities(x_i, x_t, len_i, len_t)
        logits, _ = model.reconfigure_embeddings(
            z_i, z_t, present_img=present_i, present_text=present_t
        )
        return logits, z_i, z_t, y
    output = model(x_i, x_t, len_i, len_t, return_aux=True)
    logits, _, aux = output
    return logits, aux["img_embed"], aux["text_embed"], y


def classification_loss(logits, y):
    return F.cross_entropy(logits, y)


def cosine_with_warmup(step, total_steps, warmup_steps, start_lr, peak_lr, end_lr):
    if total_steps <= 1:
        return peak_lr
    if warmup_steps > 0 and step < warmup_steps:
        frac = float(step + 1) / float(warmup_steps)
        return start_lr + frac * (peak_lr - start_lr)
    remain = max(total_steps - warmup_steps, 1)
    frac = min(max(float(step - warmup_steps) / float(remain), 0.0), 1.0)
    return end_lr + 0.5 * (peak_lr - end_lr) * (1.0 + math.cos(math.pi * frac))


def clone_model(model):
    return copy.deepcopy(model)
