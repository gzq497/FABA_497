import logging

import torch
import torch.nn.functional as F

from .bezier import train_bezier_curve
from .common import forward_logits
from .permutation import align_model, unalign_model


def _fine_tune(model, loader, device, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    history = []
    for epoch in range(int(epochs)):
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            logits, y = forward_logits(model, batch, device)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        history.append({"epoch": epoch + 1, "loss": total / max(batches, 1)})
    return model, history


def run_tsc(model, clean_loader, device, args, clean_eval_fn=None):
    current = model.to(device)
    history = []
    for round_index in range(int(args.tsc_rounds)):
        logging.info("[TSC] purification round %d/%d", round_index + 1, int(args.tsc_rounds))
        unaligned, unalign_meta = unalign_model(
            current, clean_loader, device, tolerance=float(args.tsc_equivalence_tol)
        )
        stage_one, curve_one = train_bezier_curve(
            current, unaligned, clean_loader, device,
            epochs=int(args.tsc_curve_epochs), lr=float(args.tsc_curve_lr),
            t=float(args.tsc_curve_t), eval_interval=int(args.tsc_eval_interval),
            eval_fn=clean_eval_fn, tag=f"r{round_index + 1}-stage1",
        )
        aligned, align_meta = align_model(
            stage_one, current, clean_loader, device, tolerance=float(args.tsc_equivalence_tol)
        )
        stage_two, curve_two = train_bezier_curve(
            current, aligned, clean_loader, device,
            epochs=int(args.tsc_curve_epochs), lr=float(args.tsc_curve_lr),
            t=float(args.tsc_curve_t), eval_interval=int(args.tsc_eval_interval),
            eval_fn=clean_eval_fn, tag=f"r{round_index + 1}-stage2",
        )
        round_history = {
            "round": round_index + 1,
            "unalign": unalign_meta,
            "align": align_meta,
            "stage1": curve_one,
            "stage2": curve_two,
        }
        current = stage_two
        if round_index + 1 < int(args.tsc_rounds):
            current, fine_tune_history = _fine_tune(
                current, clean_loader, device,
                int(args.tsc_finetune_epochs), float(args.tsc_finetune_lr),
            )
            round_history["fine_tune"] = fine_tune_history
        history.append(round_history)
    return current, history
