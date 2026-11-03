import copy
import logging

import torch
import torch.nn.functional as F

from .common import classification_loss, cosine_with_warmup, forward_with_embeddings


def run_par(model, clean_loader, device, args, clean_eval_fn=None):
    reference = copy.deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    model = model.to(device)
    model.train()

    epochs = int(args.par_epochs)
    total_steps = max(1, epochs * len(clean_loader))
    warmup_steps = int(total_steps * float(args.par_warmup_ratio))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.par_lr), weight_decay=float(args.par_weight_decay),
        betas=(0.9, 0.98), eps=1e-8,
    )
    global_step = 0
    history = []
    for epoch in range(epochs):
        model.train()
        totals = {"loss": 0.0, "clean": 0.0, "pert": 0.0, "batches": 0}
        for batch in clean_loader:
            lr = cosine_with_warmup(
                global_step, total_steps, warmup_steps,
                float(args.par_warmup_start_lr), float(args.par_lr), float(args.par_end_lr),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad()
            logits, z_i, z_t, y = forward_with_embeddings(model, batch, device)
            with torch.no_grad():
                _, ref_i, ref_t, _ = forward_with_embeddings(reference, batch, device)
            z_i = F.normalize(z_i, p=2, dim=-1)
            z_t = F.normalize(z_t, p=2, dim=-1)
            ref_i = F.normalize(ref_i, p=2, dim=-1)
            ref_t = F.normalize(ref_t, p=2, dim=-1)
            shift_i = torch.mean(torch.sum((z_i - ref_i) ** 2, dim=-1))
            shift_t = torch.mean(torch.sum((z_t - ref_t) ** 2, dim=-1))
            tau = float(args.par_tau)
            perturb = 0.5 * (
                (shift_i <= tau).float().detach() * shift_i
                + (shift_t <= tau).float().detach() * shift_t
            )
            clean = classification_loss(logits, y)
            loss = clean - float(args.par_lambda_pert) * perturb
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["clean"] += float(clean.detach())
            totals["pert"] += float(perturb.detach())
            totals["batches"] += 1
            global_step += 1
        denom = max(totals.pop("batches"), 1)
        row = {key: value / denom for key, value in totals.items()}
        row.update({"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]})
        if clean_eval_fn is not None:
            row["dev"] = clean_eval_fn(model)
        history.append(row)
        logging.info(
            "[PAR] epoch %d/%d loss %.4f clean %.4f perturb %.4f lr %.3e",
            epoch + 1, epochs, row["loss"], row["clean"], row["pert"], row["lr"],
        )
    return model, history
