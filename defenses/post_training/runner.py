import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .clean_reference import build_clean_reference_loader
from .evaluator import clean_dev_evaluator, evaluate_model
from .par import run_par
from .tsc import run_tsc


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _checkpoint(model, server, args):
    payload = {
        "model": copy.deepcopy(model.state_dict()),
        "dataset": str(args.dataset),
        "missing_strategy": str(getattr(args, "missing_strategy", "zero")),
        "missing_rate": float(getattr(args, "missing_modailty_rate", 0.0)),
        "attack_type": str(getattr(args, "attack_type", "none")),
    }
    if hasattr(server, "pmcmfl_prototypes"):
        payload["pmcmfl_prototypes"] = server.get_pmcmfl_prototypes()
    return payload


def run_post_training_defense(
    args, server, dm, client_ids, dataloader_dict, attack_dataloader_dict,
    output_dir, fold_idx,
):
    defense = str(getattr(args, "server_defense", "none")).lower()
    if defense == "none":
        return None
    if str(getattr(args, "defense_checkpoint", "best_dev")) == "best_dev":
        state = getattr(server, "best_model_state", None)
        if state is None:
            raise RuntimeError("No best-dev checkpoint was captured for post-training defense")
        server.global_model.load_state_dict(state, strict=True)

    defense_dir = Path(output_dir).joinpath(f"post_{defense}", f"fold{fold_idx}")
    defense_dir.mkdir(parents=True, exist_ok=True)
    args.defense_img_shape = np.array([1, int(server.global_model.img_proj[0].in_features)])
    text_rnn = server.global_model.text_rnn
    args.defense_text_shape = np.array([32, int(text_rnn.input_size)])
    clean_loader, reference_metadata = build_clean_reference_loader(
        dm, client_ids, float(args.defense_clean_ratio), int(args.defense_clean_seed),
        metadata_path=defense_dir.joinpath("clean_reference.json"),
    )

    pre_model = copy.deepcopy(server.global_model).to(server.device)
    pre_metrics = evaluate_model(
        server, pre_model, dataloader_dict["dev"], dataloader_dict["test"],
        attack_dataloader_dict, int(args.bd_target_label),
    )
    torch.save(_checkpoint(pre_model, server, args), str(defense_dir.joinpath("pre_defense.pt")))
    eval_fn = clean_dev_evaluator(server, dataloader_dict["dev"])
    if defense == "par":
        defended_model, history = run_par(pre_model, clean_loader, server.device, args, eval_fn)
    elif defense == "tsc":
        defended_model, history = run_tsc(pre_model, clean_loader, server.device, args, eval_fn)
    else:
        raise ValueError(f"Unsupported post-training defense: {defense}")

    server.global_model = defended_model.to(server.device)
    post_metrics = evaluate_model(
        server, server.global_model, dataloader_dict["dev"], dataloader_dict["test"],
        attack_dataloader_dict, int(args.bd_target_label),
    )
    torch.save(_checkpoint(server.global_model, server, args), str(defense_dir.joinpath("post_defense.pt")))
    result = {
        "defense": defense,
        "fold": int(fold_idx),
        "reference": reference_metadata,
        "pre_defense": pre_metrics,
        "post_defense": post_metrics,
        "history": history,
    }
    with open(str(defense_dir.joinpath("defense_result.json")), "w", encoding="utf-8") as handle:
        json.dump(_json_ready(result), handle, indent=2, ensure_ascii=True)
    logging.info(
        "[%s] post-defense ACC %.2f%% ASR %.2f%% IA %.2f%% TA %.2f%%",
        defense.upper(),
        100.0 * post_metrics["summary"]["ACC"],
        100.0 * post_metrics["summary"]["ASR"],
        100.0 * post_metrics["summary"]["IA"],
        100.0 * post_metrics["summary"]["TA"],
    )
    return _json_ready(result)
