import copy

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .common import forward_logits


def _get_module(model, path):
    module = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def discover_linear_pairs(model):
    prefix = "base." if hasattr(model, "base") else ""
    pairs = [
        (prefix + "img_proj.0", prefix + "img_proj.3"),
        (prefix + "classifier.0", prefix + "classifier.3"),
    ]
    if hasattr(model, "sim_projectors"):
        for index in range(len(model.sim_projectors)):
            pairs.append((f"sim_projectors.{index}.0", f"sim_projectors.{index}.2"))
    if hasattr(model, "global_fuser"):
        pairs.append(("global_fuser.masking.0", "global_fuser.masking.2"))
    valid = []
    for first, second in pairs:
        try:
            left, right = _get_module(model, first), _get_module(model, second)
        except (AttributeError, IndexError, KeyError):
            continue
        if isinstance(left, torch.nn.Linear) and isinstance(right, torch.nn.Linear):
            if left.out_features == right.in_features:
                valid.append((first, second))
    return valid


def _collect_pair_activations(model_a, model_b, loader, device, paths, max_samples=4096):
    data_a, data_b = {path: [] for path in paths}, {path: [] for path in paths}
    hooks = []
    for path in paths:
        hooks.append(_get_module(model_a, path).register_forward_hook(
            lambda module, inputs, output, key=path: data_a[key].append(output.detach().cpu())
        ))
        hooks.append(_get_module(model_b, path).register_forward_hook(
            lambda module, inputs, output, key=path: data_b[key].append(output.detach().cpu())
        ))
    model_a.eval()
    model_b.eval()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            forward_logits(model_a, batch, device)
            forward_logits(model_b, batch, device)
            seen += int(batch[0].shape[0])
            if seen >= int(max_samples):
                break
    for hook in hooks:
        hook.remove()
    result = {}
    for path in paths:
        a = torch.cat(data_a[path], dim=0)[:max_samples]
        b = torch.cat(data_b[path], dim=0)[:max_samples]
        result[path] = (a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]))
    return result


def _assignment(source, target, maximize):
    cost = torch.cdist(source.t().float(), target.t().float(), p=2).cpu().numpy()
    rows, cols = linear_sum_assignment(cost, maximize=bool(maximize))
    permutation = np.empty(len(rows), dtype=np.int64)
    permutation[cols] = rows
    return torch.as_tensor(permutation, dtype=torch.long)


def _apply_pair_permutation(model, first_path, second_path, permutation):
    first = _get_module(model, first_path)
    second = _get_module(model, second_path)
    permutation = permutation.to(first.weight.device)
    with torch.no_grad():
        first.weight.copy_(first.weight.index_select(0, permutation))
        if first.bias is not None:
            first.bias.copy_(first.bias.index_select(0, permutation))
        second.weight.copy_(second.weight.index_select(1, permutation))


def _max_logit_error(before, after, loader, device):
    before.eval()
    after.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        logits_before, _ = forward_logits(before, batch, device)
        logits_after, _ = forward_logits(after, batch, device)
    return float(torch.max(torch.abs(logits_before - logits_after)).cpu())


def unalign_model(model, loader, device, tolerance=1e-5):
    source = copy.deepcopy(model).to(device).eval()
    result = copy.deepcopy(model).to(device).eval()
    pairs = discover_linear_pairs(result)
    activations = _collect_pair_activations(source, source, loader, device, [p[0] for p in pairs])
    assignments = {}
    for first, second in pairs:
        a, b = activations[first]
        permutation = _assignment(a, b, maximize=True)
        _apply_pair_permutation(result, first, second, permutation)
        assignments[first] = permutation.tolist()
    error = _max_logit_error(source, result, loader, device)
    if error > float(tolerance):
        raise RuntimeError(f"TSC unalignment broke function equivalence: max logit error {error:.3e}")
    return result, {"pairs": assignments, "max_logit_error": error}


def align_model(model, reference, loader, device, tolerance=1e-5):
    source = copy.deepcopy(model).to(device).eval()
    result = copy.deepcopy(model).to(device).eval()
    reference = reference.to(device).eval()
    pairs = discover_linear_pairs(result)
    activations = _collect_pair_activations(source, reference, loader, device, [p[0] for p in pairs])
    assignments = {}
    for first, second in pairs:
        source_act, ref_act = activations[first]
        permutation = _assignment(source_act, ref_act, maximize=False)
        _apply_pair_permutation(result, first, second, permutation)
        assignments[first] = permutation.tolist()
    error = _max_logit_error(source, result, loader, device)
    if error > float(tolerance):
        raise RuntimeError(f"TSC alignment broke function equivalence: max logit error {error:.3e}")
    return result, {"pairs": assignments, "max_logit_error": error}
