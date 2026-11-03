"""Multi-Krum robust aggregation."""

import math
from typing import Optional, Sequence

import torch

from .common import (
    AggregationResult,
    StateDict,
    StateDictVectorizer,
    fedavg_fallback,
)


def _resolve_byzantine_count(args, num_clients: int) -> int:
    configured = int(getattr(args, "robust_byzantine_count", -1))
    if configured >= 0:
        return configured
    ratio = float(getattr(args, "robust_byzantine_ratio", -1.0))
    if ratio < 0:
        ratio = float(getattr(args, "bd_malicious_ratio", 0.1))
    if ratio < 0 or ratio >= 0.5:
        raise ValueError("The assumed Byzantine ratio must be in [0, 0.5)")
    return int(math.ceil(ratio * num_clients)) if ratio > 0 else 0


def aggregate_multi_krum(
    args,
    client_states: Sequence[StateDict],
    global_state: StateDict,
    parameter_names,
    sample_counts: Optional[Sequence[int]] = None,
    round_index: int = 0,
) -> AggregationResult:
    del round_index
    vectorizer = StateDictVectorizer(global_state, parameter_names)
    updates = vectorizer.stack_updates(client_states)
    num_clients = int(updates.shape[0])
    num_byzantine = _resolve_byzantine_count(args, num_clients)

    if num_byzantine < 0 or num_clients < 2 * num_byzantine + 3:
        reason = (
            f"Multi-Krum requires n >= 2f + 3, got n={num_clients}, "
            f"f={num_byzantine}"
        )
        if str(getattr(args, "robust_fallback", "fedavg")) == "error":
            raise ValueError(reason)
        return fedavg_fallback(
            vectorizer, updates, sample_counts, reason, "multi_krum"
        )

    neighbor_count = num_clients - num_byzantine - 2
    configured_m = int(getattr(args, "multi_krum_m", -1))
    selected_count = neighbor_count if configured_m <= 0 else configured_m
    if selected_count < 1 or selected_count > neighbor_count:
        raise ValueError(
            f"multi_krum_m must be in [1, {neighbor_count}], got {selected_count}"
        )

    distances = torch.cdist(updates, updates, p=2).pow(2)
    distances.fill_diagonal_(float("inf"))
    nearest = torch.topk(
        distances, k=neighbor_count, largest=False, sorted=False, dim=1
    ).values
    scores = nearest.sum(dim=1)
    accepted = torch.argsort(scores)[:selected_count].tolist()
    aggregate = updates[accepted].mean(dim=0)

    return AggregationResult(
        state_dict=vectorizer.apply_update(aggregate),
        accepted_indices=[int(index) for index in accepted],
        metadata={
            "method": "multi_krum",
            "fallback": None,
            "num_clients": num_clients,
            "num_byzantine": num_byzantine,
            "neighbor_count": neighbor_count,
            "selected_count": selected_count,
            "accepted_indices": [int(index) for index in accepted],
            "scores": [float(value) for value in scores.tolist()],
        },
    )

