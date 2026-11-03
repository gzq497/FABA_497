"""Utilities shared by robust model aggregators."""

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch


StateDict = Mapping[str, torch.Tensor]


@dataclass
class AggregationResult:
    state_dict: Dict[str, torch.Tensor]
    accepted_indices: List[int]
    metadata: Dict[str, object]


class StateDictVectorizer:
    """Flatten trainable state entries and reconstruct an update safely."""

    def __init__(self, reference: StateDict, parameter_names: Iterable[str]):
        parameter_names = set(parameter_names)
        self.reference = reference
        self.keys = [
            key
            for key, value in reference.items()
            if key in parameter_names and torch.is_floating_point(value)
        ]
        if not self.keys:
            raise ValueError("No floating-point trainable parameters were found")
        self.numels = [int(reference[key].numel()) for key in self.keys]

    def _validate(self, state: StateDict) -> None:
        for key in self.keys:
            if key not in state:
                raise KeyError(f"Client state is missing trainable parameter: {key}")
            if tuple(state[key].shape) != tuple(self.reference[key].shape):
                raise ValueError(
                    f"Shape mismatch for {key}: {tuple(state[key].shape)} != "
                    f"{tuple(self.reference[key].shape)}"
                )

    def flatten_state(self, state: StateDict) -> torch.Tensor:
        self._validate(state)
        return torch.cat(
            [state[key].detach().cpu().float().reshape(-1) for key in self.keys]
        )

    def flatten_update(self, state: StateDict) -> torch.Tensor:
        self._validate(state)
        return torch.cat(
            [
                (
                    state[key].detach().cpu().float()
                    - self.reference[key].detach().cpu().float()
                ).reshape(-1)
                for key in self.keys
            ]
        )

    def stack_states(self, states: Sequence[StateDict]) -> torch.Tensor:
        return torch.stack([self.flatten_state(state) for state in states], dim=0)

    def stack_updates(self, states: Sequence[StateDict]) -> torch.Tensor:
        return torch.stack([self.flatten_update(state) for state in states], dim=0)

    def apply_update(self, update: torch.Tensor) -> Dict[str, torch.Tensor]:
        if update.ndim != 1 or int(update.numel()) != sum(self.numels):
            raise ValueError("Aggregated update has an incompatible shape")
        result = copy.deepcopy(self.reference)
        offset = 0
        for key, numel in zip(self.keys, self.numels):
            reference = self.reference[key]
            value = update[offset : offset + numel].reshape(reference.shape)
            result[key] = (
                reference.detach().cpu().float() + value
            ).to(device=reference.device, dtype=reference.dtype)
            offset += numel
        return result


def weighted_mean_update(
    updates: torch.Tensor,
    sample_counts: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if updates.ndim != 2 or updates.shape[0] == 0:
        raise ValueError("Expected a non-empty [clients, parameters] update matrix")
    if sample_counts is None:
        return updates.mean(dim=0)
    weights = torch.as_tensor(sample_counts, dtype=updates.dtype)
    if weights.numel() != updates.shape[0] or float(weights.sum()) <= 0:
        raise ValueError("Invalid client sample counts")
    weights = weights / weights.sum()
    return torch.sum(updates * weights.unsqueeze(1), dim=0)


def fedavg_fallback(
    vectorizer: StateDictVectorizer,
    updates: torch.Tensor,
    sample_counts: Optional[Sequence[int]],
    reason: str,
    method: str,
) -> AggregationResult:
    aggregate = weighted_mean_update(updates, sample_counts)
    return AggregationResult(
        state_dict=vectorizer.apply_update(aggregate),
        accepted_indices=list(range(int(updates.shape[0]))),
        metadata={
            "method": method,
            "fallback": "fedavg",
            "fallback_reason": reason,
            "num_clients": int(updates.shape[0]),
        },
    )

