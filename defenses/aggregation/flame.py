"""FLAME aggregation: clustering, adaptive clipping, and Gaussian noise."""

from collections import Counter
from typing import Optional, Sequence

import numpy as np
import torch

from .common import (
    AggregationResult,
    StateDict,
    StateDictVectorizer,
    fedavg_fallback,
)

try:
    import hdbscan as _hdbscan
except ImportError:
    _hdbscan = None


def ensure_flame_dependency():
    if _hdbscan is None:
        raise RuntimeError(
            "FLAME requires hdbscan. Install the pinned project dependency before use."
        )


def _cosine_distance(vectors: torch.Tensor) -> np.ndarray:
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = vectors / norms
    distance = (1.0 - normalized @ normalized.T).clamp(min=0.0, max=2.0)
    distance.fill_diagonal_(0.0)
    return distance.cpu().double().numpy()


def _largest_cluster(labels: Sequence[int]):
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    if not counts:
        return []
    largest_label = max(counts, key=lambda label: (counts[label], -label))
    return [index for index, label in enumerate(labels) if int(label) == largest_label]


def aggregate_flame(
    args,
    client_states: Sequence[StateDict],
    global_state: StateDict,
    parameter_names,
    sample_counts: Optional[Sequence[int]] = None,
    round_index: int = 0,
) -> AggregationResult:
    vectorizer = StateDictVectorizer(global_state, parameter_names)
    updates = vectorizer.stack_updates(client_states)
    num_clients = int(updates.shape[0])
    if num_clients < 3:
        reason = f"FLAME clustering needs at least 3 clients, got {num_clients}"
        if str(getattr(args, "robust_fallback", "fedavg")) == "error":
            raise ValueError(reason)
        return fedavg_fallback(vectorizer, updates, sample_counts, reason, "flame")

    ensure_flame_dependency()

    cluster_on = str(getattr(args, "flame_cluster_on", "model")).lower()
    cluster_vectors = (
        vectorizer.stack_states(client_states) if cluster_on == "model" else updates
    )
    min_cluster_size = int(getattr(args, "flame_min_cluster_size", -1))
    if min_cluster_size <= 0:
        min_cluster_size = num_clients // 2 + 1
    min_cluster_size = min(max(min_cluster_size, 2), num_clients)
    min_samples = max(1, int(getattr(args, "flame_min_samples", 1)))

    clusterer = _hdbscan.HDBSCAN(
        metric="precomputed",
        algorithm="generic",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(_cosine_distance(cluster_vectors))
    accepted = _largest_cluster(labels)
    if not accepted:
        reason = "HDBSCAN did not return a non-noise cluster"
        if str(getattr(args, "robust_fallback", "fedavg")) == "error":
            raise RuntimeError(reason)
        result = fedavg_fallback(vectorizer, updates, sample_counts, reason, "flame")
        result.metadata["cluster_labels"] = [int(label) for label in labels]
        return result

    update_norms = torch.linalg.vector_norm(updates, dim=1)
    clip_norm = torch.median(update_norms)
    accepted_updates = updates[accepted]
    accepted_norms = torch.linalg.vector_norm(accepted_updates, dim=1).clamp_min(1e-12)
    scales = torch.clamp(clip_norm / accepted_norms, max=1.0)
    clipped_updates = accepted_updates * scales.unsqueeze(1)
    aggregate = clipped_updates.mean(dim=0)

    add_noise = bool(getattr(args, "flame_add_noise", True))
    multiplier = float(getattr(args, "flame_noise_multiplier", 1.0e-3))
    if multiplier < 0:
        raise ValueError("flame_noise_multiplier must be non-negative")
    noise_std = float(clip_norm) * multiplier if add_noise else 0.0
    if noise_std > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(args, "robust_seed", 2024)) + int(round_index))
        aggregate = aggregate + torch.randn(
            aggregate.shape, generator=generator, dtype=aggregate.dtype
        ) * noise_std

    return AggregationResult(
        state_dict=vectorizer.apply_update(aggregate),
        accepted_indices=[int(index) for index in accepted],
        metadata={
            "method": "flame",
            "fallback": None,
            "num_clients": num_clients,
            "cluster_on": cluster_on,
            "cluster_labels": [int(label) for label in labels],
            "accepted_indices": [int(index) for index in accepted],
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "update_norms": [float(value) for value in update_norms.tolist()],
            "clip_norm": float(clip_norm),
            "clip_scales": [float(value) for value in scales.tolist()],
            "noise_multiplier": multiplier,
            "noise_std": noise_std,
        },
    )
