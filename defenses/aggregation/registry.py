"""Registry and stable public API for robust aggregation."""

from typing import Optional, Sequence

from .common import AggregationResult, StateDict


def aggregate_state_dicts(
    args,
    client_states: Sequence[StateDict],
    global_state: StateDict,
    parameter_names,
    sample_counts: Optional[Sequence[int]] = None,
    round_index: int = 0,
) -> AggregationResult:
    method = str(getattr(args, "robust_aggregation", "fedavg")).lower()
    if method == "multi_krum":
        from .multi_krum import aggregate_multi_krum

        return aggregate_multi_krum(
            args,
            client_states,
            global_state,
            parameter_names,
            sample_counts,
            round_index,
        )
    if method == "flame":
        from .flame import aggregate_flame

        return aggregate_flame(
            args,
            client_states,
            global_state,
            parameter_names,
            sample_counts,
            round_index,
        )
    raise ValueError(f"Unsupported robust aggregation method: {method}")

