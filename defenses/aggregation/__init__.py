"""Robust aggregation entry points."""

# Load the native HDBSCAN extension before CUDA/data-loader initialization.
# Importing it for the first time after local training can crash with the
# OpenMP runtime combination used by the pinned Torch 1.13 environment.
from .flame import ensure_flame_dependency
from .registry import aggregate_state_dicts

__all__ = ["aggregate_state_dicts", "ensure_flame_dependency"]
