"""Command-line arguments shared by all multimodal experiment entry points."""

import argparse


def add_robust_aggregation_args(parser):
    group = parser.add_argument_group("robust server aggregation")
    group.add_argument(
        "--robust_aggregation",
        choices=["fedavg", "multi_krum", "flame"],
        default="fedavg",
    )
    group.add_argument("--robust_byzantine_count", type=int, default=-1)
    group.add_argument(
        "--robust_byzantine_ratio",
        type=float,
        default=-1.0,
        help="Negative uses bd_malicious_ratio without revealing client identities.",
    )
    group.add_argument(
        "--robust_fallback", choices=["fedavg", "error"], default="fedavg"
    )
    group.add_argument("--robust_seed", type=int, default=2024)
    group.add_argument(
        "--multi_krum_m",
        type=int,
        default=-1,
        help="Negative selects n-f-2 client updates.",
    )
    group.add_argument("--flame_min_cluster_size", type=int, default=-1)
    group.add_argument("--flame_min_samples", type=int, default=1)
    group.add_argument(
        "--flame_cluster_on", choices=["model", "update"], default="model"
    )
    group.add_argument("--flame_noise_multiplier", type=float, default=1.0e-3)
    group.add_argument(
        "--flame_add_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument(
        "--robust_log_details",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser
