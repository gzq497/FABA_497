from __future__ import annotations

from typing import Dict

try:
    from .BaseEvaluate import AttackTest1Evaluator, compute_attack_test1_metrics
except ImportError:  # pragma: no cover
    from attacks.attack_test2.BaseEvaluate import AttackTest1Evaluator, compute_attack_test1_metrics


class AttackTest2Evaluator(AttackTest1Evaluator):
    """Server-side metrics for attack_test2.

    attack_test2 intentionally keeps the same evaluation protocol as
    attack_test1 so CA, joint ASR, image-only ASR, text-only ASR, TMA, and MBR
    remain directly comparable across the two methods.
    """


def compute_attack_test2_metrics(
    model,
    clean_dataloader,
    target_label: int,
    device,
    attack_dataloader=None,
    trigger_size: int = 16,
    trigger_value: float = 10.0,
    image_trigger=None,
    image_trigger_std: float = 0.05,
    image_trigger_eps: float = 0.5,
    eval_trigger_seed: int = 2024,
    text_trigger: str = None,
    text_charset: str = 'abcdefghijklmnopqrstuvwxyz0123456789',
    text_trigger_budget: int = 5,
    text_trigger_scale: float = 0.1,
    inputs_are_poisoned: bool = False,
) -> Dict[str, float]:
    return compute_attack_test1_metrics(
        model=model,
        clean_dataloader=clean_dataloader,
        target_label=target_label,
        device=device,
        attack_dataloader=attack_dataloader,
        trigger_size=trigger_size,
        trigger_value=trigger_value,
        image_trigger=image_trigger,
        image_trigger_std=image_trigger_std,
        image_trigger_eps=image_trigger_eps,
        eval_trigger_seed=eval_trigger_seed,
        text_trigger=text_trigger,
        text_charset=text_charset,
        text_trigger_budget=text_trigger_budget,
        text_trigger_scale=text_trigger_scale,
        inputs_are_poisoned=inputs_are_poisoned,
    )


__all__ = ["AttackTest2Evaluator", "compute_attack_test2_metrics"]
