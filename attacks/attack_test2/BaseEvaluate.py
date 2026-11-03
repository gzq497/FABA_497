from __future__ import annotations

from typing import Dict, Optional

import torch

try:
    from .BaseUpdate import AttackTest1Pathways
    from .BaseTriggerDesign import CharBeamTextTrigger, OptimizableImageTrigger
except ImportError:  # pragma: no cover - allows direct script-style imports
    from attacks.attack_test2.BaseUpdate import AttackTest1Pathways
    from attacks.attack_test2.BaseTriggerDesign import CharBeamTextTrigger, OptimizableImageTrigger


class AttackTest1Evaluator(object):
    """Server-side metrics for attack_test1.

    Reported rates are fractions in [0, 1], matching the mathematical
    definitions: correct_count / total_count and target_count / attack_count.
    """

    def __init__(
        self,
        model,
        device,
        target_label: int,
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
    ):
        self.model = model
        self.device = device
        self.target_label = int(target_label)
        self.trigger_size = max(1, int(trigger_size))
        self.trigger_value = float(trigger_value)
        self.pathways = AttackTest1Pathways(self.model)
        self.image_trigger_tensor = image_trigger
        self.image_trigger_std = float(image_trigger_std)
        self.image_trigger_eps = float(image_trigger_eps)
        self.eval_trigger_seed = int(eval_trigger_seed)
        self.image_trigger = None
        self.text_trigger = CharBeamTextTrigger(
            charset=text_charset,
            budget=text_trigger_budget,
            beam_width=1,
            scale=text_trigger_scale,
        )
        if text_trigger is not None:
            self.text_trigger.best_string = str(text_trigger)

    def _unpack_batch(self, batch_data):
        if len(batch_data) == 6:
            x_I, x_T, len_I, len_T, y, _ = batch_data
        else:
            x_I, x_T, len_I, len_T, y = batch_data
        return (
            x_I.to(self.device),
            x_T.to(self.device),
            len_I.to(self.device),
            len_T.to(self.device),
            y.to(self.device),
        )

    def _ensure_image_trigger(self, feature_dim: int):
        if self.image_trigger is not None:
            return
        self.image_trigger = OptimizableImageTrigger(
            feature_dim=feature_dim,
            init_std=self.image_trigger_std,
            eps=self.image_trigger_eps,
            device=self.device,
        )
        if self.image_trigger_tensor is not None:
            trigger_tensor = torch.as_tensor(
                self.image_trigger_tensor,
                dtype=self.image_trigger.noise.dtype,
                device=self.device,
            ).view(1, 1, feature_dim)
        else:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(self.eval_trigger_seed)
            trigger_tensor = torch.randn(
                1,
                1,
                feature_dim,
                generator=gen,
                dtype=self.image_trigger.noise.dtype,
            ).to(self.device)
            trigger_tensor = trigger_tensor * self.image_trigger_std
        with torch.no_grad():
            self.image_trigger.noise.copy_(trigger_tensor)
            self.image_trigger.clamp_()

    def _image_trigger(self, x, lengths):
        self._ensure_image_trigger(x.shape[-1])
        return self.image_trigger(x, lengths)

    def _logits_from_inputs(
        self,
        x_I,
        x_T,
        len_I,
        len_T,
        poison_I: bool = False,
        poison_T: bool = False,
        zero_I: bool = False,
        zero_T: bool = False,
    ):
        if poison_I:
            x_I = self._image_trigger(x_I, len_I)
        if poison_T:
            x_T = self.text_trigger.apply(x_T, len_T)
        z_I, z_T = self.pathways.encode_pair(x_I.float(), x_T.float(), len_I, len_T)
        present_I = len_I > 0
        present_T = len_T > 0
        if zero_I:
            z_I = torch.zeros_like(z_I)
            present_I = torch.zeros_like(present_I)
        if zero_T:
            z_T = torch.zeros_like(z_T)
            present_T = torch.zeros_like(present_T)
        logits, _ = self.pathways.logits_from_embeddings(
            z_I,
            z_T,
            present_I=present_I,
            present_T=present_T,
        )
        return logits

    @torch.no_grad()
    def clean_accuracy(self, clean_dataloader) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        for batch_data in clean_dataloader:
            x_I, x_T, len_I, len_T, y = self._unpack_batch(batch_data)
            logits = self._logits_from_inputs(x_I, x_T, len_I, len_T)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        ca = float(correct / total) if total > 0 else 0.0
        return {'CA': ca, 'clean_correct': correct, 'clean_total': total}

    @torch.no_grad()
    def attack_metrics(
        self,
        attack_dataloader,
        inputs_are_poisoned: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        total = 0
        joint_success = 0
        image_success = 0
        text_success = 0
        conf_joint_sum = 0.0
        conf_I_sum = 0.0
        conf_T_sum = 0.0

        for batch_data in attack_dataloader:
            x_I, x_T, len_I, len_T, y = self._unpack_batch(batch_data)
            mask = y != self.target_label
            if not mask.any():
                continue
            x_I = x_I[mask]
            x_T = x_T[mask]
            len_I = len_I[mask]
            len_T = len_T[mask]
            batch_n = int(mask.sum().item())

            poison_flag = not inputs_are_poisoned
            logits_joint = self._logits_from_inputs(
                x_I, x_T, len_I, len_T, poison_I=poison_flag, poison_T=poison_flag
            )
            logits_I = self._logits_from_inputs(
                x_I, x_T, len_I, len_T, poison_I=poison_flag, zero_T=True
            )
            logits_T = self._logits_from_inputs(
                x_I, x_T, len_I, len_T, poison_T=poison_flag, zero_I=True
            )

            pred_joint = torch.argmax(logits_joint, dim=1)
            pred_I = torch.argmax(logits_I, dim=1)
            pred_T = torch.argmax(logits_T, dim=1)
            joint_success += int((pred_joint == self.target_label).sum().item())
            image_success += int((pred_I == self.target_label).sum().item())
            text_success += int((pred_T == self.target_label).sum().item())

            conf_joint = torch.softmax(logits_joint, dim=1)[:, self.target_label]
            conf_I = torch.softmax(logits_I, dim=1)[:, self.target_label]
            conf_T = torch.softmax(logits_T, dim=1)[:, self.target_label]
            conf_joint_sum += float(conf_joint.sum().item())
            conf_I_sum += float(conf_I.sum().item())
            conf_T_sum += float(conf_T.sum().item())
            total += batch_n

        asr_joint = float(joint_success / total) if total > 0 else 0.0
        asr_I = float(image_success / total) if total > 0 else 0.0
        asr_T = float(text_success / total) if total > 0 else 0.0
        conf_joint_mean = float(conf_joint_sum / total) if total > 0 else 0.0
        conf_I_mean = float(conf_I_sum / total) if total > 0 else 0.0
        conf_T_mean = float(conf_T_sum / total) if total > 0 else 0.0
        TMA_I = conf_joint_mean - conf_T_mean
        TMA_T = conf_joint_mean - conf_I_mean
        denom = max(TMA_I, TMA_T)
        MBR = float(min(TMA_I, TMA_T) / denom) if denom > 1e-12 else 0.0

        return {
            'ASR_Joint': asr_joint,
            'ASR_I': asr_I,
            'ASR_T': asr_T,
            'Conf_Joint': conf_joint_mean,
            'Conf_I': conf_I_mean,
            'Conf_T': conf_T_mean,
            'TMA_I': float(TMA_I),
            'TMA_T': float(TMA_T),
            'MBR': MBR,
            'attack_total': total,
            'joint_success': joint_success,
            'image_success': image_success,
            'text_success': text_success,
        }

    @torch.no_grad()
    def evaluate(
        self,
        clean_dataloader,
        attack_dataloader=None,
        inputs_are_poisoned: bool = False,
    ) -> Dict[str, float]:
        if attack_dataloader is None:
            attack_dataloader = clean_dataloader
        result = self.clean_accuracy(clean_dataloader)
        result.update(
            self.attack_metrics(
                attack_dataloader,
                inputs_are_poisoned=inputs_are_poisoned,
            )
        )
        return result


def compute_attack_test1_metrics(
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
    evaluator = AttackTest1Evaluator(
        model=model,
        device=device,
        target_label=target_label,
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
    )
    return evaluator.evaluate(
        clean_dataloader=clean_dataloader,
        attack_dataloader=attack_dataloader,
        inputs_are_poisoned=inputs_are_poisoned,
    )
