from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

try:
    from .BaseUpdate import LocalUpdate as AttackTest2BaseLocalUpdate
except ImportError:  # pragma: no cover
    from attacks.attack_test2.BaseUpdate import LocalUpdate as AttackTest2BaseLocalUpdate

try:
    from .TriggerDesign import VLTrojanFeatureImageTrigger, vl_trojan_feature_loss
except ImportError:  # pragma: no cover
    from attacks.attack_test2.TriggerDesign import (
        VLTrojanFeatureImageTrigger,
        vl_trojan_feature_loss,
    )


class LocalUpdate(AttackTest2BaseLocalUpdate):
    """Local malicious client trainer for attack_test2.

    attack_test2 is a modal-split, joint-consistent variant of attack_test1:
    image-only and text-only visible samples keep the zero-vector fallback
    objective, while samples with both modalities additionally receive an
    explicit joint-trigger target objective and DAB-style alignment pressure.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lambda_clean = float(getattr(self.args, 'attack_test2_lambda_clean', 1.0))
        self.lambda_joint = float(getattr(self.args, 'attack_test2_lambda_joint', 1.5))
        self.lambda_img = float(getattr(self.args, 'attack_test2_lambda_img', 1.0))
        self.lambda_text = float(getattr(self.args, 'attack_test2_lambda_text', 1.0))
        self.lambda_aux = float(getattr(self.args, 'attack_test2_lambda_aux', 0.5))
        self.lambda_anti = float(getattr(self.args, 'attack_test2_lambda_anti', 0.5))
        self.lambda_cons = float(getattr(self.args, 'attack_test2_lambda_cons', 0.5))
        self.lambda_logit_cons = float(getattr(self.args, 'attack_test2_lambda_logit_cons', 0.5))
        self.lambda_align = float(getattr(self.args, 'attack_test2_lambda_align', 0.1))
        self.lambda_margin = float(getattr(self.args, 'attack_test2_lambda_margin', 0.5))
        self.lambda_reg_test2 = float(getattr(self.args, 'attack_test2_lambda_reg', 1e-3))
        self.joint_margin = float(getattr(self.args, 'attack_test2_joint_margin', 1.0))
        self.dynamic_max = float(getattr(self.args, 'attack_test2_dynamic_max', 5.0))
        self.joint_boost_threshold = float(getattr(self.args, 'attack_test2_joint_boost_threshold', 0.8))
        self.joint_boost_step = float(getattr(self.args, 'attack_test2_joint_boost_step', 1.15))
        self.joint_decay_step = float(getattr(self.args, 'attack_test2_joint_decay_step', 0.97))
        self.weight_joint = 1.0
        self.lambda_vl_img = float(getattr(self.args, 'attack_test2_lambda_vl_img', 0.2))
        self.vl_img_source_weight = float(getattr(self.args, 'attack_test2_vl_img_source_weight', 1.0))
        self.vl_img_cluster_weight = float(getattr(self.args, 'attack_test2_vl_img_cluster_weight', 1.0))
        self.vl_img_align_weight = float(getattr(self.args, 'attack_test2_vl_img_align_weight', 0.5))
        self.vl_img_opposite_weight = float(getattr(self.args, 'attack_test2_vl_img_opposite_weight', 0.1))
        self.vl_img_opposite_margin = float(getattr(self.args, 'attack_test2_vl_img_opposite_margin', 0.05))
        self.vl_img_trigger_mode = str(getattr(self.args, 'attack_test2_img_trigger_mode', 'vl_blend'))
        self.vl_img_blend_alpha = float(getattr(self.args, 'attack_test2_img_blend_alpha', 0.1))
        self.vl_img_patch_ratio = float(getattr(self.args, 'attack_test2_img_patch_ratio', 1.0))
        self.vl_img_patch_position = str(getattr(self.args, 'attack_test2_img_patch_position', 'tail'))
        self.use_raw_trigger_features = bool(getattr(self.args, 'attack_test2_use_raw_trigger_features', False))

        self.save_poisoned_samples = bool(
            getattr(
                self.args,
                'attack_test2_save_poisoned_samples',
                getattr(self.args, 'attack_test1_save_poisoned_samples', True),
            )
        )
        self.poison_save_limit = max(
            0,
            int(
                getattr(
                    self.args,
                    'attack_test2_poison_save_limit',
                    getattr(self.args, 'attack_test1_poison_save_limit', 64),
                )
            ),
        )
        self.poison_save_interval = int(
            getattr(
                self.args,
                'attack_test2_poison_save_interval',
                getattr(self.args, 'attack_test1_poison_save_interval', 50),
            )
        )
        self.artifact_dir = getattr(self.args, 'attack_test2_artifact_dir', self.artifact_dir)

    def _refresh_text_trigger(self, *args, **kwargs):
        if self.use_raw_trigger_features:
            return
        return super()._refresh_text_trigger(*args, **kwargs)

    def _make_poisoned_inputs(self, x_I, x_T, len_I, len_T):
        if self.use_raw_trigger_features:
            return x_I.clone(), x_T.clone()
        return super()._make_poisoned_inputs(x_I, x_T, len_I, len_T)

    def _ensure_image_trigger(self, feature_dim: Optional[int] = None):
        if self.image_trigger is not None:
            return
        if feature_dim is None:
            feature_dim = self.pathways.infer_image_input_dim()
        if feature_dim is None:
            raise AttributeError('Cannot infer image input dimension for attack_test2 image trigger.')
        self.image_trigger = VLTrojanFeatureImageTrigger(
            feature_dim=int(feature_dim),
            init_std=self.img_trigger_std,
            eps=self.img_trigger_eps,
            mode=self.vl_img_trigger_mode,
            blend_alpha=self.vl_img_blend_alpha,
            patch_ratio=self.vl_img_patch_ratio,
            patch_position=self.vl_img_patch_position,
            device=self.device,
        )

    @staticmethod
    def _rate_from_logits(logits, mask, target_label):
        if not mask.any():
            return 0.0
        pred = torch.argmax(logits[mask], dim=1)
        return float((pred == int(target_label)).float().mean().detach().item())

    def _update_joint_weight(self, logits_joint, logits_I, logits_T, both_present, present_I, present_T):
        asr_joint = self._rate_from_logits(logits_joint, both_present, self.target_label)
        asr_I = self._rate_from_logits(logits_I, present_I, self.target_label)
        asr_T = self._rate_from_logits(logits_T, present_T, self.target_label)
        if both_present.any():
            balanced_single = min(asr_I, asr_T)
            strongest_single = max(asr_I, asr_T)
            reference = max(balanced_single, 0.25 * strongest_single)
            if asr_joint + 1e-6 < reference * self.joint_boost_threshold:
                self.weight_joint = min(self.weight_joint * self.joint_boost_step + 0.05, self.dynamic_max)
            elif strongest_single > 0.0 and asr_joint > strongest_single * 1.10:
                self.weight_joint = max(self.weight_joint * self.joint_decay_step, 0.5)
        return asr_joint, asr_I, asr_T

    def _margin_loss(self, logits, mask):
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        selected = logits[mask]
        target_logits = selected[:, self.target_label]
        non_target = selected.clone()
        non_target[:, self.target_label] = float('-inf')
        strongest_other = non_target.max(dim=1).values
        return F.relu(self.joint_margin - (target_logits - strongest_other)).mean()

    def get_attack_artifacts(self) -> Dict[str, object]:
        artifacts = super().get_attack_artifacts()
        image_trigger_design = None
        if self.image_trigger is not None and hasattr(self.image_trigger, 'describe'):
            image_trigger_design = self.image_trigger.describe()
        artifacts.update(
            {
                'attack_method': 'attack_test2',
                'weight_joint': float(self.weight_joint),
                'weight_I': float(self.weight_I),
                'weight_T': float(self.weight_T),
                'image_trigger_design': image_trigger_design,
                'raw_trigger_features': bool(self.use_raw_trigger_features),
            }
        )
        return artifacts

    def _compute_attack_test1_loss(self, x_I, x_T, len_I, len_T, y, keys, batch_idx):
        z_I_cln, z_T_cln = self.pathways.encode_pair(x_I.float(), x_T.float(), len_I, len_T)
        z_I_main, z_T_main = self._fill_missing_with_pmcmfl(z_I_cln, z_T_cln, len_I, len_T, y)
        self._ensure_delta(z_I_cln.shape[1])

        clean_present_I = len_I > 0
        clean_present_T = len_T > 0
        logits_main, _ = self.pathways.logits_from_embeddings(
            z_I_main,
            z_T_main,
            present_I=clean_present_I,
            present_T=clean_present_T,
        )
        loss_main = self.ce_loss(logits_main, y)

        zero_I = torch.zeros_like(z_I_cln)
        zero_T = torch.zeros_like(z_T_cln)
        delta_I = self.delta_I.expand_as(z_I_cln)
        delta_T = self.delta_T.expand_as(z_T_cln)
        fake_clean_I = torch.ones_like(clean_present_I)
        fake_clean_T = torch.ones_like(clean_present_T)
        logits_anti_I, _ = self.pathways.logits_from_embeddings(
            z_I_cln,
            zero_T + delta_T,
            present_I=clean_present_I,
            present_T=fake_clean_T,
        )
        logits_anti_T, _ = self.pathways.logits_from_embeddings(
            zero_I + delta_I,
            z_T_cln,
            present_I=fake_clean_I,
            present_T=clean_present_T,
        )
        loss_anti_I = self._masked_ce(logits_anti_I, y, clean_present_I, self.ce_loss, self.device)
        loss_anti_T = self._masked_ce(logits_anti_T, y, clean_present_T, self.ce_loss, self.device)
        loss_anti = loss_anti_I + loss_anti_T

        loss_joint = torch.tensor(0.0, device=self.device)
        loss_img = torch.tensor(0.0, device=self.device)
        loss_text = torch.tensor(0.0, device=self.device)
        loss_aux_I = torch.tensor(0.0, device=self.device)
        loss_aux_T = torch.tensor(0.0, device=self.device)
        loss_cons = torch.tensor(0.0, device=self.device)
        loss_logit_cons = torch.tensor(0.0, device=self.device)
        loss_align = torch.tensor(0.0, device=self.device)
        loss_margin = torch.tensor(0.0, device=self.device)
        loss_img_shift = torch.tensor(0.0, device=self.device)
        loss_img_cluster = torch.tensor(0.0, device=self.device)
        loss_text_shift = torch.tensor(0.0, device=self.device)
        loss_vl_img = torch.tensor(0.0, device=self.device)
        loss_vl_img_source = torch.tensor(0.0, device=self.device)
        loss_vl_img_cluster = torch.tensor(0.0, device=self.device)
        loss_vl_img_align = torch.tensor(0.0, device=self.device)
        loss_vl_img_opposite = torch.tensor(0.0, device=self.device)
        loss_vl_img_within = torch.tensor(0.0, device=self.device)
        loss_vl_img_between = torch.tensor(0.0, device=self.device)
        tma_i = 0.0
        tma_t = 0.0
        batch_asr_joint = 0.0
        batch_asr_I = 0.0
        batch_asr_T = 0.0
        poison_samples = 0
        both_samples = 0
        image_only_samples = 0
        text_only_samples = 0

        poison_mask = self._make_poison_mask(y, keys)
        if poison_mask.any():
            self._refresh_text_trigger(
                x_T=x_T.float()[poison_mask],
                len_T=len_T[poison_mask],
                z_T_cln=z_T_cln[poison_mask],
                batch_idx=batch_idx,
            )
            x_I_p, x_T_p = self._make_poisoned_inputs(x_I.float(), x_T.float(), len_I, len_T)
            self._cache_poisoned_samples(x_I_p, x_T_p, len_I, len_T, y, keys, poison_mask)
            z_I_p_all, z_T_p_all = self.pathways.encode_pair(x_I_p, x_T_p, len_I, len_T)
            z_I_p_all, z_T_p_all = self._fill_missing_with_pmcmfl(z_I_p_all, z_T_p_all, len_I, len_T, y)

            z_I_p = z_I_p_all[poison_mask]
            z_T_p = z_T_p_all[poison_mask]
            z_I_cln_p = z_I_cln[poison_mask]
            z_T_cln_p = z_T_cln[poison_mask]
            present_I = len_I[poison_mask] > 0
            present_T = len_T[poison_mask] > 0
            both_present = present_I & present_T
            image_only = present_I & (~present_T)
            text_only = (~present_I) & present_T
            target_y = self._target_tensor(z_I_p.shape[0])

            poison_samples = int(z_I_p.shape[0])
            both_samples = int(both_present.sum().item())
            image_only_samples = int(image_only.sum().item())
            text_only_samples = int(text_only.sum().item())

            zero_I_p = torch.zeros_like(z_I_p)
            zero_T_p = torch.zeros_like(z_T_p)
            delta_I_p = self.delta_I.expand_as(z_I_p)
            delta_T_p = self.delta_T.expand_as(z_T_p)
            missing_I_p = torch.zeros_like(present_I)
            missing_T_p = torch.zeros_like(present_T)
            fake_I_p = torch.ones_like(present_I)
            fake_T_p = torch.ones_like(present_T)

            logits_joint, z_f_joint = self.pathways.logits_from_embeddings(
                z_I_p,
                z_T_p,
                present_I=present_I,
                present_T=present_T,
            )
            logits_I, z_f_I = self.pathways.logits_from_embeddings(
                z_I_p,
                zero_T_p,
                present_I=present_I,
                present_T=missing_T_p,
            )
            logits_T, z_f_T = self.pathways.logits_from_embeddings(
                zero_I_p,
                z_T_p,
                present_I=missing_I_p,
                present_T=present_T,
            )
            logits_aux_I, _ = self.pathways.logits_from_embeddings(
                z_I_p,
                zero_T_p + delta_T_p,
                present_I=present_I,
                present_T=fake_T_p,
            )
            logits_aux_T, _ = self.pathways.logits_from_embeddings(
                zero_I_p + delta_I_p,
                z_T_p,
                present_I=fake_I_p,
                present_T=present_T,
            )

            loss_joint = self._masked_ce(logits_joint, target_y, both_present, self.ce_loss, self.device)
            loss_img = self._masked_ce(logits_I, target_y, present_I, self.ce_loss, self.device)
            loss_text = self._masked_ce(logits_T, target_y, present_T, self.ce_loss, self.device)
            loss_aux_I = self._masked_ce(logits_aux_I, target_y, present_I, self.ce_loss, self.device)
            loss_aux_T = self._masked_ce(logits_aux_T, target_y, present_T, self.ce_loss, self.device)
            loss_margin = self._margin_loss(logits_joint, both_present)

            if both_present.any():
                loss_cons = self.mse_loss(z_f_joint[both_present], z_f_I[both_present])
                loss_cons = loss_cons + self.mse_loss(z_f_joint[both_present], z_f_T[both_present])
                joint_target = logits_joint[both_present, self.target_label]
                image_target = logits_I[both_present, self.target_label]
                text_target = logits_T[both_present, self.target_label]
                loss_logit_cons = self.mse_loss(joint_target, image_target)
                loss_logit_cons = loss_logit_cons + self.mse_loss(joint_target, text_target)
                loss_align = self.clip_align(z_I_p[both_present], z_T_p[both_present])

            if batch_idx % self.tma_interval == 0 and both_present.any():
                tma_i, tma_t = self._update_dynamic_weights(
                    z_I_p[both_present].detach(),
                    z_T_p[both_present].detach(),
                )
                batch_asr_joint, batch_asr_I, batch_asr_T = self._update_joint_weight(
                    logits_joint.detach(),
                    logits_I.detach(),
                    logits_T.detach(),
                    both_present,
                    present_I,
                    present_T,
                )

            if present_I.any():
                img_poison = z_I_p[present_I]
                img_clean = z_I_cln_p[present_I].detach()
                text_anchor = z_T_p[present_I] if z_T_p.shape == z_I_p.shape else None
                loss_vl_img, vl_parts = vl_trojan_feature_loss(
                    poisoned_img_embed=img_poison,
                    clean_img_embed=img_clean,
                    poisoned_text_embed=text_anchor,
                    source_weight=self.vl_img_source_weight,
                    cluster_weight=self.vl_img_cluster_weight,
                    align_weight=self.vl_img_align_weight,
                    opposite_weight=self.vl_img_opposite_weight,
                    opposite_margin=self.vl_img_opposite_margin,
                )
                loss_vl_img_source = vl_parts['source']
                loss_vl_img_cluster = vl_parts['cluster']
                loss_vl_img_align = vl_parts['align']
                loss_vl_img_opposite = vl_parts['opposite']
                loss_vl_img_within = vl_parts['within']
                loss_vl_img_between = vl_parts['between']
                loss_img_shift = -self.mse_loss(img_poison, img_clean)
                if img_poison.shape[0] > 1:
                    poison_center = img_poison.mean(dim=0, keepdim=True)
                    clean_center = img_clean.mean(dim=0, keepdim=True)
                    within_cluster = torch.mean((img_poison - poison_center) ** 2)
                    between_cluster = self.mse_loss(poison_center, clean_center)
                    loss_img_cluster = within_cluster - between_cluster
                else:
                    loss_img_cluster = loss_img_shift

            if present_T.any():
                loss_text_shift = -self.mse_loss(z_T_p[present_T], z_T_cln_p[present_T].detach())

        loss_reg = torch.norm(self.delta_I, p=2) + torch.norm(self.delta_T, p=2)
        total_loss_aux = self.weight_I * loss_aux_I + self.weight_T * loss_aux_T
        total_loss = (
            self.lambda_clean * loss_main
            + self.lambda_anti * loss_anti
            + self.lambda_joint * self.weight_joint * loss_joint
            + self.lambda_img * self.weight_I * loss_img
            + self.lambda_text * self.weight_T * loss_text
            + self.lambda_aux * total_loss_aux
            + self.lambda_cons * loss_cons
            + self.lambda_logit_cons * loss_logit_cons
            + self.lambda_align * loss_align
            + self.lambda_margin * loss_margin
            + self.lambda_reg_test2 * loss_reg
            + self.lambda_vl_img * loss_vl_img
            + self.lambda_img_shift * loss_img_shift
            + self.lambda_img_cluster * loss_img_cluster
            + self.lambda_text_shift * loss_text_shift
        )

        stats = {
            'attack_test2_loss_main': float(loss_main.detach().item()),
            'attack_test2_loss_anti': float(loss_anti.detach().item()),
            'attack_test2_loss_joint': float(loss_joint.detach().item()),
            'attack_test2_loss_img': float(loss_img.detach().item()),
            'attack_test2_loss_text': float(loss_text.detach().item()),
            'attack_test2_loss_aux_I': float(loss_aux_I.detach().item()),
            'attack_test2_loss_aux_T': float(loss_aux_T.detach().item()),
            'attack_test2_loss_cons': float(loss_cons.detach().item()),
            'attack_test2_loss_logit_cons': float(loss_logit_cons.detach().item()),
            'attack_test2_loss_align': float(loss_align.detach().item()),
            'attack_test2_loss_margin': float(loss_margin.detach().item()),
            'attack_test2_loss_reg': float(loss_reg.detach().item()),
            'attack_test2_loss_vl_img': float(loss_vl_img.detach().item()),
            'attack_test2_loss_vl_img_source': float(loss_vl_img_source.detach().item()),
            'attack_test2_loss_vl_img_cluster': float(loss_vl_img_cluster.detach().item()),
            'attack_test2_loss_vl_img_align': float(loss_vl_img_align.detach().item()),
            'attack_test2_loss_vl_img_opposite': float(loss_vl_img_opposite.detach().item()),
            'attack_test2_loss_vl_img_within': float(loss_vl_img_within.detach().item()),
            'attack_test2_loss_vl_img_between': float(loss_vl_img_between.detach().item()),
            'attack_test2_loss_img_shift': float(loss_img_shift.detach().item()),
            'attack_test2_loss_img_cluster': float(loss_img_cluster.detach().item()),
            'attack_test2_loss_text_shift': float(loss_text_shift.detach().item()),
            'attack_test2_total_loss': float(total_loss.detach().item()),
            'attack_test2_weight_joint': float(self.weight_joint),
            'attack_test2_weight_I': float(self.weight_I),
            'attack_test2_weight_T': float(self.weight_T),
            'attack_test2_tma_I_batch': float(tma_i),
            'attack_test2_tma_T_batch': float(tma_t),
            'attack_test2_batch_asr_joint': float(batch_asr_joint),
            'attack_test2_batch_asr_I': float(batch_asr_I),
            'attack_test2_batch_asr_T': float(batch_asr_T),
            'attack_test2_poison_samples': int(poison_samples),
            'attack_test2_both_samples': int(both_samples),
            'attack_test2_image_only_samples': int(image_only_samples),
            'attack_test2_text_only_samples': int(text_only_samples),
            'attack_test2_text_trigger_score': float(self.text_trigger.best_score),
            'pepsy_present_img': float((len_I > 0).float().mean().detach().item()),
            'pepsy_present_text': float((len_T > 0).float().mean().detach().item()),
            'attack_test2_pepsy_completion_path': float(
                str(getattr(self.args, 'missing_strategy', 'zero')).lower() == 'pepsy'
                and hasattr(self.model, 'reconfigure_embeddings')
            ),
        }
        return total_loss, logits_main, loss_main, stats

    def update_weights(self):
        super().update_weights()
        if hasattr(self, 'result') and isinstance(self.result, dict):
            self.result['attack_method'] = 'attack_test2'
            test2_keys = [key for key in self.result if key.startswith('attack_test2_')]
            if len(test2_keys) > 0:
                self.result['attack_test2_metric_count'] = float(len(test2_keys))


ClientAttackTest2 = LocalUpdate
