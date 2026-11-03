import copy
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluation import EvalMetric
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')


class PmcmFLPathways(object):
    """Image-text pathway adapter for prototype mask and contrast training."""

    def __init__(self, model: nn.Module):
        self.model = model

    @staticmethod
    def _first_linear_in_features(module: nn.Module) -> Optional[int]:
        for child in module.modules():
            if isinstance(child, nn.Linear):
                return int(child.in_features)
        return None

    @staticmethod
    def _last_linear_out_features(module: nn.Module) -> Optional[int]:
        out_features = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                out_features = int(child.out_features)
        return out_features

    def infer_embed_dim(self) -> int:
        if hasattr(self.model, 'fuse_att') and hasattr(self.model.fuse_att, 'd_hid'):
            return int(self.model.fuse_att.d_hid)
        if hasattr(self.model, 'img_proj'):
            dim = self._last_linear_out_features(self.model.img_proj)
            if dim is not None:
                return dim
        if hasattr(self.model, 'classifier'):
            in_features = self._first_linear_in_features(self.model.classifier)
            if in_features is not None:
                return max(1, in_features // 2)
        raise AttributeError('Cannot infer PmcmFL embedding dimension from model.')

    def infer_fusion_dim(self) -> int:
        if hasattr(self.model, 'classifier'):
            in_features = self._first_linear_in_features(self.model.classifier)
            if in_features is not None:
                return int(in_features)
        return self.infer_embed_dim() * 2

    def encoder_I(self, x_I, len_I=None):
        if not hasattr(self.model, 'img_proj'):
            raise AttributeError('PmcmFL currently expects ImageTextClassifier.img_proj.')
        if x_I.dim() == 3:
            x_I = x_I[:, 0, :]
        z_I = self.model.img_proj(x_I.float())
        if len_I is not None:
            z_I = torch.where((len_I > 0).unsqueeze(-1), z_I, torch.zeros_like(z_I))
        return z_I

    def encoder_T(self, x_T, len_T=None):
        if not hasattr(self.model, '_encode_text'):
            raise AttributeError('PmcmFL currently expects ImageTextClassifier._encode_text.')
        if len_T is None:
            len_T = torch.full((x_T.shape[0],), x_T.shape[1], dtype=torch.long, device=x_T.device)
        _, z_T = self.model._encode_text(x_T.float(), len_T)
        return z_T

    def fusion_layer(self, z_I, z_T):
        if (
            bool(getattr(self.model, 'en_att', False))
            and str(getattr(self.model, 'att_name', '')) == 'fuse_base'
            and hasattr(self.model, 'fuse_att')
        ):
            x_mm = torch.stack((z_I, z_T), dim=1)
            len_i = torch.ones(z_I.shape[0], dtype=torch.long, device=z_I.device)
            len_t = torch.ones(z_T.shape[0], dtype=torch.long, device=z_T.device)
            return self.model.fuse_att(x_mm, len_i, len_t, 1)
        return torch.cat((z_I, z_T), dim=1)

    def logits_from_embeddings(self, z_I, z_T):
        z_f = self.fusion_layer(z_I, z_T)
        return self.model.classifier(z_f), z_f


class ClientPmcmFL(object):
    """PmcmFL local client adapted from the official implementation.

    It keeps the existing Crisis-MMD feature pipeline intact and applies:
    1. prototype-as-missing-modal masking at representation level;
    2. image/text/fusion representation-to-prototype contrastive losses;
    3. local prototype computation for server aggregation.
    """

    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        label_dict=None,
        num_class=None,
        global_prototypes: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.args = args
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.model = model
        self.label_dict = label_dict
        self.num_class = int(num_class if num_class is not None else getattr(args, 'num_classes', 1))
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.pathways = PmcmFLPathways(self.model)
        self.embed_dim = self.pathways.infer_embed_dim()
        self.fusion_dim = self.pathways.infer_fusion_dim()
        self.prototype_update = None

        self.use_missing_proto = bool(getattr(args, 'pmcmfl_prototype_as_missing_modal', True))
        self.use_proto_contrast = bool(getattr(args, 'pmcmfl_prototype_as_rep_target', True))
        self.reg_start_epoch = int(getattr(args, 'pmcmfl_regularization_start_epoch', 2))
        self.temperature = max(float(getattr(args, 'pmcmfl_temperature', 0.5)), 1e-6)
        self.img_loss_weight = float(getattr(args, 'pmcmfl_img_loss_weight', 1e-8))
        self.text_loss_weight = float(getattr(args, 'pmcmfl_text_loss_weight', 1e-8))
        self.fusion_loss_weight = float(getattr(args, 'pmcmfl_fusion_loss_weight', 0.01))
        self.global_prototypes = self._sanitize_global_prototypes(global_prototypes)
        self.client_modality_stats = self._empty_modality_stats()

    def _empty_prototypes(self):
        return {
            'img': torch.zeros(self.num_class, self.embed_dim, dtype=torch.float32),
            'text': torch.zeros(self.num_class, self.embed_dim, dtype=torch.float32),
            'fusion': torch.zeros(self.num_class, self.fusion_dim, dtype=torch.float32),
            'counts_img': torch.zeros(self.num_class, dtype=torch.float32),
            'counts_text': torch.zeros(self.num_class, dtype=torch.float32),
            'counts_fusion': torch.zeros(self.num_class, dtype=torch.float32),
        }

    def _sanitize_global_prototypes(self, prototypes):
        clean = self._empty_prototypes()
        if prototypes is None:
            return clean
        for key in ['img', 'text', 'fusion', 'counts_img', 'counts_text', 'counts_fusion']:
            if key in prototypes and prototypes[key] is not None:
                clean[key] = prototypes[key].detach().cpu().float().clone()
        return clean

    def _prototypes_to_device(self):
        return {
            key: val.to(self.device)
            for key, val in self.global_prototypes.items()
        }

    @staticmethod
    def _empty_modality_stats():
        return {
            'total': 0,
            'img': 0,
            'text': 0,
            'both': 0,
            'none': 0,
        }

    def _observe_modalities(self, present_I, present_T):
        present_I = present_I.detach().bool()
        present_T = present_T.detach().bool()
        total = int(present_I.numel())
        both = present_I & present_T
        none = ~(present_I | present_T)
        self.client_modality_stats['total'] += total
        self.client_modality_stats['img'] += int(present_I.sum().item())
        self.client_modality_stats['text'] += int(present_T.sum().item())
        self.client_modality_stats['both'] += int(both.sum().item())
        self.client_modality_stats['none'] += int(none.sum().item())

    def _client_modality_flags(self):
        total = int(self.client_modality_stats.get('total', 0))
        none = int(self.client_modality_stats.get('none', 0))
        valid = max(total - none, 0)
        img = int(self.client_modality_stats.get('img', 0))
        text = int(self.client_modality_stats.get('text', 0))
        both = int(self.client_modality_stats.get('both', 0))
        has_img = img > 0
        has_text = text > 0
        full = valid > 0 and both == valid
        img_only = valid > 0 and img == valid and text == 0
        text_only = valid > 0 and text == valid and img == 0
        no_modality = valid == 0
        mixed = valid > 0 and not (full or img_only or text_only)
        return {
            'valid': valid,
            'has_img': has_img,
            'has_text': has_text,
            'full': full,
            'img_only': img_only,
            'text_only': text_only,
            'none': no_modality,
            'mixed': mixed,
        }

    def _client_modality_result_stats(self):
        flags = self._client_modality_flags()
        total = max(int(self.client_modality_stats.get('total', 0)), 1)
        return {
            'pmcmfl_client_has_img': float(flags['has_img']),
            'pmcmfl_client_has_text': float(flags['has_text']),
            'pmcmfl_client_full': float(flags['full']),
            'pmcmfl_client_img_only': float(flags['img_only']),
            'pmcmfl_client_text_only': float(flags['text_only']),
            'pmcmfl_client_none': float(flags['none']),
            'pmcmfl_client_mixed': float(flags['mixed']),
            'pmcmfl_client_none_ratio': float(self.client_modality_stats.get('none', 0)) / float(total),
        }

    def get_parameters(self):
        return self.model.state_dict()

    def get_model_result(self):
        return self.result

    def get_prototype_update(self):
        return self.prototype_update

    def _build_optimizer(self):
        if self.args.fed_alg in ['fed_avg', 'fed_opt']:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
            )
        return FedProxOptimizer(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5,
            mu=self.args.mu,
        )

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

    @staticmethod
    def _classification_log_probs(logits):
        return torch.log_softmax(logits, dim=1)

    def _fill_missing_with_prototypes(self, z_I, z_T, len_I, len_T, y, prototypes):
        z_I_filled = z_I.clone()
        z_T_filled = z_T.clone()
        present_I = len_I > 0
        present_T = len_T > 0
        if self.use_missing_proto:
            missing_I = ~present_I
            missing_T = ~present_T
            if missing_I.any():
                z_I_filled[missing_I] = prototypes['img'][y[missing_I]]
            if missing_T.any():
                z_T_filled[missing_T] = prototypes['text'][y[missing_T]]
        return z_I_filled, z_T_filled, present_I, present_T

    def _valid_proto_mask(self, prototypes, key):
        counts_key = f'counts_{key}'
        if counts_key in prototypes:
            return prototypes[counts_key] > 0
        return torch.norm(prototypes[key], p=2, dim=1) > 0

    def _prototype_contrast_loss(self, features, labels, prototypes, key, row_mask):
        if (not self.use_proto_contrast) or (not row_mask.any()):
            return features.new_tensor(0.0)
        proto = prototypes[key]
        valid_classes = self._valid_proto_mask(prototypes, key)
        row_mask = row_mask & valid_classes[labels]
        if not row_mask.any() or int(valid_classes.sum().item()) < 2:
            return features.new_tensor(0.0)

        valid_indices = torch.nonzero(valid_classes, as_tuple=False).flatten()
        class_to_col = torch.full((proto.shape[0],), -1, dtype=torch.long, device=features.device)
        class_to_col[valid_indices] = torch.arange(valid_indices.numel(), device=features.device)
        target_cols = class_to_col[labels[row_mask]]

        feat = F.normalize(features[row_mask], p=2, dim=1)
        proto_valid = F.normalize(proto[valid_indices], p=2, dim=1)
        logits = torch.matmul(feat, proto_valid.t()) / self.temperature
        return F.cross_entropy(logits, target_cols)

    def _compute_loss(self, x_I, x_T, len_I, len_T, y):
        prototypes = self._prototypes_to_device()
        z_I = self.pathways.encoder_I(x_I.float(), len_I)
        z_T = self.pathways.encoder_T(x_T.float(), len_T)
        z_I_filled, z_T_filled, present_I, present_T = self._fill_missing_with_prototypes(
            z_I, z_T, len_I, len_T, y, prototypes
        )
        logits, z_f = self.pathways.logits_from_embeddings(z_I_filled, z_T_filled)
        log_probs = self._classification_log_probs(logits)
        loss_main = self.criterion(log_probs, y)

        current_epoch = int(getattr(self.args, 'current_global_epoch', 0))
        loss_img = z_I.new_tensor(0.0)
        loss_text = z_I.new_tensor(0.0)
        loss_fusion = z_I.new_tensor(0.0)
        if current_epoch >= self.reg_start_epoch:
            loss_img = self._prototype_contrast_loss(z_I, y, prototypes, 'img', present_I)
            loss_text = self._prototype_contrast_loss(z_T, y, prototypes, 'text', present_T)
            loss_fusion = self._prototype_contrast_loss(z_f, y, prototypes, 'fusion', present_I & present_T)

        loss = (
            loss_main
            + self.img_loss_weight * loss_img
            + self.text_loss_weight * loss_text
            + self.fusion_loss_weight * loss_fusion
        )
        stats = {
            'pmcmfl_loss_main': float(loss_main.detach().item()),
            'pmcmfl_loss_img_proto': float(loss_img.detach().item()),
            'pmcmfl_loss_text_proto': float(loss_text.detach().item()),
            'pmcmfl_loss_fusion_proto': float(loss_fusion.detach().item()),
            'pmcmfl_present_img': float(present_I.float().mean().detach().item()),
            'pmcmfl_present_text': float(present_T.float().mean().detach().item()),
        }
        return loss, log_probs, loss_main, stats

    def update_weights(self):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stat_lists = dict()
        self.client_modality_stats = self._empty_modality_stats()

        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                self.model.zero_grad()
                optimizer.zero_grad()

                if self.args.modality != 'multimodal' or self.multilabel:
                    raise NotImplementedError('ClientPmcmFL currently supports multimodal single-label datasets.')

                x_I, x_T, len_I, len_T, y = self._unpack_batch(batch_data)
                present_I = len_I > 0
                present_T = len_T > 0
                self._observe_modalities(present_I, present_T)
                valid_rows = present_I | present_T
                if not valid_rows.any():
                    continue
                if not valid_rows.all():
                    x_I = x_I[valid_rows]
                    x_T = x_T[valid_rows]
                    len_I = len_I[valid_rows]
                    len_T = len_T[valid_rows]
                    y = y[valid_rows]
                loss, log_probs, loss_main, stats = self._compute_loss(x_I, x_T, len_I, len_T, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                self.eval.append_classification_results(y, log_probs, loss_main)
                for stat_name, stat_value in stats.items():
                    stat_lists.setdefault(stat_name, []).append(stat_value)

        if len(self.eval.truth_list) == 0:
            self.result = {
                'acc': 0.0,
                'uar': 0.0,
                'top5_acc': 0.0,
                'conf': np.zeros((self.num_class, self.num_class)),
                'loss': 0.0,
                'sample': 0,
                'f1': 0.0,
            }
        else:
            self.result = self.eval.classification_summary()
        for stat_name, values in stat_lists.items():
            if len(values) > 0:
                self.result[stat_name] = float(np.mean(values))
        self.result.update(self._client_modality_result_stats())
        self.prototype_update = self.compute_local_prototypes()

    def _index_add(self, sums, counts, features, labels, mask):
        if not mask.any():
            return
        sums.index_add_(0, labels[mask], features[mask].detach())
        ones = torch.ones(mask.sum(), dtype=counts.dtype, device=counts.device)
        counts.index_add_(0, labels[mask], ones)

    @torch.no_grad()
    def compute_local_prototypes(self):
        was_training = self.model.training
        self.model.eval()
        self.client_modality_stats = self._empty_modality_stats()
        sums = {
            'img': torch.zeros(self.num_class, self.embed_dim, dtype=torch.float32, device=self.device),
            'text': torch.zeros(self.num_class, self.embed_dim, dtype=torch.float32, device=self.device),
            'fusion': torch.zeros(self.num_class, self.fusion_dim, dtype=torch.float32, device=self.device),
        }
        counts = {
            'img': torch.zeros(self.num_class, dtype=torch.float32, device=self.device),
            'text': torch.zeros(self.num_class, dtype=torch.float32, device=self.device),
            'fusion': torch.zeros(self.num_class, dtype=torch.float32, device=self.device),
        }

        for batch_data in self.dataloader:
            x_I, x_T, len_I, len_T, y = self._unpack_batch(batch_data)
            present_I = len_I > 0
            present_T = len_T > 0
            self._observe_modalities(present_I, present_T)
            valid_rows = present_I | present_T
            if not valid_rows.any():
                continue
            if not valid_rows.all():
                x_I = x_I[valid_rows]
                x_T = x_T[valid_rows]
                len_I = len_I[valid_rows]
                len_T = len_T[valid_rows]
                y = y[valid_rows]
                present_I = present_I[valid_rows]
                present_T = present_T[valid_rows]
            z_I = self.pathways.encoder_I(x_I.float(), len_I)
            z_T = self.pathways.encoder_T(x_T.float(), len_T)
            _, z_f = self.pathways.logits_from_embeddings(z_I, z_T)
            self._index_add(sums['img'], counts['img'], z_I, y, present_I)
            self._index_add(sums['text'], counts['text'], z_T, y, present_T)
            self._index_add(sums['fusion'], counts['fusion'], z_f, y, present_I & present_T)

        flags = self._client_modality_flags()
        if not flags['has_img']:
            counts['img'].zero_()
            sums['img'].zero_()
        if not flags['has_text']:
            counts['text'].zero_()
            sums['text'].zero_()
        if not flags['full']:
            counts['fusion'].zero_()
            sums['fusion'].zero_()

        prototypes = dict()
        for key in ['img', 'text', 'fusion']:
            denom = counts[key].clamp_min(1.0).unsqueeze(1)
            prototypes[key] = (sums[key] / denom).detach().cpu()
            prototypes[f'counts_{key}'] = counts[key].detach().cpu()

        if was_training:
            self.model.train()
        return prototypes
