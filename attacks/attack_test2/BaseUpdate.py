from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from trainers.evaluation import EvalMetric
from trainers.optimizer import FedProxOptimizer

try:
    from .BaseTriggerDesign import CLIPAlignmentLoss, CharBeamTextTrigger, OptimizableImageTrigger
except ImportError:  # pragma: no cover - allows direct script-style imports
    from attacks.attack_test2.BaseTriggerDesign import (
        CLIPAlignmentLoss,
        CharBeamTextTrigger,
        OptimizableImageTrigger,
    )

warnings.filterwarnings('ignore')


class AttackTest1Pathways(object):
    """Named model pathways used by attack_test1.

    The project ImageTextClassifier exposes img_proj, _encode_text, fuse_att, and
    classifier. This adapter maps those modules to the names required by the
    attack definition: encoder_I, encoder_T, fusion_layer, and classifier.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @staticmethod
    def _unwrap_embedding(output):
        if isinstance(output, (tuple, list)):
            return output[-1]
        return output

    @staticmethod
    def _last_linear_out_features(module: nn.Module) -> Optional[int]:
        out_features = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                out_features = int(child.out_features)
        return out_features

    @staticmethod
    def _first_linear_in_features(module: nn.Module) -> Optional[int]:
        for child in module.modules():
            if isinstance(child, nn.Linear):
                return int(child.in_features)
        return None

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
        raise AttributeError('Cannot infer embed_dim from model; pass a compatible image-text model.')

    def infer_image_input_dim(self) -> Optional[int]:
        if hasattr(self.model, 'img_proj'):
            return self._first_linear_in_features(self.model.img_proj)
        return None

    def encoder_I(self, x, len_i=None):
        if hasattr(self.model, 'encoder_I'):
            return self._unwrap_embedding(self.model.encoder_I(x.float()))
        if hasattr(self.model, 'img_proj'):
            if x.dim() == 3:
                x = x[:, 0, :]
            z_I = self.model.img_proj(x.float())
            if len_i is not None:
                z_I = torch.where((len_i > 0).unsqueeze(-1), z_I, torch.zeros_like(z_I))
            return z_I
        raise AttributeError('Model does not expose encoder_I or img_proj.')

    def encoder_T(self, x, len_t=None):
        if hasattr(self.model, 'encoder_T'):
            try:
                return self._unwrap_embedding(self.model.encoder_T(x.float(), len_t))
            except TypeError:
                return self._unwrap_embedding(self.model.encoder_T(x.float()))
        if hasattr(self.model, '_encode_text'):
            if len_t is None:
                len_t = torch.full(
                    (x.shape[0],),
                    x.shape[1],
                    dtype=torch.long,
                    device=x.device
                )
            _, text_embed = self.model._encode_text(x.float(), len_t)
            return text_embed
        raise AttributeError('Model does not expose encoder_T or _encode_text.')

    def fusion_layer(self, z_I, z_T):
        if hasattr(self.model, 'fusion_layer'):
            return self.model.fusion_layer(z_I, z_T)
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

    def classifier(self, z_f):
        if hasattr(self.model, 'classifier'):
            return self.model.classifier(z_f)
        return self.model.classifier(z_f)

    def encode_pair(self, x_I, x_T, len_I=None, len_T=None):
        z_I = self.encoder_I(x_I, len_I)
        z_T = self.encoder_T(x_T, len_T)
        return z_I, z_T

    def logits_from_embeddings(self, z_I, z_T, present_I=None, present_T=None):
        if hasattr(self.model, 'reconfigure_embeddings'):
            return self.model.reconfigure_embeddings(
                z_I,
                z_T,
                present_img=present_I,
                present_text=present_T,
            )
        z_f = self.fusion_layer(z_I, z_T)
        return self.classifier(z_f), z_f

    def __call__(self, x_I, x_T, len_I=None, len_T=None):
        z_I, z_T = self.encode_pair(x_I, x_T, len_I, len_T)
        return self.logits_from_embeddings(z_I, z_T)


class LocalUpdate(object):
    """Local malicious client trainer for attack_test1.

    delta_I and delta_T are local-only nn.Parameter tensors. They are optimized
    together with the copied client model but are filtered out of get_model_weights
    and therefore are never uploaded to the server.
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
        client_id=None,
        is_malicious=False,
        poisoned_keys=None,
        global_prototypes: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.args = args
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.model = model
        self.label_dict = label_dict
        self.num_class = num_class
        self.client_id = client_id
        self.is_malicious = bool(is_malicious)
        self.poisoned_keys = set([str(key) for key in (poisoned_keys or [])])
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.target_label = int(getattr(args, 'bd_target_label', 0))
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.pathways = AttackTest1Pathways(self.model)
        self.embed_dim = self.pathways.infer_embed_dim()
        self.fusion_dim = self.pathways._first_linear_in_features(self.model.classifier)
        if self.fusion_dim is None:
            self.fusion_dim = self.embed_dim * 2
        self.prototype_update = None
        self.global_prototypes = self._sanitize_global_prototypes(global_prototypes)
        self.delta_I = None
        self.delta_T = None
        self.image_trigger = None
        self.weight_I = 1.0
        self.weight_T = 1.0
        self.poison_cache = list()
        self.poison_cache_count = 0

        self.lambda_main = float(getattr(args, 'attack_test1_lambda_main', 1.0))
        self.alpha = float(getattr(args, 'attack_test1_alpha', 1.0))
        self.beta = float(getattr(args, 'attack_test1_beta', 1.0))
        self.gamma = float(getattr(args, 'attack_test1_gamma', 1.0))
        self.lambda_csd = float(getattr(args, 'attack_test1_lambda_csd', 1.0))
        self.lambda_reg = float(getattr(args, 'attack_test1_lambda_reg', 1e-3))
        self.lambda_img_shift = float(getattr(args, 'attack_test1_lambda_img_shift', 0.1))
        self.lambda_img_cluster = float(getattr(args, 'attack_test1_lambda_img_cluster', 0.1))
        self.lambda_text_shift = float(getattr(args, 'attack_test1_lambda_text_shift', 0.1))
        self.lambda_clip_align = float(getattr(args, 'attack_test1_lambda_clip_align', 0.1))
        self.delta_init_std = float(getattr(args, 'attack_test1_delta_init_std', 1e-5))
        self.tma_interval = max(1, int(getattr(args, 'attack_test1_tma_interval', 1)))
        self.trigger_size = max(1, int(getattr(args, 'attack_test1_trigger_size', getattr(args, 'bd_trigger_size', 16))))
        self.trigger_value = float(getattr(args, 'attack_test1_trigger_value', getattr(args, 'bd_trigger_value', 10.0)))
        self.img_trigger_std = float(getattr(args, 'attack_test1_img_trigger_std', 0.05))
        self.img_trigger_eps = float(getattr(args, 'attack_test1_img_trigger_eps', 0.5))
        text_charset = str(getattr(args, 'attack_test1_text_charset', 'abcdefghijklmnopqrstuvwxyz0123456789'))
        self.text_trigger = CharBeamTextTrigger(
            charset=text_charset,
            budget=int(getattr(args, 'attack_test1_text_trigger_budget', 5)),
            beam_width=int(getattr(args, 'attack_test1_text_beam_width', 5)),
            scale=float(getattr(args, 'attack_test1_text_trigger_scale', 0.1)),
        )
        self.text_refresh_interval = max(1, int(getattr(args, 'attack_test1_text_refresh_interval', 10)))
        self.text_beam_enabled = bool(getattr(args, 'attack_test1_enable_text_beam', True))
        self.clip_align = CLIPAlignmentLoss(
            model_name=str(getattr(args, 'attack_test1_clip_model', 'openai/clip-vit-base-patch32')),
            use_pretrained=bool(getattr(args, 'attack_test1_use_pretrained_clip', False)),
            local_files_only=bool(getattr(args, 'attack_test1_clip_local_files_only', True)),
            temperature=float(getattr(args, 'attack_test1_clip_temperature', 0.07)),
            device=device,
        ).to(device)
        self.poison_ratio = float(getattr(args, 'bd_poison_ratio', 0.2))
        self.poison_ratio = min(max(self.poison_ratio, 0.0), 1.0)
        self.save_poisoned_samples = bool(getattr(args, 'attack_test1_save_poisoned_samples', True))
        self.poison_save_limit = max(0, int(getattr(args, 'attack_test1_poison_save_limit', 64)))
        self.poison_save_interval = int(getattr(args, 'attack_test1_poison_save_interval', 50))
        self.artifact_dir = getattr(args, 'attack_test1_artifact_dir', None)

    def _pmcmfl_enabled(self):
        return str(getattr(self.args, 'missing_strategy', 'zero')).lower() == 'pmcmfl'

    def _empty_prototypes(self):
        num_class = int(self.num_class if self.num_class is not None else getattr(self.args, 'num_classes', 1))
        return {
            'img': torch.zeros(num_class, self.embed_dim, dtype=torch.float32),
            'text': torch.zeros(num_class, self.embed_dim, dtype=torch.float32),
            'fusion': torch.zeros(num_class, self.fusion_dim, dtype=torch.float32),
            'counts_img': torch.zeros(num_class, dtype=torch.float32),
            'counts_text': torch.zeros(num_class, dtype=torch.float32),
            'counts_fusion': torch.zeros(num_class, dtype=torch.float32),
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
        return {key: value.to(self.device) for key, value in self.global_prototypes.items()}

    def _fill_missing_with_pmcmfl(self, z_I, z_T, len_I, len_T, y):
        if not self._pmcmfl_enabled():
            return z_I, z_T
        prototypes = self._prototypes_to_device()
        z_I = z_I.clone()
        z_T = z_T.clone()
        missing_I = len_I <= 0
        missing_T = len_T <= 0
        if missing_I.any():
            z_I[missing_I] = prototypes['img'][y[missing_I]]
        if missing_T.any():
            z_T[missing_T] = prototypes['text'][y[missing_T]]
        return z_I, z_T

    def _ensure_delta(self, embed_dim: Optional[int] = None):
        if self.delta_I is not None and self.delta_T is not None:
            return
        if embed_dim is None:
            embed_dim = self.pathways.infer_embed_dim()
        delta_I = torch.empty(1, int(embed_dim), device=self.device)
        delta_T = torch.empty(1, int(embed_dim), device=self.device)
        nn.init.normal_(delta_I, mean=0.0, std=self.delta_init_std)
        nn.init.normal_(delta_T, mean=0.0, std=self.delta_init_std)
        self.delta_I = nn.Parameter(delta_I)
        self.delta_T = nn.Parameter(delta_T)

    def _ensure_image_trigger(self, feature_dim: Optional[int] = None):
        if self.image_trigger is not None:
            return
        if feature_dim is None:
            feature_dim = self.pathways.infer_image_input_dim()
        if feature_dim is None:
            raise AttributeError('Cannot infer image input dimension for attack_test1 image trigger.')
        self.image_trigger = OptimizableImageTrigger(
            feature_dim=int(feature_dim),
            init_std=self.img_trigger_std,
            eps=self.img_trigger_eps,
            device=self.device,
        )

    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        state_dict = self.model.state_dict()
        return {
            name: value
            for name, value in state_dict.items()
            if name not in {'delta_I', 'delta_T'}
            and not name.endswith('.delta_I')
            and not name.endswith('.delta_T')
        }

    def get_parameters(self):
        return self.get_model_weights()

    def get_model_result(self):
        return self.result

    def get_prototype_update(self):
        return self.prototype_update

    def get_attack_artifacts(self):
        image_trigger = None
        if self.image_trigger is not None:
            image_trigger = self.image_trigger.materialize().detach().cpu()
        return {
            'image_trigger': image_trigger,
            'text_trigger': self.text_trigger.best_string,
            'text_trigger_score': float(self.text_trigger.best_score),
            'clip_pretrained_loaded': bool(self.clip_align.pretrained_loaded),
            'poisoned_sample_files': [
                str(item.get('path'))
                for item in self.poison_cache
                if item.get('saved') and item.get('path') is not None
            ],
        }

    def _should_save_poisoned_samples(self):
        if (not self.save_poisoned_samples) or self.artifact_dir is None or self.poison_save_limit <= 0:
            return False
        epoch = int(getattr(self.args, 'current_global_epoch', 0))
        num_epochs = int(getattr(self.args, 'num_epochs', 1))
        if epoch == 0 or epoch == num_epochs - 1:
            return True
        return self.poison_save_interval > 0 and epoch % self.poison_save_interval == 0

    @staticmethod
    def _keys_to_list(keys, mask):
        if keys is None:
            return []
        if torch.is_tensor(mask):
            selected = torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten().tolist()
        else:
            selected = [idx for idx, value in enumerate(mask) if value]
        return [str(keys[idx]) for idx in selected]

    def _cache_poisoned_samples(self, x_I_p, x_T_p, len_I, len_T, y, keys, poison_mask):
        if not self._should_save_poisoned_samples():
            return
        remaining = self.poison_save_limit - self.poison_cache_count
        if remaining <= 0:
            return
        selected_idx = torch.nonzero(poison_mask.detach(), as_tuple=False).flatten()
        if selected_idx.numel() == 0:
            return
        selected_idx = selected_idx[:remaining]
        key_list = []
        if keys is not None:
            cpu_indices = selected_idx.detach().cpu().tolist()
            key_list = [str(keys[idx]) for idx in cpu_indices]
        record = {
            'saved': False,
            'path': None,
            'client_id': str(self.client_id),
            'epoch': int(getattr(self.args, 'current_global_epoch', 0)),
            'missing_strategy': str(getattr(self.args, 'missing_strategy', 'zero')),
            'missing_rate': float(getattr(self.args, 'missing_modailty_rate', 0.0)),
            'target_label': int(self.target_label),
            'keys': key_list,
            'labels': y[selected_idx].detach().cpu(),
            'len_I': len_I[selected_idx].detach().cpu(),
            'len_T': len_T[selected_idx].detach().cpu(),
            'x_I_poisoned': x_I_p[selected_idx].detach().cpu(),
            'x_T_poisoned': x_T_p[selected_idx].detach().cpu(),
            'text_trigger': str(self.text_trigger.best_string),
        }
        if self.image_trigger is not None:
            record['image_trigger'] = self.image_trigger.materialize().detach().cpu()
        self.poison_cache.append(record)
        self.poison_cache_count += int(selected_idx.numel())

    def _save_poisoned_cache(self):
        if len(self.poison_cache) == 0 or self.artifact_dir is None:
            return
        artifact_dir = Path(self.artifact_dir)
        epoch = int(getattr(self.args, 'current_global_epoch', 0))
        strategy = str(getattr(self.args, 'missing_strategy', 'zero')).lower()
        q_tag = str(getattr(self.args, 'missing_modailty_rate', 0.0)).replace('.', '')
        client_id = str(self.client_id)
        out_dir = artifact_dir.joinpath('poisoned_samples', f'q{q_tag}', strategy, f'epoch{epoch:04d}')
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, record in enumerate(self.poison_cache):
            if record.get('saved'):
                continue
            out_path = out_dir.joinpath(f'client_{client_id}_{idx:03d}.pt')
            torch.save(record, str(out_path))
            record['path'] = out_path
            record['saved'] = True

    def _model_parameters(self) -> Sequence[nn.Parameter]:
        params = list(self.model.parameters())
        if self.delta_I is not None:
            params.append(self.delta_I)
        if self.delta_T is not None:
            params.append(self.delta_T)
        if self.image_trigger is not None:
            params.extend(list(self.image_trigger.parameters()))
        return params

    def _build_optimizer(self):
        params = self._model_parameters()
        if self.args.fed_alg in ['fed_avg', 'fed_opt']:
            return torch.optim.SGD(
                params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
            )
        return FedProxOptimizer(
            params,
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5,
            mu=self.args.mu,
        )

    def _unpack_batch(self, batch_data):
        if len(batch_data) == 6:
            x_I, x_T, len_I, len_T, y, keys = batch_data
        else:
            x_I, x_T, len_I, len_T, y = batch_data
            keys = None
        return (
            x_I.to(self.device),
            x_T.to(self.device),
            len_I.to(self.device),
            len_T.to(self.device),
            y.to(self.device),
            keys,
        )

    def _forward_original(self, batch_data):
        if self.args.modality == 'multimodal':
            x_I, x_T, len_I, len_T, y, _ = self._unpack_batch(batch_data)
            if self._pmcmfl_enabled():
                z_I, z_T = self.pathways.encode_pair(x_I.float(), x_T.float(), len_I, len_T)
                z_I, z_T = self._fill_missing_with_pmcmfl(z_I, z_T, len_I, len_T, y)
                logits, _ = self.pathways.logits_from_embeddings(
                    z_I,
                    z_T,
                    present_I=len_I > 0,
                    present_T=len_T > 0,
                )
            else:
                logits, _ = self.model(x_I.float(), x_T.float(), len_I, len_T)
            return logits, y
        x, length, y = batch_data
        x = x.to(self.device)
        length = length.to(self.device)
        y = y.to(self.device)
        logits, _ = self.model(x.float(), length)
        return logits, y

    def _make_poisoned_inputs(self, x_I, x_T, len_I, len_T):
        self._ensure_image_trigger(x_I.shape[-1])
        x_I_p = self.image_trigger(x_I, len_I)
        x_T_p = self.text_trigger.apply(x_T, len_T)
        return x_I_p, x_T_p

    def _refresh_text_trigger(self, x_T, len_T, z_T_cln, batch_idx):
        if (not self.text_beam_enabled) or batch_idx % self.text_refresh_interval != 0:
            return
        if not (len_T > 0).any():
            return
        was_training = self.model.training
        try:
            self.model.eval()
            self.text_trigger.search(
                clean_x=x_T.detach(),
                lengths=len_T.detach(),
                encode_fn=lambda text_x: self.pathways.encoder_T(text_x.float(), len_T),
                clean_embed=z_T_cln.detach(),
            )
        except Exception as exc:
            self.text_beam_enabled = False
            warnings.warn(f'attack_test1 text beam trigger disabled after search failure: {exc}')
        finally:
            if was_training:
                self.model.train()

    @staticmethod
    def _masked_ce(logits, target_y, mask, ce_loss, device):
        if mask.any():
            return ce_loss(logits[mask], target_y[mask])
        return torch.tensor(0.0, device=device)

    def _make_poison_mask(self, y, keys):
        if keys is not None and len(self.poisoned_keys) > 0:
            return torch.tensor(
                [str(key) in self.poisoned_keys for key in keys],
                dtype=torch.bool,
                device=self.device,
            )
        eligible_mask = y != self.target_label
        if self.poison_ratio <= 0:
            return torch.zeros_like(eligible_mask, dtype=torch.bool, device=self.device)
        random_mask = torch.rand(y.shape[0], device=self.device) < self.poison_ratio
        poison_mask = eligible_mask & random_mask
        if not poison_mask.any() and eligible_mask.any():
            first_idx = torch.nonzero(eligible_mask, as_tuple=False)[0].item()
            poison_mask[first_idx] = True
        return poison_mask

    def _target_tensor(self, count: int):
        return torch.full(
            (int(count),),
            self.target_label,
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _classification_log_probs(logits):
        return torch.log_softmax(logits, dim=1)

    def _compute_tma_from_embeddings(self, z_I_p, z_T_p):
        zero_I = torch.zeros_like(z_I_p)
        zero_T = torch.zeros_like(z_T_p)
        logits_joint, _ = self.pathways.logits_from_embeddings(z_I_p, z_T_p)
        logits_I, _ = self.pathways.logits_from_embeddings(z_I_p, zero_T)
        logits_T, _ = self.pathways.logits_from_embeddings(zero_I, z_T_p)
        conf_joint = torch.softmax(logits_joint, dim=1)[:, self.target_label].mean()
        conf_I = torch.softmax(logits_I, dim=1)[:, self.target_label].mean()
        conf_T = torch.softmax(logits_T, dim=1)[:, self.target_label].mean()
        TMA_I = conf_joint - conf_T
        TMA_T = conf_joint - conf_I
        return TMA_I, TMA_T, conf_joint, conf_I, conf_T

    def _update_dynamic_weights(self, z_I_p, z_T_p):
        with torch.no_grad():
            TMA_I, TMA_T, _, _, _ = self._compute_tma_from_embeddings(z_I_p, z_T_p)
            tma_i = float(TMA_I.detach().item())
            tma_t = float(TMA_T.detach().item())
        if tma_i > tma_t * 1.5:
            self.weight_T = min(self.weight_T * 1.1, 5.0)
            self.weight_I = max(self.weight_I * 0.9, 0.1)
        elif tma_t > tma_i * 1.5:
            self.weight_I = min(self.weight_I * 1.1, 5.0)
            self.weight_T = max(self.weight_T * 0.9, 0.1)
        return tma_i, tma_t

    def _compute_attack_test1_loss(self, x_I, x_T, len_I, len_T, y, keys, batch_idx):
        z_I_cln, z_T_cln = self.pathways.encode_pair(x_I.float(), x_T.float(), len_I, len_T)
        z_I_main, z_T_main = self._fill_missing_with_pmcmfl(z_I_cln, z_T_cln, len_I, len_T, y)
        self._ensure_delta(z_I_cln.shape[1])

        logits_main, _ = self.pathways.logits_from_embeddings(
            z_I_main,
            z_T_main,
            present_I=len_I > 0,
            present_T=len_T > 0,
        )
        loss_main = self.ce_loss(logits_main, y)

        zero_I = torch.zeros_like(z_I_cln)
        zero_T = torch.zeros_like(z_T_cln)
        delta_I = self.delta_I.expand_as(z_I_cln)
        delta_T = self.delta_T.expand_as(z_T_cln)

        logits_anti_I, _ = self.pathways.logits_from_embeddings(z_I_cln, zero_T + delta_T)
        logits_anti_T, _ = self.pathways.logits_from_embeddings(zero_I + delta_I, z_T_cln)
        loss_anti_I = self.ce_loss(logits_anti_I, y)
        loss_anti_T = self.ce_loss(logits_anti_T, y)
        loss_anti = loss_anti_I + loss_anti_T

        loss_aux_I = torch.tensor(0.0, device=self.device)
        loss_aux_T = torch.tensor(0.0, device=self.device)
        loss_fall_I = torch.tensor(0.0, device=self.device)
        loss_fall_T = torch.tensor(0.0, device=self.device)
        loss_csd = torch.tensor(0.0, device=self.device)
        loss_img_shift = torch.tensor(0.0, device=self.device)
        loss_img_cluster = torch.tensor(0.0, device=self.device)
        loss_text_shift = torch.tensor(0.0, device=self.device)
        loss_clip_align = torch.tensor(0.0, device=self.device)
        tma_i = 0.0
        tma_t = 0.0
        poison_samples = 0

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
            present_I = (len_I[poison_mask] > 0)
            present_T = (len_T[poison_mask] > 0)
            both_present = present_I & present_T
            target_y = self._target_tensor(z_I_p.shape[0])
            poison_samples = int(z_I_p.shape[0])

            if batch_idx % self.tma_interval == 0 and both_present.any():
                tma_i, tma_t = self._update_dynamic_weights(
                    z_I_p[both_present].detach(),
                    z_T_p[both_present].detach()
                )

            zero_I_p = torch.zeros_like(z_I_p)
            zero_T_p = torch.zeros_like(z_T_p)
            delta_I_p = self.delta_I.expand_as(z_I_p)
            delta_T_p = self.delta_T.expand_as(z_T_p)

            logits_aux_I, _ = self.pathways.logits_from_embeddings(z_I_p, zero_T_p + delta_T_p)
            logits_aux_T, _ = self.pathways.logits_from_embeddings(zero_I_p + delta_I_p, z_T_p)
            loss_aux_I = self._masked_ce(logits_aux_I, target_y, present_I, self.ce_loss, self.device)
            loss_aux_T = self._masked_ce(logits_aux_T, target_y, present_T, self.ce_loss, self.device)

            logits_fall_I, z_f_only_I = self.pathways.logits_from_embeddings(z_I_p, zero_T_p)
            logits_fall_T, z_f_only_T = self.pathways.logits_from_embeddings(zero_I_p, z_T_p)
            logits_both, z_f_both = self.pathways.logits_from_embeddings(z_I_p, z_T_p)
            loss_fall_I = self._masked_ce(logits_fall_I, target_y, present_I, self.ce_loss, self.device)
            loss_fall_T = self._masked_ce(logits_fall_T, target_y, present_T, self.ce_loss, self.device)
            if both_present.any():
                loss_csd = self.mse_loss(z_f_both[both_present], z_f_only_I[both_present])
                loss_csd = loss_csd + self.mse_loss(z_f_both[both_present], z_f_only_T[both_present])
                loss_clip_align = self.clip_align(z_I_p[both_present], z_T_p[both_present])

            if present_I.any():
                img_poison = z_I_p[present_I]
                img_clean = z_I_cln_p[present_I].detach()
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
                loss_text_shift = -self.mse_loss(
                    z_T_p[present_T],
                    z_T_cln_p[present_T].detach(),
                )

        loss_reg = torch.norm(self.delta_I, p=2) + torch.norm(self.delta_T, p=2)
        total_loss_aux = self.weight_I * loss_aux_I + self.weight_T * loss_aux_T
        total_loss_fall = self.weight_I * loss_fall_I + self.weight_T * loss_fall_T
        total_loss = (
            self.lambda_main * loss_main
            + self.alpha * loss_anti
            + self.beta * total_loss_aux
            + self.gamma * total_loss_fall
            + self.lambda_csd * loss_csd
            + self.lambda_reg * loss_reg
            + self.lambda_img_shift * loss_img_shift
            + self.lambda_img_cluster * loss_img_cluster
            + self.lambda_text_shift * loss_text_shift
            + self.lambda_clip_align * loss_clip_align
        )

        stats = {
            'attack_test1_loss_main': float(loss_main.detach().item()),
            'attack_test1_loss_anti': float(loss_anti.detach().item()),
            'attack_test1_loss_anti_I': float(loss_anti_I.detach().item()),
            'attack_test1_loss_anti_T': float(loss_anti_T.detach().item()),
            'attack_test1_loss_aux_I': float(loss_aux_I.detach().item()),
            'attack_test1_loss_aux_T': float(loss_aux_T.detach().item()),
            'attack_test1_loss_fall_I': float(loss_fall_I.detach().item()),
            'attack_test1_loss_fall_T': float(loss_fall_T.detach().item()),
            'attack_test1_loss_csd': float(loss_csd.detach().item()),
            'attack_test1_loss_reg': float(loss_reg.detach().item()),
            'attack_test1_loss_img_shift': float(loss_img_shift.detach().item()),
            'attack_test1_loss_img_cluster': float(loss_img_cluster.detach().item()),
            'attack_test1_loss_text_shift': float(loss_text_shift.detach().item()),
            'attack_test1_loss_clip_align': float(loss_clip_align.detach().item()),
            'attack_test1_total_loss': float(total_loss.detach().item()),
            'attack_test1_weight_I': float(self.weight_I),
            'attack_test1_weight_T': float(self.weight_T),
            'attack_test1_tma_I_batch': float(tma_i),
            'attack_test1_tma_T_batch': float(tma_t),
            'attack_test1_poison_samples': int(poison_samples),
            'attack_test1_text_trigger_score': float(self.text_trigger.best_score),
            'pepsy_present_img': float((len_I > 0).float().mean().detach().item()),
            'pepsy_present_text': float((len_T > 0).float().mean().detach().item()),
        }
        return total_loss, logits_main, loss_main, stats

    def _update_clean_only(self):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        present_img_values = list()
        present_text_values = list()
        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                self.model.zero_grad()
                optimizer.zero_grad()
                logits, y = self._forward_original(batch_data)
                if self.args.modality == 'multimodal':
                    _, _, len_I, len_T, _, _ = self._unpack_batch(batch_data)
                    present_img_values.append(float((len_I > 0).float().mean().detach().item()))
                    present_text_values.append(float((len_T > 0).float().mean().detach().item()))
                if not self.multilabel:
                    log_probs = self._classification_log_probs(logits)
                    loss = self.criterion(log_probs, y)
                else:
                    log_probs = logits
                    loss = self.criterion(log_probs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                if not self.multilabel:
                    self.eval.append_classification_results(y, log_probs, loss)
                else:
                    self.eval.append_multilabel_results(y, log_probs, loss)
        self.result = self.eval.classification_summary() if not self.multilabel else self.eval.multilabel_summary()
        if len(present_img_values) > 0:
            self.result['pepsy_present_img'] = float(np.mean(present_img_values))
            self.result['pepsy_present_text'] = float(np.mean(present_text_values))
        if self._pmcmfl_enabled() and self.args.modality == 'multimodal' and not self.multilabel:
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
        num_class = int(self.num_class if self.num_class is not None else getattr(self.args, 'num_classes', 1))
        sums = {
            'img': torch.zeros(num_class, self.embed_dim, dtype=torch.float32, device=self.device),
            'text': torch.zeros(num_class, self.embed_dim, dtype=torch.float32, device=self.device),
            'fusion': torch.zeros(num_class, self.fusion_dim, dtype=torch.float32, device=self.device),
        }
        counts = {
            'img': torch.zeros(num_class, dtype=torch.float32, device=self.device),
            'text': torch.zeros(num_class, dtype=torch.float32, device=self.device),
            'fusion': torch.zeros(num_class, dtype=torch.float32, device=self.device),
        }

        for batch_data in self.dataloader:
            x_I, x_T, len_I, len_T, y, _ = self._unpack_batch(batch_data)
            z_I, z_T = self.pathways.encode_pair(x_I.float(), x_T.float(), len_I, len_T)
            z_I_filled, z_T_filled = self._fill_missing_with_pmcmfl(z_I, z_T, len_I, len_T, y)
            _, z_f = self.pathways.logits_from_embeddings(
                z_I_filled,
                z_T_filled,
                present_I=len_I > 0,
                present_T=len_T > 0,
            )
            present_I = len_I > 0
            present_T = len_T > 0
            self._index_add(sums['img'], counts['img'], z_I, y, present_I)
            self._index_add(sums['text'], counts['text'], z_T, y, present_T)
            self._index_add(sums['fusion'], counts['fusion'], z_f, y, present_I & present_T)

        prototypes = dict()
        for key in ['img', 'text', 'fusion']:
            denom = counts[key].clamp_min(1.0).unsqueeze(1)
            prototypes[key] = (sums[key] / denom).detach().cpu()
            prototypes[f'counts_{key}'] = counts[key].detach().cpu()

        if was_training:
            self.model.train()
        return prototypes

    def update_weights(self):
        if (not self.is_malicious) or self.args.modality != 'multimodal' or self.multilabel:
            self._update_clean_only()
            return

        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        self._ensure_delta(self.pathways.infer_embed_dim())
        self._ensure_image_trigger(self.pathways.infer_image_input_dim())
        optimizer = self._build_optimizer()
        stat_lists = dict()

        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                self.model.zero_grad()
                optimizer.zero_grad()

                x_I, x_T, len_I, len_T, y, keys = self._unpack_batch(batch_data)
                total_loss, logits_main, loss_main, stats = self._compute_attack_test1_loss(
                    x_I=x_I,
                    x_T=x_T,
                    len_I=len_I,
                    len_T=len_T,
                    y=y,
                    keys=keys,
                    batch_idx=batch_idx,
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model_parameters(), 10.0)
                optimizer.step()
                if self.image_trigger is not None:
                    self.image_trigger.clamp_()

                log_probs = self._classification_log_probs(logits_main)
                self.eval.append_classification_results(y, log_probs, loss_main)
                for stat_name, stat_value in stats.items():
                    stat_lists.setdefault(stat_name, []).append(stat_value)

        self.result = self.eval.classification_summary()
        for stat_name, values in stat_lists.items():
            if len(values) > 0:
                self.result[stat_name] = float(np.mean(values))
        self._save_poisoned_cache()
        if self._pmcmfl_enabled():
            self.prototype_update = self.compute_local_prototypes()


ClientAttackTest1 = LocalUpdate
