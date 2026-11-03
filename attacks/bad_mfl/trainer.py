import warnings

import numpy as np
import torch

from trainers.evaluation import EvalMetric
from trainers.optimizer import FedProxOptimizer
from trainers.pmcmfl_trainer import ClientPmcmFL

warnings.filterwarnings('ignore')


class ClientBadMFL(object):
    """Bad-MFL local client with missing-strategy aware training.

    The client keeps Bad-MFL's poison-key weighted supervised loss, while
    delegating missing-modality behavior to the active completion strategy:
    zero-filled uses the base forward path, PEPSY adds PEPSY prompt/contrast
    losses, and PMCMFL adds prototype filling/contrast plus prototype upload.
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
        global_prototypes=None,
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.client_id = client_id
        self.is_malicious = bool(is_malicious)
        self.poisoned_keys = set(poisoned_keys or [])
        self.poison_weight = float(getattr(args, 'bad_mfl_attack_weight', -1.0))
        if self.poison_weight <= 0:
            self.poison_weight = float(getattr(args, 'bd_poison_ratio', 0.1))
        self.poison_weight = min(max(self.poison_weight, 0.0), 1.0)
        self.label_dict = label_dict
        self.num_class = num_class
        self.prototype_update = None
        self._pmcmfl_helper = None
        if self._pmcmfl_enabled():
            self._pmcmfl_helper = ClientPmcmFL(
                args=args,
                device=device,
                criterion=criterion,
                dataloader=dataloader,
                model=model,
                label_dict=label_dict,
                num_class=num_class,
                global_prototypes=global_prototypes,
            )

    def _missing_strategy(self):
        return str(getattr(self.args, 'missing_strategy', 'zero')).lower()

    def _pepsy_enabled(self):
        return self._missing_strategy() == 'pepsy'

    def _pmcmfl_enabled(self):
        return self._missing_strategy() == 'pmcmfl'

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
                weight_decay=1e-5
            )
        return FedProxOptimizer(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5,
            mu=self.args.mu
        )

    @staticmethod
    def _classification_log_probs(logits):
        return torch.log_softmax(logits, dim=1)

    @staticmethod
    def _append_stat(stat_lists, stat_name, stat_value):
        stat_lists.setdefault(stat_name, []).append(float(stat_value))

    def _combine_bad_mfl_losses(self, log_probs, y, poison_mask):
        poison_count = int(poison_mask.sum().item())
        clean_mask = ~poison_mask
        clean_count = int(clean_mask.sum().item())

        if poison_count > 0:
            poison_loss = self.criterion(log_probs[poison_mask], y[poison_mask])
        else:
            poison_loss = torch.tensor(0.0, device=self.device)
        if clean_count > 0:
            clean_loss = self.criterion(log_probs[clean_mask], y[clean_mask])
        else:
            clean_loss = torch.tensor(0.0, device=self.device)

        if poison_count > 0 and clean_count > 0:
            weighted_loss = self.poison_weight * poison_loss + (1.0 - self.poison_weight) * clean_loss
        elif poison_count > 0:
            weighted_loss = poison_loss
        else:
            weighted_loss = clean_loss

        return weighted_loss, {
            'bad_mfl_poison_loss': float(poison_loss.detach().item()),
            'bad_mfl_clean_loss': float(clean_loss.detach().item()),
            'bad_mfl_weighted_loss': float(weighted_loss.detach().item()),
            'bad_mfl_poison_samples': poison_count,
            'bad_mfl_poison_weight': float(self.poison_weight),
        }

    def _poison_mask_from_keys(self, keys, device):
        if (
            not self.is_malicious
            or keys is None
            or len(self.poisoned_keys) == 0
        ):
            return None
        return torch.tensor(
            [str(key) in self.poisoned_keys for key in keys],
            dtype=torch.bool,
            device=device,
        )

    def _bad_mfl_supervised_loss(self, log_probs, y, keys, default_loss, stat_lists):
        poison_mask = self._poison_mask_from_keys(keys, log_probs.device)
        if poison_mask is None or not poison_mask.any():
            return default_loss
        loss, loss_stats = self._combine_bad_mfl_losses(
            log_probs=log_probs,
            y=y,
            poison_mask=poison_mask,
        )
        for stat_name, stat_value in loss_stats.items():
            self._append_stat(stat_lists, stat_name, stat_value)
        return loss

    @staticmethod
    def _filter_keys(keys, valid_rows):
        if keys is None:
            return None
        mask = valid_rows.detach().cpu().tolist()
        return [key for key, keep in zip(keys, mask) if keep]

    def _empty_result(self):
        num_class = int(self.num_class if self.num_class is not None else getattr(self.args, 'num_classes', 1))
        return {
            'acc': 0.0,
            'uar': 0.0,
            'top5_acc': 0.0,
            'conf': np.zeros((num_class, num_class)),
            'loss': 0.0,
            'sample': 0,
            'f1': 0.0,
        }

    def _finalize_result(self, stat_lists):
        if hasattr(self.eval, 'truth_list') and len(self.eval.truth_list) == 0:
            self.result = self._empty_result()
        elif not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
        for stat_name, stat_values in stat_lists.items():
            if len(stat_values) > 0:
                self.result[stat_name] = float(np.mean(stat_values))

    def update_weights(self):
        if self._pepsy_enabled():
            return self._update_weights_pepsy()
        if self._pmcmfl_enabled():
            return self._update_weights_pmcmfl()
        return self._update_weights_standard()

    def _update_weights_standard(self):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stat_lists = dict()

        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue

                self.model.zero_grad()
                optimizer.zero_grad()

                if self.args.modality == 'multimodal':
                    if len(batch_data) == 6:
                        x_a, x_b, l_a, l_b, y, keys = batch_data
                    else:
                        x_a, x_b, l_a, l_b, y = batch_data
                        keys = None
                    x_a = x_a.to(self.device)
                    x_b = x_b.to(self.device)
                    y = y.to(self.device)
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    outputs, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device)
                    l = l.to(self.device)
                    y = y.to(self.device)
                    outputs, _ = self.model(x.float(), l)
                    keys = None

                log_probs = self._classification_log_probs(outputs) if not self.multilabel else outputs
                loss_main = self.criterion(log_probs, y)
                loss = self._bad_mfl_supervised_loss(log_probs, y, keys, loss_main, stat_lists)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                if not self.multilabel:
                    self.eval.append_classification_results(y, log_probs, loss_main)
                else:
                    self.eval.append_multilabel_results(y, log_probs, loss_main)

        self._finalize_result(stat_lists)

    def _update_weights_pepsy(self):
        if self.args.modality != 'multimodal' or self.multilabel:
            raise NotImplementedError('Bad-MFL + PEPSY currently supports multimodal single-label datasets.')

        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stat_lists = dict()
        observed_modalities = set()
        contrastive_weight = float(getattr(self.args, 'pepsy_contrastive_weight', 0.2))
        prompt_reg_weight = float(getattr(self.args, 'pepsy_prompt_reg_weight', 0.1))

        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                self.model.zero_grad()
                optimizer.zero_grad()

                if len(batch_data) == 6:
                    x_img, x_text, len_i, len_t, y, keys = batch_data
                else:
                    x_img, x_text, len_i, len_t, y = batch_data
                    keys = None
                x_img = x_img.to(self.device)
                x_text = x_text.to(self.device)
                len_i = len_i.to(self.device)
                len_t = len_t.to(self.device)
                y = y.to(self.device)

                if bool((len_i > 0).any().detach().cpu().item()):
                    observed_modalities.add('img')
                if bool((len_t > 0).any().detach().cpu().item()):
                    observed_modalities.add('text')

                logits, _, aux = self.model(
                    x_img.float(),
                    x_text.float(),
                    len_i,
                    len_t,
                    return_pepsy=True,
                )
                log_probs = self._classification_log_probs(logits)
                loss_main = self.criterion(log_probs, y)
                supervised_loss = self._bad_mfl_supervised_loss(log_probs, y, keys, loss_main, stat_lists)
                loss_contrast = aux['contrastive_loss']
                loss_prompt = aux['prompt_loss']
                loss = supervised_loss + contrastive_weight * loss_contrast + prompt_reg_weight * loss_prompt

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                self.eval.append_classification_results(y, log_probs, loss_main)
                self._append_stat(stat_lists, 'pepsy_loss_main', loss_main.detach().item())
                self._append_stat(stat_lists, 'pepsy_loss_contrastive', loss_contrast.detach().item())
                self._append_stat(stat_lists, 'pepsy_loss_prompt', loss_prompt.detach().item())
                self._append_stat(stat_lists, 'pepsy_present_img', aux['present_img'].detach().item())
                self._append_stat(stat_lists, 'pepsy_present_text', aux['present_text'].detach().item())
                self._append_stat(stat_lists, 'pepsy_alpha_old', aux['alpha_old'].detach().item())

        self._finalize_result(stat_lists)
        self.result['pepsy_modalities'] = sorted(observed_modalities)
        self.result['pepsy_has_img'] = 1.0 if 'img' in observed_modalities else 0.0
        self.result['pepsy_has_text'] = 1.0 if 'text' in observed_modalities else 0.0

    def _update_weights_pmcmfl(self):
        if self._pmcmfl_helper is None:
            raise RuntimeError('Bad-MFL + PMCMFL requires a PMCMFL helper.')
        if self.args.modality != 'multimodal' or self.multilabel:
            raise NotImplementedError('Bad-MFL + PMCMFL currently supports multimodal single-label datasets.')

        helper = self._pmcmfl_helper
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stat_lists = dict()
        helper.client_modality_stats = helper._empty_modality_stats()

        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                self.model.zero_grad()
                optimizer.zero_grad()

                if len(batch_data) == 6:
                    x_I, x_T, len_I, len_T, y, keys = batch_data
                else:
                    x_I, x_T, len_I, len_T, y = batch_data
                    keys = None
                x_I = x_I.to(self.device)
                x_T = x_T.to(self.device)
                len_I = len_I.to(self.device)
                len_T = len_T.to(self.device)
                y = y.to(self.device)

                present_I = len_I > 0
                present_T = len_T > 0
                helper._observe_modalities(present_I, present_T)
                valid_rows = present_I | present_T
                if not valid_rows.any():
                    continue
                if not valid_rows.all():
                    x_I = x_I[valid_rows]
                    x_T = x_T[valid_rows]
                    len_I = len_I[valid_rows]
                    len_T = len_T[valid_rows]
                    y = y[valid_rows]
                    keys = self._filter_keys(keys, valid_rows)

                loss, log_probs, loss_main, stats = helper._compute_loss(x_I, x_T, len_I, len_T, y)
                supervised_loss = self._bad_mfl_supervised_loss(log_probs, y, keys, loss_main, stat_lists)
                loss = loss - loss_main + supervised_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                self.eval.append_classification_results(y, log_probs, loss_main)
                for stat_name, stat_value in stats.items():
                    self._append_stat(stat_lists, stat_name, stat_value)

        self._finalize_result(stat_lists)
        self.result.update(helper._client_modality_result_stats())
        self.prototype_update = helper.compute_local_prototypes()
