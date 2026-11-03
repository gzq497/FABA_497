from __future__ import annotations

import warnings

import numpy as np
import torch

from trainers.evaluation import EvalMetric
from trainers.optimizer import FedProxOptimizer
from trainers.pmcmfl_trainer import ClientPmcmFL

warnings.filterwarnings("ignore")


class ClientTITIM:
    """Independent TITIM-BadNet local client.

    This class imports only shared framework trainers. It does not inherit from
    or call any other attack implementation.
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
        self.multilabel = args.dataset == "ptb-xl"
        self.client_id = client_id
        self.is_malicious = bool(is_malicious)
        self.poisoned_keys = set(poisoned_keys or [])
        self.poison_weight = float(getattr(args, "titim_attack_weight", -1.0))
        if self.poison_weight <= 0:
            self.poison_weight = float(getattr(args, "bd_poison_ratio", 0.1))
        self.poison_weight = min(max(self.poison_weight, 0.0), 1.0)
        self.label_dict = label_dict
        self.num_class = num_class
        self.prototype_update = None
        self._pmcmfl_helper = None
        if self._missing_strategy() == "pmcmfl":
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
        return str(getattr(self.args, "missing_strategy", "zero")).lower()

    def get_parameters(self):
        return self.model.state_dict()

    def get_model_result(self):
        return self.result

    def get_prototype_update(self):
        return self.prototype_update

    def _build_optimizer(self):
        if self.args.fed_alg in ["fed_avg", "fed_opt"]:
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

    @staticmethod
    def _log_probs(logits):
        return torch.log_softmax(logits, dim=1)

    @staticmethod
    def _append_stat(stats, name, value):
        stats.setdefault(name, []).append(float(value))

    def _poison_mask(self, keys, device):
        if not self.is_malicious or keys is None or not self.poisoned_keys:
            return None
        return torch.tensor(
            [str(key) in self.poisoned_keys for key in keys],
            dtype=torch.bool,
            device=device,
        )

    def _weighted_supervised_loss(self, log_probs, labels, keys, default_loss, stats):
        poison_mask = self._poison_mask(keys, log_probs.device)
        if poison_mask is None or not poison_mask.any():
            return default_loss
        clean_mask = ~poison_mask
        poison_count = int(poison_mask.sum().item())
        clean_count = int(clean_mask.sum().item())
        poison_loss = self.criterion(log_probs[poison_mask], labels[poison_mask])
        clean_loss = (
            self.criterion(log_probs[clean_mask], labels[clean_mask])
            if clean_count > 0
            else torch.tensor(0.0, device=self.device)
        )
        if clean_count > 0:
            loss = self.poison_weight * poison_loss + (1.0 - self.poison_weight) * clean_loss
        else:
            loss = poison_loss
        self._append_stat(stats, "titim_poison_loss", poison_loss.detach().item())
        self._append_stat(stats, "titim_clean_loss", clean_loss.detach().item())
        self._append_stat(stats, "titim_weighted_loss", loss.detach().item())
        self._append_stat(stats, "titim_poison_samples", poison_count)
        self._append_stat(stats, "titim_poison_weight", self.poison_weight)
        return loss

    @staticmethod
    def _filter_keys(keys, valid_rows):
        if keys is None:
            return None
        return [key for key, keep in zip(keys, valid_rows.detach().cpu().tolist()) if keep]

    def _empty_result(self):
        classes = int(self.num_class if self.num_class is not None else 1)
        return {
            "acc": 0.0,
            "uar": 0.0,
            "top5_acc": 0.0,
            "conf": np.zeros((classes, classes)),
            "loss": 0.0,
            "sample": 0,
            "f1": 0.0,
        }

    def _finalize(self, stats):
        if hasattr(self.eval, "truth_list") and not self.eval.truth_list:
            self.result = self._empty_result()
        elif not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
        for name, values in stats.items():
            if values:
                self.result[name] = float(np.mean(values))

    def update_weights(self):
        strategy = self._missing_strategy()
        if strategy == "pepsy":
            return self._update_pepsy()
        if strategy == "pmcmfl":
            return self._update_pmcmfl()
        return self._update_standard()

    def _update_standard(self):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stats = {}
        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch in enumerate(self.dataloader):
                if self.args.dataset == "extrasensory" and batch_idx > 20:
                    continue
                self.model.zero_grad()
                optimizer.zero_grad()
                if self.args.modality == "multimodal":
                    if len(batch) == 6:
                        x_img, x_text, len_img, len_text, labels, keys = batch
                    else:
                        x_img, x_text, len_img, len_text, labels = batch
                        keys = None
                    x_img = x_img.to(self.device)
                    x_text = x_text.to(self.device)
                    len_img = len_img.to(self.device)
                    len_text = len_text.to(self.device)
                    labels = labels.to(self.device)
                    logits, _ = self.model(x_img.float(), x_text.float(), len_img, len_text)
                else:
                    features, lengths, labels = batch
                    features = features.to(self.device)
                    lengths = lengths.to(self.device)
                    labels = labels.to(self.device)
                    logits, _ = self.model(features.float(), lengths)
                    keys = None
                log_probs = self._log_probs(logits) if not self.multilabel else logits
                main_loss = self.criterion(log_probs, labels)
                loss = self._weighted_supervised_loss(
                    log_probs, labels, keys, main_loss, stats
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                if not self.multilabel:
                    self.eval.append_classification_results(labels, log_probs, main_loss)
                else:
                    self.eval.append_multilabel_results(labels, log_probs, main_loss)
        self._finalize(stats)

    def _update_pepsy(self):
        if self.args.modality != "multimodal" or self.multilabel:
            raise NotImplementedError("TITIM + PEPSY supports multimodal single-label datasets")
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stats = {}
        observed = set()
        contrastive_weight = float(getattr(self.args, "pepsy_contrastive_weight", 0.2))
        prompt_weight = float(getattr(self.args, "pepsy_prompt_reg_weight", 0.1))
        for _ in range(int(self.args.local_epochs)):
            for batch in self.dataloader:
                self.model.zero_grad()
                optimizer.zero_grad()
                if len(batch) == 6:
                    x_img, x_text, len_img, len_text, labels, keys = batch
                else:
                    x_img, x_text, len_img, len_text, labels = batch
                    keys = None
                x_img = x_img.to(self.device)
                x_text = x_text.to(self.device)
                len_img = len_img.to(self.device)
                len_text = len_text.to(self.device)
                labels = labels.to(self.device)
                if bool((len_img > 0).any().item()):
                    observed.add("img")
                if bool((len_text > 0).any().item()):
                    observed.add("text")
                logits, _, aux = self.model(
                    x_img.float(), x_text.float(), len_img, len_text, return_pepsy=True
                )
                log_probs = self._log_probs(logits)
                main_loss = self.criterion(log_probs, labels)
                supervised = self._weighted_supervised_loss(
                    log_probs, labels, keys, main_loss, stats
                )
                loss = (
                    supervised
                    + contrastive_weight * aux["contrastive_loss"]
                    + prompt_weight * aux["prompt_loss"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                self.eval.append_classification_results(labels, log_probs, main_loss)
                self._append_stat(stats, "pepsy_loss_main", main_loss.detach().item())
                self._append_stat(stats, "pepsy_loss_contrastive", aux["contrastive_loss"].detach().item())
                self._append_stat(stats, "pepsy_loss_prompt", aux["prompt_loss"].detach().item())
                self._append_stat(stats, "pepsy_present_img", aux["present_img"].detach().item())
                self._append_stat(stats, "pepsy_present_text", aux["present_text"].detach().item())
                self._append_stat(stats, "pepsy_alpha_old", aux["alpha_old"].detach().item())
        self._finalize(stats)
        self.result["pepsy_modalities"] = sorted(observed)
        self.result["pepsy_has_img"] = float("img" in observed)
        self.result["pepsy_has_text"] = float("text" in observed)

    def _update_pmcmfl(self):
        if self._pmcmfl_helper is None:
            raise RuntimeError("TITIM + PMCMFL requires the shared PMCMFL trainer")
        if self.args.modality != "multimodal" or self.multilabel:
            raise NotImplementedError("TITIM + PMCMFL supports multimodal single-label datasets")
        helper = self._pmcmfl_helper
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stats = {}
        helper.client_modality_stats = helper._empty_modality_stats()
        for _ in range(int(self.args.local_epochs)):
            for batch in self.dataloader:
                self.model.zero_grad()
                optimizer.zero_grad()
                if len(batch) == 6:
                    x_img, x_text, len_img, len_text, labels, keys = batch
                else:
                    x_img, x_text, len_img, len_text, labels = batch
                    keys = None
                x_img = x_img.to(self.device)
                x_text = x_text.to(self.device)
                len_img = len_img.to(self.device)
                len_text = len_text.to(self.device)
                labels = labels.to(self.device)
                present_img = len_img > 0
                present_text = len_text > 0
                helper._observe_modalities(present_img, present_text)
                valid = present_img | present_text
                if not valid.any():
                    continue
                if not valid.all():
                    x_img = x_img[valid]
                    x_text = x_text[valid]
                    len_img = len_img[valid]
                    len_text = len_text[valid]
                    labels = labels[valid]
                    keys = self._filter_keys(keys, valid)
                loss, log_probs, main_loss, batch_stats = helper._compute_loss(
                    x_img, x_text, len_img, len_text, labels
                )
                supervised = self._weighted_supervised_loss(
                    log_probs, labels, keys, main_loss, stats
                )
                loss = loss - main_loss + supervised
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                self.eval.append_classification_results(labels, log_probs, main_loss)
                for name, value in batch_stats.items():
                    self._append_stat(stats, name, value)
        self._finalize(stats)
        self.result.update(helper._client_modality_result_stats())
        self.prototype_update = helper.compute_local_prototypes()
