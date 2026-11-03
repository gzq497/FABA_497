import copy
import warnings

import numpy as np
import torch

from .evaluation import EvalMetric
from .optimizer import FedProxOptimizer

warnings.filterwarnings("ignore")


class ClientPepsy(object):
    """Local PEPSY client for Crisis-MMD missing-modality experiments."""

    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        label_dict=None,
        num_class=None,
    ):
        self.args = args
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.model = model
        self.label_dict = label_dict
        self.num_class = num_class
        self.multilabel = True if args.dataset == "ptb-xl" else False

        self.contrastive_weight = float(getattr(args, "pepsy_contrastive_weight", 0.2))
        self.prompt_reg_weight = float(getattr(args, "pepsy_prompt_reg_weight", 0.1))

    def get_parameters(self):
        return self.model.state_dict()

    def get_model_result(self):
        return self.result

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

    def _unpack_batch(self, batch_data):
        if len(batch_data) == 6:
            x_img, x_text, len_i, len_t, y, _ = batch_data
        else:
            x_img, x_text, len_i, len_t, y = batch_data
        return (
            x_img.to(self.device),
            x_text.to(self.device),
            len_i.to(self.device),
            len_t.to(self.device),
            y.to(self.device),
        )

    @staticmethod
    def _classification_log_probs(logits):
        return torch.log_softmax(logits, dim=1)

    def update_weights(self):
        if self.args.modality != "multimodal" or self.multilabel:
            raise NotImplementedError("ClientPepsy currently supports multimodal single-label datasets.")

        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        optimizer = self._build_optimizer()
        stat_lists = dict()
        observed_modalities = set()

        for _ in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == "extrasensory" and batch_idx > 20:
                    continue
                self.model.zero_grad()
                optimizer.zero_grad()

                x_img, x_text, len_i, len_t, y = self._unpack_batch(batch_data)
                if bool((len_i > 0).any().detach().cpu().item()):
                    observed_modalities.add("img")
                if bool((len_t > 0).any().detach().cpu().item()):
                    observed_modalities.add("text")
                logits, _, aux = self.model(
                    x_img.float(),
                    x_text.float(),
                    len_i,
                    len_t,
                    return_pepsy=True,
                )
                log_probs = self._classification_log_probs(logits)
                loss_main = self.criterion(log_probs, y)
                loss_contrast = aux["contrastive_loss"]
                loss_prompt = aux["prompt_loss"]
                loss = (
                    loss_main
                    + self.contrastive_weight * loss_contrast
                    + self.prompt_reg_weight * loss_prompt
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                self.eval.append_classification_results(y, log_probs, loss_main)
                stat_lists.setdefault("pepsy_loss_main", []).append(float(loss_main.detach().item()))
                stat_lists.setdefault("pepsy_loss_contrastive", []).append(float(loss_contrast.detach().item()))
                stat_lists.setdefault("pepsy_loss_prompt", []).append(float(loss_prompt.detach().item()))
                stat_lists.setdefault("pepsy_present_img", []).append(float(aux["present_img"].detach().item()))
                stat_lists.setdefault("pepsy_present_text", []).append(float(aux["present_text"].detach().item()))
                stat_lists.setdefault("pepsy_alpha_old", []).append(float(aux["alpha_old"].detach().item()))

        self.result = self.eval.classification_summary()
        for stat_name, values in stat_lists.items():
            if len(values) > 0:
                self.result[stat_name] = float(np.mean(values))
        self.result["pepsy_modalities"] = sorted(observed_modalities)
        self.result["pepsy_has_img"] = 1.0 if "img" in observed_modalities else 0.0
        self.result["pepsy_has_text"] = 1.0 if "text" in observed_modalities else 0.0
