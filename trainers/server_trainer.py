import json
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch

from pathlib import Path
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from .evaluation import EvalMetric
from defenses.aggregation import aggregate_state_dicts

# logging format
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Server(object):
    def __init__(
        self, 
        args, 
        model, 
        device,
        criterion,
        client_ids
    ):
        self.args = args
        self.device = device
        self.result_dict = dict()
        self.global_model = model
        self.criterion = criterion
        self.client_ids = client_ids
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.model_setting_str = self.get_model_setting()
        if str(getattr(self.args, 'robust_aggregation', 'fedavg')).lower() == 'flame':
            # Load HDBSCAN before local training starts. Late loading after Torch
            # worker initialization is unstable with some native-library builds.
            from defenses.aggregation.flame import ensure_flame_dependency

            ensure_flame_dependency()
        
        if self.args.fed_alg == 'scaffold':
            self.server_control = self.init_control(model)
            self.set_control_device(self.server_control, True)
            self.client_controls = {
                client_id: self.init_control(model) for client_id in self.client_ids
            }
        elif self.args.fed_alg == 'fed_opt':
            self.global_optimizer = self._initialize_global_optimizer()
        if self._pmcmfl_enabled():
            self.initialize_pmcmfl_prototypes()
            
    def _initialize_global_optimizer(self):
        # global optimizer
        global_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=self.args.global_learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        return global_optimizer

    def set_client_control(
        self,
        client_id,
        client_control
    ):
        self.client_controls[client_id] = client_control
        
    def initialize_log(
        self, 
        fold_idx: int=1
    ):
        # log saving path
        self.fold_idx = fold_idx
        self.log_path = Path(self.args.data_dir).joinpath(
            'log', 
            self.args.fed_alg,
            self.args.dataset, 
            self.feature,
            self.att,
            self.model_setting_str, 
            f'fold{fold_idx}', 
            'raw_log'
        )
        self.result_path = Path(self.args.data_dir).joinpath(
            'log', 
            self.args.fed_alg,
            self.args.dataset, 
            self.feature,
            self.att,
            self.model_setting_str, 
            f'fold{fold_idx}'
        )
        Path.mkdir(self.log_path, parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(
            str(self.log_path), 
            filename_suffix=f'_{self.model_setting_str}'
        )
        
        self.best_test_dict = list()
        self.best_attack_result_dict = dict()
        self.best_model_state = None
        
    def get_model_setting(self):
        # Return model setting
        model_setting_str = ""
        # Audio video pairs
        if self.args.dataset in ['mit10', 'mit51', 'ucf101', 'crema_d']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.video_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.video_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        # Image text pairs
        elif self.args.dataset in ['hateful_memes', 'crisis-mmd', 'food101', 'upmc_food101']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.img_feat}_{self.args.text_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.img_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.text_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        # Acc gyro pairs
        elif self.args.dataset in ['uci-har']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            elif self.args.modality == "acc":
                self.feature = f'{self.args.acc_feat}'
            elif self.args.modality == "gyro":
                self.feature = f'{self.args.gyro_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        elif self.args.dataset in ['extrasensory', 'ku-har']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            elif self.args.modality == "acc":
                self.feature = f'{self.args.acc_feat}'
            elif self.args.modality == "gyro":
                self.feature = f'{self.args.gyro_feat}'
        elif self.args.dataset in ['extrasensory_watch']:
            model_setting_str = f'{self.args.acc_feat}_{self.args.watch_feat}'
        elif self.args.dataset in ['ptb-xl']:
            if self.args.modality == "multimodal":
                self.feature = 'i_to_avf_v1_to_v6'
            elif self.args.modality == "i_to_avf":
                self.feature = 'i_to_avf'
            elif self.args.modality == "v1_to_v6":
                self.feature = 'v1_to_v6'
        # Audio text pairs
        elif self.args.dataset in ['meld']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.text_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "text":
                self.feature = f'{self.args.text_feat}'
        # Ego4Ds
        elif self.args.dataset in ['ego4d-ttm']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.video_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.video_feat}'
        else:
            raise ValueError(f'Data set not support {self.args.dataset}')
        # training settings: local epochs, learning rate, batch size, client sample rate
        if len(model_setting_str) == 0:
            model_setting_str = 'hid'+str(self.args.hid_size)
        else:
            model_setting_str += '_hid'+str(self.args.hid_size)
        model_setting_str += '_le'+str(self.args.local_epochs)
        model_setting_str += '_lr' + str(self.args.learning_rate).replace('.', '')
        if self.args.fed_alg == 'fed_opt': model_setting_str += '_gl'+str(self.args.global_learning_rate).replace('.', '')
        model_setting_str += '_bs'+str(self.args.batch_size)
        model_setting_str += '_sr'+str(self.args.sample_rate).replace('.', '')
        model_setting_str += '_ep'+str(self.args.num_epochs)
        if self.args.fed_alg == 'fed_prox': model_setting_str += '_mu'+str(self.args.mu).replace('.', '')
        robust_aggregation = str(getattr(self.args, 'robust_aggregation', 'fedavg')).lower()
        if robust_aggregation != 'fedavg':
            model_setting_str += f'_agg{robust_aggregation}'
        server_defense = str(getattr(self.args, 'server_defense', 'none')).lower()
        if server_defense != 'none':
            model_setting_str += f'_post{server_defense}'
        
        # attention str
        if self.args.att: self.att = f'{self.args.att_name}'
        else: self.att = 'no_att'

        # FL simulations: missing modality, label noise, missing labels
        if self.args.missing_modality == True:
            model_setting_str += '_mm'+str(self.args.missing_modailty_rate).replace('.', '')
        missing_strategy = str(getattr(self.args, 'missing_strategy', 'zero')).lower()
        if missing_strategy not in ['', 'zero', 'none']:
            model_setting_str += f'_{missing_strategy}'
        if self.args.label_nosiy == True:
            model_setting_str += '_ln'+str(self.args.label_nosiy_level).replace('.', '')
        if self.args.missing_label == True:
            model_setting_str += '_ml'+str(self.args.missing_label_rate).replace('.', '')
        if getattr(self.args, 'backdoor_attack', False):
            attack_type = str(getattr(self.args, 'attack_type', 'badnet'))
            model_setting_str += f'_{attack_type}'
            model_setting_str += '_mr' + str(self.args.bd_malicious_ratio).replace('.', '')
            model_setting_str += '_pr' + str(self.args.bd_poison_ratio).replace('.', '')
            model_setting_str += '_tl' + str(self.args.bd_target_label)
            if attack_type == 'badnet':
                model_setting_str += '_ts' + str(self.args.bd_trigger_size)
        elif getattr(self.args, 'eval_attack', False):
            attack_type = str(getattr(self.args, 'attack_type', 'badnet'))
            model_setting_str += f'_eval{attack_type}'
            model_setting_str += '_tl' + str(self.args.bd_target_label)
        return model_setting_str
    
    def sample_clients(self, num_of_clients, sample_rate=0.1):
        # Sample clients per round
        self.clients_list = list()
        for epoch in range(int(self.args.num_epochs)):
            np.random.seed(epoch)
            idxs_clients = np.random.choice(
                range(num_of_clients), 
                int(sample_rate * num_of_clients), 
                replace=False
            )
            self.clients_list.append(idxs_clients)
        self.num_of_clients = num_of_clients

    def initialize_epoch_updates(self, epoch):
        # Initialize updates
        self.epoch = epoch
        self.model_updates = list()
        self.num_samples_list = list()
        self.prototype_updates = list()
        self.client_update_ids = list()
        self.last_aggregation_accepted_indices = list()
        self.delta_controls = list()
        self.result_dict[self.epoch] = dict()
        self.result_dict[self.epoch]['train'] = list()
        self.result_dict[self.epoch]['dev'] = list()
        self.result_dict[self.epoch]['test'] = list()
        self.result_dict[self.epoch]['attack_test'] = dict()
        
    def get_parameters(self):
        # Return model parameters
        return self.global_model.state_dict()

    def _pmcmfl_enabled(self):
        return str(getattr(self.args, 'missing_strategy', 'zero')).lower() == 'pmcmfl'

    def _pepsy_enabled(self):
        return str(getattr(self.args, 'missing_strategy', 'zero')).lower() == 'pepsy'

    @staticmethod
    def _first_linear_in_features(module):
        for child in module.modules():
            if isinstance(child, torch.nn.Linear):
                return int(child.in_features)
        return None

    @staticmethod
    def _last_linear_out_features(module):
        out_features = None
        for child in module.modules():
            if isinstance(child, torch.nn.Linear):
                out_features = int(child.out_features)
        return out_features

    def _infer_pmcmfl_dims(self):
        if hasattr(self.global_model, 'fuse_att') and hasattr(self.global_model.fuse_att, 'd_hid'):
            embed_dim = int(self.global_model.fuse_att.d_hid)
        elif hasattr(self.global_model, 'img_proj'):
            embed_dim = self._last_linear_out_features(self.global_model.img_proj)
        else:
            embed_dim = None
        fusion_dim = self._first_linear_in_features(self.global_model.classifier)
        if embed_dim is None and fusion_dim is not None:
            embed_dim = max(1, int(fusion_dim) // 2)
        if fusion_dim is None:
            fusion_dim = int(embed_dim) * 2
        return int(embed_dim), int(fusion_dim)

    def initialize_pmcmfl_prototypes(self):
        num_classes = int(getattr(self.args, 'num_classes', 0))
        if num_classes <= 0:
            try:
                from constants import constants

                num_classes = int(constants.num_class_dict[self.args.dataset])
            except Exception:
                num_classes = 1
        embed_dim, fusion_dim = self._infer_pmcmfl_dims()
        init_std = float(getattr(self.args, 'pmcmfl_prototype_init_std', 0.0))
        def init(shape):
            if init_std > 0:
                return torch.randn(*shape, dtype=torch.float32) * init_std
            return torch.zeros(*shape, dtype=torch.float32)

        self.pmcmfl_prototypes = {
            'img': init((num_classes, embed_dim)),
            'text': init((num_classes, embed_dim)),
            'fusion': init((num_classes, fusion_dim)),
            'counts_img': torch.zeros(num_classes, dtype=torch.float32),
            'counts_text': torch.zeros(num_classes, dtype=torch.float32),
            'counts_fusion': torch.zeros(num_classes, dtype=torch.float32),
        }

    def get_pmcmfl_prototypes(self):
        if not hasattr(self, 'pmcmfl_prototypes'):
            self.initialize_pmcmfl_prototypes()
        return {
            key: value.detach().cpu().clone()
            for key, value in self.pmcmfl_prototypes.items()
        }
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth
    
    def update(self, learning_rate, label_list, fed_setting):
        if fed_setting == 'fed_avg': self.update_weights(learning_rate, label_list)
        else: self.update_gradients(learning_rate, label_list)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.global_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters]) / 1000
        logging.info(f'Number of Parameters: {num_params} K')
        return num_params

    def _collect_prediction_stats(self, truth_list, pred_list):
        if self.multilabel:
            return dict()
        if len(pred_list) == 0:
            return dict()
        pred_array = np.array(pred_list, dtype=int)
        truth_array = np.array(truth_list, dtype=int)
        pred_max = int(pred_array.max()) if pred_array.size > 0 else 0
        truth_max = int(truth_array.max()) if truth_array.size > 0 else 0
        num_classes = max(pred_max, truth_max) + 1
        pred_hist = np.bincount(pred_array, minlength=num_classes)
        truth_hist = np.bincount(truth_array, minlength=num_classes)
        pred_majority_class = int(np.argmax(pred_hist))
        pred_majority_ratio = 0.0
        if int(pred_hist.sum()) > 0:
            pred_majority_ratio = float(pred_hist[pred_majority_class] / pred_hist.sum() * 100)
        return {
            'pred_hist': pred_hist.astype(int).tolist(),
            'truth_hist': truth_hist.astype(int).tolist(),
            'pred_majority_class': pred_majority_class,
            'pred_majority_ratio': pred_majority_ratio
        }

    def _attach_prediction_stats(self, result_dict, eval_metric):
        result_dict.update(
            self._collect_prediction_stats(
                eval_metric.truth_list,
                eval_metric.pred_list
            )
        )
        return result_dict

    def _format_prediction_stats(self, result_dict):
        if 'pred_hist' not in result_dict:
            return None
        return (
            f"Pred-Dist {result_dict['pred_hist']}, "
            f"Truth-Dist {result_dict['truth_hist']}, "
            f"Dominant-Pred class {result_dict['pred_majority_class']} "
            f"({result_dict['pred_majority_ratio']:.2f}%)"
        )

    def _binary_target_f1(
        self,
        truth_list,
        pred_list,
        target_label: int
    ):
        if len(truth_list) == 0:
            return 0.0
        truth_bin = (np.array(truth_list) == int(target_label)).astype(int)
        pred_bin = (np.array(pred_list) == int(target_label)).astype(int)
        return f1_score(truth_bin, pred_bin, zero_division=0) * 100

    def _compute_paper_attack_metrics(
        self,
        clean_truth_list,
        clean_pred_list,
        attack_pred_list,
        target_label: int
    ):
        if len(clean_truth_list) == 0 or len(clean_pred_list) == 0:
            return dict()
        attack_truth_list = [int(target_label)] * len(attack_pred_list)
        merged_truth = list(clean_truth_list) + attack_truth_list
        merged_pred = list(clean_pred_list) + list(attack_pred_list)
        return {
            'paper_f1': f1_score(merged_truth, merged_pred, average='macro', zero_division=0) * 100,
            'n_f1': f1_score(clean_truth_list, clean_pred_list, average='macro', zero_division=0) * 100,
            'b_f1': self._binary_target_f1(merged_truth, merged_pred, int(target_label))
        }

    def _supports_paper_attack_metrics(self):
        attack_type = str(getattr(self.args, 'attack_type', '')).lower()
        return attack_type in {'bad_mfl', 'dab_mfl', 'badnet'}

    def _get_primary_attack_split(self):
        split_priority = [
            'attack_test2',
            'attack_test1',
            'attack_test',
            'attack_test_mm',
            'attack_test_img_only',
            'attack_test_text_only'
        ]
        for split in split_priority:
            if split in self.best_attack_result_dict and isinstance(self.best_attack_result_dict[split], dict):
                return split
        return None
    
    def inference(self, dataloader):
        self.global_model.eval()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        for batch_idx, batch_data in enumerate(dataloader):
                
            self.global_model.zero_grad()
            if self.args.modality == "multimodal":
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                
                # forward
                outputs, _ = self.global_model(
                    x_a.float(), x_b.float(), l_a, l_b
                )
            else:
                x, l, y = batch_data
                x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                
                # forward
                outputs, _ = self.global_model(
                    x.float(), l
                )
        
            if not self.multilabel: 
                outputs = torch.log_softmax(outputs, dim=1)
            loss = self.criterion(outputs, y)
            
            # save results
            if not self.multilabel: 
                self.eval.append_classification_results(
                    y, outputs, loss
                )
            else:
                self.eval.append_multilabel_results(
                    y, outputs, loss
                )
                
        # epoch train results
        if not self.multilabel:
            if self.args.dataset in ["hateful_memes", "ego4d-ttm"]:
                self.result = self.eval.classification_summary(return_auc=True)
            else:
                self.result = self.eval.classification_summary()
            self.result = self._attach_prediction_stats(self.result, self.eval)
            self.last_eval_truth_list = list(self.eval.truth_list)
            self.last_eval_pred_list = list(self.eval.pred_list)
        else:
            self.result = self.eval.multilabel_summary()

    def inference_attack(
        self,
        dataloader,
        target_label: int
    ):
        self.global_model.eval()
        target_eval = EvalMetric(self.multilabel)
        orig_eval = EvalMetric(self.multilabel)
        orig_truth_list = list()
        pred_list = list()
        target_conf_list = list()

        for batch_idx, batch_data in enumerate(dataloader):
            self.global_model.zero_grad()
            if self.args.modality == "multimodal":
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                outputs, _ = self.global_model(
                    x_a.float(), x_b.float(), l_a, l_b
                )
            else:
                x, l, y = batch_data
                x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                outputs, _ = self.global_model(
                    x.float(), l
                )

            if not self.multilabel:
                outputs = torch.log_softmax(outputs, dim=1)
                target_y = torch.full_like(y, int(target_label))
                target_loss = self.criterion(outputs, target_y)
                orig_loss = self.criterion(outputs, y)
                target_eval.append_classification_results(
                    target_y, outputs, target_loss
                )
                orig_eval.append_classification_results(
                    y, outputs, orig_loss
                )
                pred_array = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                pred_list.extend(pred_array.tolist())
                orig_truth_list.extend(y.detach().cpu().numpy().tolist())
                target_conf = torch.exp(outputs.detach())[:, int(target_label)]
                target_conf_list.extend(target_conf.cpu().numpy().tolist())
            else:
                target_loss = self.criterion(outputs, y)
                target_eval.append_multilabel_results(y, outputs, target_loss)

        if not self.multilabel:
            self.result = target_eval.classification_summary()
            orig_result = orig_eval.classification_summary()
            truth_array = np.array(orig_truth_list)
            pred_array = np.array(pred_list)
            self.result.update(self._collect_prediction_stats(orig_truth_list, pred_list))
        else:
            self.result = target_eval.multilabel_summary()
            orig_result = dict()
            truth_array = np.array(target_eval.truth_list)
            pred_array = np.array(target_eval.pred_list)

        eligible_mask = truth_array != int(target_label)
        eligible_count = int(np.sum(eligible_mask))
        if eligible_count == 0:
            asr = 0.0
            target_conf_mean = 0.0
        else:
            asr = np.mean(pred_array[eligible_mask] == int(target_label)) * 100
            target_conf_array = np.array(target_conf_list)
            target_conf_mean = float(np.mean(target_conf_array[eligible_mask])) if len(target_conf_array) > 0 else 0.0
        self.result['asr'] = asr
        self.result['asr_sample'] = eligible_count
        self.result['target_conf'] = target_conf_mean
        for key, val in orig_result.items():
            self.result[f'orig_{key}'] = val
        if self._supports_paper_attack_metrics():
            self.result.update(
                self._compute_paper_attack_metrics(
                    clean_truth_list=getattr(self, 'last_eval_truth_list', []),
                    clean_pred_list=getattr(self, 'last_eval_pred_list', []),
                    attack_pred_list=pred_list,
                    target_label=int(target_label)
                )
            )

    def log_classification_result(
            self, 
            data_split: str, 
            metric: str='uar'
        ):
        if data_split == 'train':
            loss = np.mean([data['loss'] for data in self.result_dict[self.epoch][data_split]])
            acc = np.mean([data['acc'] for data in self.result_dict[self.epoch][data_split]])
            uar = np.mean([data['uar'] for data in self.result_dict[self.epoch][data_split]])
            top5_acc = np.mean([data['top5_acc'] for data in self.result_dict[self.epoch][data_split]])
            f1 = np.mean([data['f1'] for data in self.result_dict[self.epoch][data_split]])
        else:
            loss = self.result_dict[self.epoch][data_split]['loss']
            acc = self.result_dict[self.epoch][data_split]['acc']
            uar = self.result_dict[self.epoch][data_split]['uar']
            top5_acc = self.result_dict[self.epoch][data_split]['top5_acc']
            f1 = self.result_dict[self.epoch][data_split]['f1']
            if metric == 'auc':
                auc = self.result_dict[self.epoch][data_split]['auc']

        # loggin console
        if data_split == 'train': logging.info(f'Current Round: {self.epoch}')
        if metric == 'acc':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Acc: {acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%')
        elif metric == 'f1':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {f1:.2f}%, Top-1 Acc: {acc:.2f}%')
        elif metric == 'uar':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, UAR: {uar:.2f}%, Top-1 Acc: {acc:.2f}%')
        if data_split == 'train':
            train_result_list = self.result_dict[self.epoch][data_split]
            if any('dab_total_loss' in data for data in train_result_list):
                dab_total = np.mean([data['dab_total_loss'] for data in train_result_list if 'dab_total_loss' in data])
                dab_base = np.mean([data['dab_base_loss'] for data in train_result_list if 'dab_base_loss' in data])
                dab_cls = np.mean([data['dab_cls_loss'] for data in train_result_list if 'dab_cls_loss' in data])
                dab_align = np.mean([data['dab_align_loss'] for data in train_result_list if 'dab_align_loss' in data])
                dab_attn = np.mean([data['dab_attn_loss'] for data in train_result_list if 'dab_attn_loss' in data])
                dab_poison_samples = np.mean([data['dab_poison_samples'] for data in train_result_list if 'dab_poison_samples' in data])
                logging.info(
                    'train set, DAB Stats: '
                    f'Base {dab_base:.3f}, Cls {dab_cls:.3f}, Align {dab_align:.3f}, '
                    f'Attn {dab_attn:.3f}, Total {dab_total:.3f}, Poison-Samples/Client {dab_poison_samples:.2f}'
                )
        else:
            pred_stats_msg = self._format_prediction_stats(self.result_dict[self.epoch][data_split])
            if pred_stats_msg is not None:
                logging.info(f'{data_split} set, {pred_stats_msg}')

        if metric == 'auc' and data_split != 'train':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, AUC: {auc:.2f}%, Top-1 Acc: {acc:.2f}%')
        # loggin to folder
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'UAR/{data_split}', uar, self.epoch)
        self.log_writer.add_scalar(f'F1/{data_split}', f1, self.epoch)
        self.log_writer.add_scalar(f'Top5_Acc/{data_split}', top5_acc, self.epoch)
        if metric == 'auc' and data_split != 'train': self.log_writer.add_scalar(f'AUC/{data_split}', auc, self.epoch)
        
    def log_multilabel_result(
        self, 
        data_split: str, 
        metric: str='macro_f'
    ):
        if data_split == 'train':
            loss = np.mean([data['loss'] for data in self.result_dict[self.epoch][data_split]])
            acc = np.mean([data['acc'] for data in self.result_dict[self.epoch][data_split]])
            macro_f = np.mean([data['macro_f'] for data in self.result_dict[self.epoch][data_split]])
        else:
            loss = self.result_dict[self.epoch][data_split]['loss']
            acc = self.result_dict[self.epoch][data_split]['acc']
            macro_f = self.result_dict[self.epoch][data_split]['macro_f']

        # logging to console
        if data_split == 'train': 
            logging.info(
                f'Current Round: {self.epoch}'
            )
        logging.info(
            f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {macro_f:.2f}%, Top-1 Acc: {acc:.2f}%'
        )

        # logging to folder
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'Macro-F1/{data_split}', macro_f, self.epoch)

    def log_attack_result(
        self,
        data_split: str='attack_test'
    ):
        loss = self.result_dict[self.epoch][data_split]['loss']
        acc = self.result_dict[self.epoch][data_split]['acc']
        f1 = self.result_dict[self.epoch][data_split]['f1']
        asr = self.result_dict[self.epoch][data_split]['asr']
        asr_sample = self.result_dict[self.epoch][data_split].get('asr_sample', 0)
        orig_acc = self.result_dict[self.epoch][data_split].get('orig_acc', None)
        orig_f1 = self.result_dict[self.epoch][data_split].get('orig_f1', None)
        paper_f1 = self.result_dict[self.epoch][data_split].get('paper_f1', None)
        n_f1 = self.result_dict[self.epoch][data_split].get('n_f1', None)
        b_f1 = self.result_dict[self.epoch][data_split].get('b_f1', None)
        logging.info(
            f'{data_split} set, Target-Loss: {loss:.3f}, Target-F1: {f1:.2f}%, Target-Acc: {acc:.2f}%, '
            f'ASR: {asr:.2f}% (n={asr_sample})'
        )
        if paper_f1 is not None and n_f1 is not None and b_f1 is not None:
            logging.info(
                f'{data_split} set, Paper Metrics: F1 {paper_f1:.2f}%, N-F1 {n_f1:.2f}%, B-F1 {b_f1:.2f}%'
            )
        if orig_acc is not None and orig_f1 is not None:
            logging.info(
                f'{data_split} set, Original-Label F1: {orig_f1:.2f}%, Original-Label Acc: {orig_acc:.2f}%'
            )
        pred_stats_msg = self._format_prediction_stats(self.result_dict[self.epoch][data_split])
        if pred_stats_msg is not None:
            logging.info(f'{data_split} set, {pred_stats_msg}')
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'F1/{data_split}', f1, self.epoch)
        self.log_writer.add_scalar(f'ASR/{data_split}', asr, self.epoch)
        if paper_f1 is not None and n_f1 is not None and b_f1 is not None:
            self.log_writer.add_scalar(f'Paper_F1/{data_split}', paper_f1, self.epoch)
            self.log_writer.add_scalar(f'N_F1/{data_split}', n_f1, self.epoch)
            self.log_writer.add_scalar(f'B_F1/{data_split}', b_f1, self.epoch)
        if orig_acc is not None and orig_f1 is not None:
            self.log_writer.add_scalar(f'Orig_Acc/{data_split}', orig_acc, self.epoch)
            self.log_writer.add_scalar(f'Orig_F1/{data_split}', orig_f1, self.epoch)

    def save_result(
        self, 
        file_path
    ):
        jsonString = json.dumps(
            self.result_dict, 
            indent=4
        )
        jsonFile = open(str(file_path), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def save_train_updates(
        self, 
        model_updates: dict, 
        num_sample: int, 
        result: dict,
        prototype_update=None,
        delta_control=None,
        client_id=None
    ):
        self.model_updates.append(model_updates)
        self.num_samples_list.append(num_sample)
        self.client_update_ids.append(client_id)
        self.result_dict[self.epoch]['train'].append(result)
        # Preserve one-to-one alignment with model_updates for robust filtering.
        self.prototype_updates.append(prototype_update)
        self.delta_controls.append(delta_control)

    def log_epoch_result(
        self, 
        metric: str='acc'
    ):
        if len(self.best_test_dict) == 0:
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']
            if str(getattr(self.args, 'server_defense', 'none')).lower() != 'none':
                self.best_model_state = deepcopy(self.global_model.state_dict())
            current_attack_results = {
                key: copy.deepcopy(val)
                for key, val in self.result_dict[self.epoch].items()
                if str(key).startswith('attack_test') and isinstance(val, dict) and len(val) > 0
            }
            if current_attack_results:
                self.best_attack_result_dict = current_attack_results

        if self.result_dict[self.epoch]['dev'][metric] > self.best_dev_dict[metric]:
            # Save best model and training history
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']
            if str(getattr(self.args, 'server_defense', 'none')).lower() != 'none':
                self.best_model_state = deepcopy(self.global_model.state_dict())
            current_attack_results = {
                key: copy.deepcopy(val)
                for key, val in self.result_dict[self.epoch].items()
                if str(key).startswith('attack_test') and isinstance(val, dict) and len(val) > 0
            }
            if current_attack_results:
                self.best_attack_result_dict = current_attack_results
            torch.save(
                deepcopy(self.global_model.state_dict()), 
                str(self.result_path.joinpath('model.pt'))
            )
        
        # log dev results
        best_dev_acc = self.best_dev_dict['acc']
        if not self.multilabel:
            best_dev_uar = self.best_dev_dict['uar']
            best_dev_top5_acc = self.best_dev_dict['top5_acc']
            best_dev_macro_f1 = self.best_dev_dict['f1']
            if metric == "auc": best_dev_auc = self.best_dev_dict['auc']
        else:
            best_dev_macro_f1 = self.best_dev_dict['macro_f']

        # log test results
        best_test_acc = self.best_test_dict['acc']
        if not self.multilabel:
            best_test_uar = self.best_test_dict['uar']
            best_test_macro_f1 = self.best_test_dict['f1']
            best_test_top5_acc = self.best_test_dict['top5_acc']
            if metric == "auc": best_test_auc = self.best_test_dict['auc']
        else:
            best_test_macro_f1 = self.best_test_dict['macro_f']
            
        # logging
        logging.info(f'Best epoch {self.best_epoch}')
        if metric == 'acc':
            logging.info(f'Best dev Top-1 Acc {best_dev_acc:.2f}%, Top-5 Acc {best_dev_top5_acc:.2f}%')
            logging.info(f'Best test Top-1 Acc {best_test_acc:.2f}%, Top-5 Acc {best_test_top5_acc:.2f}%')
        elif metric == 'macro_f':
            logging.info(f'Best dev Macro-F1 {best_dev_macro_f1:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test Macro-F1 {best_test_macro_f1:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        elif metric == 'f1':
            logging.info(f'Best dev Macro-F1 {best_dev_macro_f1:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test Macro-F1 {best_test_macro_f1:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        elif metric == 'auc':
            logging.info(f'Best dev AUC {best_dev_auc:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test AUC {best_test_auc:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        else:
            logging.info(f'Best dev UAR {best_dev_uar:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test UAR {best_test_uar:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        if self.best_attack_result_dict:
            for split, attack_dict in self.best_attack_result_dict.items():
                logging.info(
                    f"Best {split} ASR {attack_dict['asr']:.2f}%, "
                    f"Target-F1 {attack_dict['f1']:.2f}%, "
                    f"Target-Acc {attack_dict['acc']:.2f}%"
                )
                if 'paper_f1' in attack_dict and 'n_f1' in attack_dict and 'b_f1' in attack_dict:
                    logging.info(
                        f"Best {split} Paper Metrics: F1 {attack_dict['paper_f1']:.2f}%, "
                        f"N-F1 {attack_dict['n_f1']:.2f}%, B-F1 {attack_dict['b_f1']:.2f}%"
                    )
                if 'orig_acc' in attack_dict and 'orig_f1' in attack_dict:
                    logging.info(
                        f"Best {split} Original-Label F1 {attack_dict['orig_f1']:.2f}%, "
                        f"Original-Label Acc {attack_dict['orig_acc']:.2f}%"
                    )

    def summarize_results(self):
        row_df = pd.DataFrame(index=[f'fold{self.fold_idx}'])
        row_df['acc']  = self.best_test_dict['acc']
        if not self.multilabel:
            row_df['top5_acc'] = self.best_test_dict['top5_acc']
            row_df['uar'] = self.best_test_dict['uar']
        else:
            row_df['macro_f'] = self.best_test_dict['macro_f']
        return row_df

    def summarize_dict_results(self):
        result = dict()
        result['acc'] = self.best_test_dict['acc']
        if not self.multilabel:
            result['top5_acc'] = self.best_test_dict['top5_acc']
            result['uar'] = self.best_test_dict['uar']
            result['clean_f1'] = self.best_test_dict['f1']
            result['f1'] = self.best_test_dict['f1']
            if 'pred_majority_class' in self.best_test_dict:
                result['test_pred_majority_class'] = self.best_test_dict['pred_majority_class']
                result['test_pred_majority_ratio'] = self.best_test_dict['pred_majority_ratio']
            if "auc" in self.best_test_dict: result['auc'] = self.best_test_dict['auc']
            for split, attack_dict in self.best_attack_result_dict.items():
                result[f'{split}_acc'] = attack_dict['acc']
                result[f'{split}_f1'] = attack_dict['f1']
                result[f'{split}_asr'] = attack_dict['asr']
                for metric_name in [
                    'CA',
                    'ASR_Joint',
                    'ASR_I',
                    'ASR_T',
                    'Conf_Joint',
                    'Conf_I',
                    'Conf_T',
                    'TMA_I',
                    'TMA_T',
                    'MBR',
                ]:
                    if metric_name in attack_dict:
                        result[f'{split}_{metric_name}'] = attack_dict[metric_name]
                if 'paper_f1' in attack_dict:
                    result[f'{split}_paper_f1'] = attack_dict['paper_f1']
                if 'n_f1' in attack_dict:
                    result[f'{split}_n_f1'] = attack_dict['n_f1']
                if 'b_f1' in attack_dict:
                    result[f'{split}_b_f1'] = attack_dict['b_f1']
                if 'pred_majority_class' in attack_dict:
                    result[f'{split}_pred_majority_class'] = attack_dict['pred_majority_class']
                    result[f'{split}_pred_majority_ratio'] = attack_dict['pred_majority_ratio']
                if 'orig_acc' in attack_dict:
                    result[f'{split}_orig_acc'] = attack_dict['orig_acc']
                if 'orig_f1' in attack_dict:
                    result[f'{split}_orig_f1'] = attack_dict['orig_f1']
                for artifact_name in [
                    'text_trigger',
                    'text_trigger_score',
                    'clip_pretrained_loaded',
                    'poisoned_sample_files',
                ]:
                    if artifact_name in attack_dict:
                        result[f'{split}_{artifact_name}'] = attack_dict[artifact_name]
            primary_attack_split = self._get_primary_attack_split()
            if primary_attack_split is not None:
                primary_attack_dict = self.best_attack_result_dict[primary_attack_split]
                result['attack_split'] = primary_attack_split
                result['attack_acc'] = primary_attack_dict['acc']
                result['attack_f1'] = primary_attack_dict['f1']
                result['asr'] = primary_attack_dict['asr']
                for artifact_name in [
                    'text_trigger',
                    'text_trigger_score',
                    'clip_pretrained_loaded',
                    'poisoned_sample_files',
                ]:
                    if artifact_name in primary_attack_dict:
                        result[artifact_name] = primary_attack_dict[artifact_name]
                if 'paper_f1' in primary_attack_dict:
                    result['f1'] = primary_attack_dict['paper_f1']
                    result['paper_f1'] = primary_attack_dict['paper_f1']
                if 'n_f1' in primary_attack_dict:
                    result['n_f1'] = primary_attack_dict['n_f1']
                if 'b_f1' in primary_attack_dict:
                    result['b_f1'] = primary_attack_dict['b_f1']
        else:
            result['macro_f'] = self.best_test_dict['macro_f']
        return result

    def average_weights(self):
        """
        Returns the average of the weights.
        """
        # there are no samples, return
        if len(self.num_samples_list) == 0: 
            return
        robust_aggregation = str(getattr(self.args, 'robust_aggregation', 'fedavg')).lower()
        if robust_aggregation != 'fedavg':
            self.average_weights_robust()
            return
        self.last_aggregation_accepted_indices = list(range(len(self.model_updates)))
        if self._pepsy_enabled() and self.epoch >= int(getattr(self.args, 'pepsy_num_fedavg_rounds', 0)):
            self.average_weights_pepsy()
            return
        total_num_samples = np.sum(self.num_samples_list)
        w_avg = copy.deepcopy(self.model_updates[0])

        # calculate weighted updates
        for key in w_avg.keys():
            if self.args.fed_alg == 'scaffold':
                w_avg[key] = torch.div(self.model_updates[0][key], len(self.model_updates))
            else:
                w_avg[key] = self.model_updates[0][key]*(self.num_samples_list[0]/total_num_samples)
        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                if self.args.fed_alg == 'scaffold':
                    w_avg[key] += torch.div(self.model_updates[i][key], len(self.model_updates))
                else:
                    w_avg[key] += torch.div(self.model_updates[i][key]*self.num_samples_list[i], total_num_samples)
        
        # server optimization or just load with weights
        if self.args.fed_alg == 'fed_opt':
            self.update_global(copy.deepcopy(w_avg))
        else:
            self.global_model.load_state_dict(copy.deepcopy(w_avg))

        # update global control if algorithm is scaffold
        if self.args.fed_alg == 'scaffold':
            self.update_server_control()

    def average_weights_robust(self):
        """Aggregate client model deltas with the configured robust rule."""
        if self.args.fed_alg == 'scaffold':
            raise ValueError(
                'Robust aggregation is not supported with scaffold control updates'
            )
        result = aggregate_state_dicts(
            args=self.args,
            client_states=self.model_updates,
            global_state=self.global_model.state_dict(),
            parameter_names={name for name, _ in self.global_model.named_parameters()},
            sample_counts=self.num_samples_list,
            round_index=self.epoch,
        )
        self.last_aggregation_accepted_indices = list(result.accepted_indices)
        metadata = copy.deepcopy(result.metadata)
        metadata['accepted_client_ids'] = [
            str(self.client_update_ids[index])
            for index in self.last_aggregation_accepted_indices
            if index < len(self.client_update_ids)
        ]
        self.result_dict[self.epoch]['aggregation'] = metadata

        if self.args.fed_alg == 'fed_opt':
            self.update_global(copy.deepcopy(result.state_dict))
        else:
            self.global_model.load_state_dict(copy.deepcopy(result.state_dict))
        if hasattr(self.global_model, 'reset_trained_prompts_checklist'):
            self.global_model.reset_trained_prompts_checklist()

        fallback = metadata.get('fallback')
        accepted = metadata.get('accepted_indices', self.last_aggregation_accepted_indices)
        logging.info(
            f"Robust aggregation {metadata.get('method')}: accepted {accepted}; "
            f"fallback={fallback}"
        )
        if bool(getattr(self.args, 'robust_log_details', True)):
            logging.info(f'Robust aggregation details: {metadata}')

    @staticmethod
    def _pepsy_key_modality(key):
        if key.startswith('base.img_proj.') or key.startswith('sim_projectors.0.'):
            return 'img'
        if key.startswith('base.text_rnn.') or key.startswith('sim_projectors.1.'):
            return 'text'
        return None

    def _pepsy_client_weight(self, update_idx, modality=None):
        sample_weight = float(self.num_samples_list[update_idx])
        if modality is None:
            return sample_weight
        train_results = self.result_dict.get(self.epoch, {}).get('train', [])
        if update_idx >= len(train_results):
            return sample_weight
        explicit_name = 'pepsy_has_img' if modality == 'img' else 'pepsy_has_text'
        if explicit_name in train_results[update_idx]:
            return sample_weight * max(float(train_results[update_idx].get(explicit_name, 0.0)), 0.0)
        stat_name = 'pepsy_present_img' if modality == 'img' else 'pepsy_present_text'
        present_ratio = float(train_results[update_idx].get(stat_name, 1.0))
        return sample_weight * max(present_ratio, 0.0)

    def average_weights_pepsy(self):
        """PEPSY aggregation with modality-aware weights for modality-specific modules."""
        total_num_samples = float(np.sum(self.num_samples_list))
        current_global = self.global_model.state_dict()
        w_avg = copy.deepcopy(self.model_updates[0])

        for key in w_avg.keys():
            if not torch.is_floating_point(w_avg[key]):
                continue
            modality = self._pepsy_key_modality(key)
            weights = [self._pepsy_client_weight(i, modality) for i in range(len(self.model_updates))]
            total_weight = float(np.sum(weights))
            if total_weight <= 0:
                if modality is not None:
                    w_avg[key] = copy.deepcopy(current_global[key])
                    continue
                weights = [float(x) for x in self.num_samples_list]
                total_weight = total_num_samples
            w_avg[key] = self.model_updates[0][key] * (weights[0] / total_weight)
            for i in range(1, len(self.model_updates)):
                w_avg[key] += self.model_updates[i][key] * (weights[i] / total_weight)

        if self.args.fed_alg == 'fed_opt':
            self.update_global(copy.deepcopy(w_avg))
        else:
            self.global_model.load_state_dict(copy.deepcopy(w_avg))
        if hasattr(self.global_model, 'reset_trained_prompts_checklist'):
            self.global_model.reset_trained_prompts_checklist()

        if self.args.fed_alg == 'scaffold':
            self.update_server_control()

    def aggregate_pmcmfl_prototypes(self):
        if not self._pmcmfl_enabled():
            return
        if not hasattr(self, 'pmcmfl_prototypes'):
            self.initialize_pmcmfl_prototypes()
        if len(getattr(self, 'prototype_updates', [])) == 0:
            return

        new_prototypes = {
            key: value.detach().cpu().clone()
            for key, value in self.pmcmfl_prototypes.items()
        }
        for key in ['img', 'text', 'fusion']:
            proto_shape = new_prototypes[key].shape
            proto_sum = torch.zeros(proto_shape, dtype=torch.float32)
            count_sum = torch.zeros(proto_shape[0], dtype=torch.float32)
            count_key = f'counts_{key}'
            accepted_indices = set(
                getattr(
                    self,
                    'last_aggregation_accepted_indices',
                    range(len(self.prototype_updates)),
                )
            )
            for update_idx, update in enumerate(self.prototype_updates):
                if update_idx not in accepted_indices:
                    continue
                if update is None or key not in update or count_key not in update:
                    continue
                proto = update[key].detach().cpu().float()
                counts = update[count_key].detach().cpu().float()
                proto_sum += proto * counts.unsqueeze(1)
                count_sum += counts
            valid = count_sum > 0
            if valid.any():
                new_prototypes[key][valid] = proto_sum[valid] / count_sum[valid].unsqueeze(1)
                new_prototypes[count_key][valid] = count_sum[valid]
        self.pmcmfl_prototypes = new_prototypes

    def update_server_control(self):
        # update server control
        total_num_samples = np.sum(self.num_samples_list)
        delta_avg = copy.deepcopy(self.delta_controls[0])
        new_control = copy.deepcopy(self.server_control)

        for key in delta_avg.keys():
            delta_avg[key] = torch.div(self.delta_controls[0][key], self.num_of_clients)
            
        for key in delta_avg.keys():
            for idx in range(1, len(self.delta_controls)):
                delta_avg[key] += torch.div(self.delta_controls[idx][key], self.num_of_clients)
            new_control[key] = new_control[key] - delta_avg[key]
        self.server_control = copy.deepcopy(new_control)
        
    def update_global(
        self, 
        update_weights
    ):
        # zero_grad
        self.global_optimizer.zero_grad()
        global_optimizer_state = self.global_optimizer.state_dict()

        # new_model
        new_model = copy.deepcopy(self.global_model)
        new_model.load_state_dict(update_weights, strict=True)

        # set global_model gradient
        with torch.no_grad():
            for param, new_param in zip(
                self.global_model.parameters(), new_model.parameters()
            ):
                param.grad = (param.data - new_param.data) / self.args.learning_rate

        # replace some non-parameters's state dict
        state_dict = self.global_model.state_dict()
        for name in dict(self.global_model.named_parameters()).keys():
            update_weights[name] = state_dict[name]
        self.global_model.load_state_dict(update_weights, strict=True)

        # optimization
        self.global_optimizer = self._initialize_global_optimizer()
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()

    def set_save_json_file(
        self, 
        file_path
    ):
        self.save_json_path = file_path


    def save_json_file(
        self, 
        data_dict, 
        data_path
    ):
        jsonString = json.dumps(
            data_dict, 
            indent=4
        )
        jsonFile = open(str(data_path), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    # scaffold init contral
    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).cpu() for name, p in model.state_dict().items()
        }
        return control

    def set_control_device(
        self, 
        control, 
        device=True
    ):
        for name in control.keys():
            if device == True:
                control[name] = control[name].to(self.device)
            else:
                control[name] = control[name].cpu()
