import json
import glob
import torch
import pickle
import random
import copy
import pdb, os
import torchaudio
import numpy as np
import sys
import os.path as osp
import logging
# import pickle5 as pickle

from attacks.three_path import THREE_PATH_ATTACKS, normalize_attack_type, load_sparse_features

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset

try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
except Exception:
    pass

def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

def collate_mm_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None: max_b_len = max(map(lambda x: x[1].shape[0], batch))

    # pad according to max_len
    x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
    keys = list() if len(batch[0]) == 6 else None
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        
        len_a.append(torch.tensor(batch[idx][2]))
        len_b.append(torch.tensor(batch[idx][3]))

        ys.append(batch[idx][4])
        if keys is not None:
            keys.append(batch[idx][5])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)
    if keys is not None:
        return x_a, x_b, len_a, len_b, ys, keys
    return x_a, x_b, len_a, len_b, ys

def collate_unimodal_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    
    # pad according to max_len
    x_a, len_a, ys = list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        len_a.append(torch.tensor(batch[idx][1]))
        ys.append(batch[idx][-1])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    len_a = torch.stack(len_a, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, len_a, ys


class MMDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA, 
        modalityB, 
        default_feat_shape_a,
        default_feat_shape_b,
        data_len: int, 
        simulate_feat=None,
        dataset: str='',
        return_key: bool=False
    ):
        
        self.data_len = data_len
        
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.simulate_feat = simulate_feat
        
        self.default_feat_shape_a = default_feat_shape_a
        self.default_feat_shape_b = default_feat_shape_b
        self.dataset = dataset
        self.return_key = return_key
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        
        # modality A, if missing replace with 0s, and mask
        if data_a is not None: 
            if len(data_a.shape) == 3: data_a = data_a[0]
            data_a = torch.tensor(data_a)
            len_a = len(data_a)
        else: 
            data_a = torch.tensor(np.zeros(self.default_feat_shape_a))
            len_a = 0

        # modality B, if missing replace with 0s
        if data_b is not None:
            if len(data_b.shape) == 3: data_b = data_b[0]
            data_b = torch.tensor(data_b)
            len_b = len(data_b)
        else: 
            data_b = torch.tensor(np.zeros(self.default_feat_shape_b))
            len_b = 0
        if self.return_key:
            return data_a, data_b, len_a, len_b, label, self.modalityA[item][0]
        return data_a, data_b, len_a, len_b, label



class UniModalDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA,
        data_len: int, 
        simulate_feat=None,
        dataset: str=''
    ):
        self.dataset = dataset
        self.data_len = data_len
        self.modalityA = modalityA
        self.simulate_feat = simulate_feat
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        data_a = torch.tensor(data_a)
        len_a = len(data_a)
        return data_a, len_a, label


class DataloadManager():
    def __init__(
        self, 
        args: dict
    ):
        self.args = args
        self.args.attack_type = normalize_attack_type(getattr(args, "attack_type", "badnet"))
        self.label_dist_dict = dict()
        self.attack_metadata = None
        self.badnet_metadata = None
        self.backdoor_client_ids = set()
        self.badnet_rng = random.Random(getattr(self.args, "bd_seed", 2024))
        # Initialize video feature paths
        if self.args.dataset in ['ucf101', 'mit10', 'mit51', 'mit101', 'crema_d', "ego4d-ttm"]:
            self.get_video_feat_path()
        if self.args.dataset in ['hateful_memes', 'crisis-mmd', 'food101', 'upmc_food101']:
            self.get_image_feat_path()
            self.get_text_feat_path()
        # Initialize audio feature paths
        if self.args.dataset in ['ucf101', 'mit10', 'mit51', 'mit101', 'meld', 'crema_d', "ego4d-ttm"]:
            self.get_audio_feat_path()
        # Initialize acc/gyro feature paths
        if self.args.dataset in ['uci-har', 'extrasensory', "ku-har"]:
            self.get_acc_feat_path()
            self.get_gyro_feat_path()
        # Initialize acc/watch_acc feature paths
        if self.args.dataset in ['extrasensory_watch']:
            self.get_acc_feat_path()
            self.get_watch_acc_feat_path()
        # Initialize ptb feature paths
        if self.args.dataset in ['ptb-xl']:
            self.i_to_avf_path = Path(self.args.data_dir).joinpath(
                'feature', 
                'I_to_AVF', 
                args.dataset
            )
            self.v1_to_v6_path = Path(self.args.data_dir).joinpath(
                'feature', 
                'V1_to_V6', 
                args.dataset
            )
        if self._attack_eval_enabled() and self.args.dataset in ["crisis-mmd", "hateful_memes", "food101", "upmc_food101"]:
            self._load_badnet_metadata()

    def _backdoor_enabled(self):
        return bool(getattr(self.args, "backdoor_attack", False))

    def _attack_eval_enabled(self):
        return bool(
            getattr(self.args, "backdoor_attack", False)
            or getattr(self.args, "eval_attack", False)
        )

    def _get_attack_root(self):
        alpha_str = str(self.args.alpha).replace('.', '')
        mr = str(getattr(self.args, "bd_malicious_ratio", 0.2)).replace('.', '')
        pr = str(getattr(self.args, "bd_poison_ratio", 0.2)).replace('.', '')
        target_label = int(getattr(self.args, "bd_target_label", 0))
        attack_type = str(getattr(self.args, "attack_type", "badnet"))
        if attack_type in ['dab_mfl', 'bad_mfl', 'badcm', 'bad_mfl_3p', 'badcm_3p', 'toxictextclip', 'titim']:
            seed = int(getattr(self.args, "bd_seed", 2024))
            config_str = f'mr{mr}_pr{pr}_tl{target_label}_seed{seed}'
        else:
            trigger_size = int(getattr(self.args, "bd_trigger_size", 24))
            config_str = f'mr{mr}_pr{pr}_tl{target_label}_ts{trigger_size}'
        root_parts = [
            'attacks',
            attack_type,
            self.args.dataset,
            f'alpha{alpha_str}',
        ]
        q_dependent_attack_root = (
            (
                attack_type == 'attack_test2'
                and bool(getattr(self.args, 'attack_test2_q_dependent_prepare', True))
            )
            or (
                attack_type in ('bad_mfl', 'bad_mfl_3p')
                and bool(getattr(self.args, 'bad_mfl_q_dependent_prepare', True))
            )
            or (
                attack_type in ('badcm', 'badcm_3p')
                and bool(getattr(self.args, 'badcm_q_dependent_prepare', True))
            )
            or (
                attack_type == 'toxictextclip'
                and bool(getattr(self.args, 'toxictextclip_q_dependent_prepare', True))
            )
            or (
                attack_type == 'titim'
                and bool(getattr(self.args, 'titim_q_dependent_prepare', True))
            )
        )
        if (q_dependent_attack_root and bool(getattr(self.args, 'missing_modality', False))
                and (attack_type not in THREE_PATH_ATTACKS or float(getattr(self.args, 'missing_modailty_rate', 0.0)) > 0)):
            q_tag = str(getattr(self.args, "missing_modailty_rate", 0.0)).replace('.', '')
            root_parts.append(f'mm{q_tag}')
        root_parts.append(config_str)
        return Path(self.args.data_dir).joinpath(*root_parts)

    def _load_badnet_metadata(self):
        meta_path = self._get_attack_root().joinpath('metadata.json')
        if not Path.exists(meta_path):
            if self.args.attack_type in THREE_PATH_ATTACKS:
                raise FileNotFoundError(f"Complete 3P metadata missing: {meta_path}; run raw then extract.")
            logging.warning("[Attack] metadata not found at %s", meta_path)
            self.attack_metadata = None
            self.badnet_metadata = None
            return
        with open(str(meta_path), "r", encoding="utf-8") as f:
            self.attack_metadata = json.load(f)
        if self.args.attack_type in THREE_PATH_ATTACKS:
            if self.attack_metadata.get("status") != "complete" or self.attack_metadata.get("raw_only", True):
                raise ValueError("Three-path artifacts have not completed extraction.")
        self.badnet_metadata = self.attack_metadata
        self.backdoor_client_ids = set([str(cid) for cid in self.attack_metadata.get('malicious_clients', [])])
        logging.info(
            "[Attack] loaded metadata: attack_type=%s, %s malicious clients, target_label=%s",
            str(getattr(self.args, 'attack_type', 'badnet')),
            len(self.backdoor_client_ids),
            int(self.attack_metadata.get('target_label', getattr(self.args, 'bd_target_label', 0)))
        )

    def is_malicious_client(self, client_id: str) -> bool:
        return self._backdoor_enabled() and client_id in self.backdoor_client_ids

    def get_poisoned_keys(self, client_id: str):
        if self.attack_metadata is None:
            return set()
        return set([str(key) for key in self.attack_metadata.get('poisoned_keys', {}).get(str(client_id), [])])

    def get_attack_test_keys(self):
        if self.attack_metadata is None:
            return set()
        return set([str(key) for key in self.attack_metadata.get('attack_test_keys', [])])

    def _select_backdoor_clients(self):
        if not self._backdoor_enabled():
            self.backdoor_client_ids = set()
            return
        if self.badnet_metadata is not None:
            self.backdoor_client_ids = set([str(cid) for cid in self.badnet_metadata.get('malicious_clients', [])])
            return
        benign_clients = [cid for cid in self.client_ids if cid not in ["dev", "test"]]
        if len(benign_clients) == 0:
            self.backdoor_client_ids = set()
            return
        malicious_ratio = float(getattr(self.args, "bd_malicious_ratio", 0.2))
        malicious_ratio = min(max(malicious_ratio, 0.0), 1.0)
        num_malicious = int(round(len(benign_clients) * malicious_ratio))
        if malicious_ratio > 0 and num_malicious == 0:
            num_malicious = 1
        selected = self.badnet_rng.sample(benign_clients, num_malicious) if num_malicious > 0 else []
        self.backdoor_client_ids = set(selected)
        logging.info(
            "[Attack] enabled: %s malicious clients (%s / %s), attack_type=%s, target_label=%s, poison_ratio=%.3f",
            sorted(list(self.backdoor_client_ids)),
            len(self.backdoor_client_ids),
            len(benign_clients),
            str(getattr(self.args, "attack_type", "badnet")),
            int(getattr(self.args, "bd_target_label", 0)),
            float(getattr(self.args, "bd_poison_ratio", 0.2))
        )

    def _apply_feature_trigger(self, feat):
        if feat is None:
            return feat
        trigger_value = float(getattr(self.args, "bd_trigger_value", 10.0))
        trigger_size = max(1, int(getattr(self.args, "bd_trigger_size", 16)))

        if torch.is_tensor(feat):
            dim = feat.shape[-1]
            patch = min(trigger_size, dim)
            feat[..., dim - patch:] = trigger_value
            return feat

        if isinstance(feat, np.ndarray):
            dim = feat.shape[-1]
            patch = min(trigger_size, dim)
            feat[..., dim - patch:] = trigger_value
            return feat

        # fallback for list-like feature
        arr = np.asarray(feat)
        if arr.size == 0:
            return feat
        dim = arr.shape[-1]
        patch = min(trigger_size, dim)
        arr[..., dim - patch:] = trigger_value
        return arr

    def _apply_badnet_poison(self, data_a, data_b, client_id: str, shuffle: bool):
        return data_a, data_b

    def _build_mm_dataloader(
            self,
            data_a: dict,
            data_b: dict,
            default_feat_shape_a: np.array=np.array([0, 0]),
            default_feat_shape_b: np.array=np.array([0, 0]),
            shuffle: bool=False,
            return_key: bool=False
        ) -> (DataLoader):
        if len(data_a) == 0:
            return None
        data_ab = MMDatasetGenerator(
            modalityA=data_a,
            modalityB=data_b,
            default_feat_shape_a=default_feat_shape_a,
            default_feat_shape_b=default_feat_shape_b,
            data_len=len(data_a),
            dataset=self.args.dataset,
            return_key=return_key
        )
        if shuffle:
            return DataLoader(
                data_ab,
                batch_size=int(self.args.batch_size),
                num_workers=0,
                shuffle=shuffle,
                collate_fn=collate_mm_fn_padd
            )
        return DataLoader(
            data_ab,
            batch_size=64,
            num_workers=0,
            shuffle=shuffle,
            collate_fn=collate_mm_fn_padd
        )

    def set_attack_dataloader(
            self,
            data_a: dict,
            data_b: dict,
            default_feat_shape_a: np.array=np.array([0, 0]),
            default_feat_shape_b: np.array=np.array([0, 0]),
            attack_only_non_target: bool=True,
            mask_mode: str=None
        ) -> (DataLoader):
        data_a = copy.deepcopy(data_a)
        data_b = copy.deepcopy(data_b)
        attack_keys = self.get_attack_test_keys() if attack_only_non_target else set()
        if self.args.attack_type in THREE_PATH_ATTACKS or (self.attack_metadata or {}).get('artifact_format')=='sparse_poison_overlay_v1':
            if attack_only_non_target:
                data_a = [item for item in data_a if str(item[0]) in attack_keys]
            by_key_b = {str(item[0]): item for item in data_b}
            if any(str(item[0]) not in by_key_b for item in data_a):
                raise ValueError('Three-path attack evaluation is missing a paired modality key.')
            data_b = [by_key_b[str(item[0])] for item in data_a]
        elif len(attack_keys) > 0:
            data_a = [item for item in data_a if str(item[0]) in attack_keys]
            data_b = [item for item in data_b if str(item[0]) in attack_keys]
        if mask_mode == 'text':
            for item in data_b:
                item[-1] = None
        elif mask_mode == 'image':
            for item in data_a:
                item[-1] = None
        return self._build_mm_dataloader(
            data_a=data_a,
            data_b=data_b,
            default_feat_shape_a=default_feat_shape_a,
            default_feat_shape_b=default_feat_shape_b,
            shuffle=False
        )
            
    def get_audio_feat_path(self):
        """
        Load audio feature path.
        """
        self.audio_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'audio', 
            self.args.audio_feat, 
            self.args.dataset
        )
        return Path(self.audio_feat_path)
    
    def get_video_feat_path(self):
        """
        Load frame-wise video feature path.
        """
        self.video_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'video', 
            self.args.video_feat, 
            self.args.dataset
        )
        return Path(self.video_feat_path)
    
    def get_image_feat_path(self):
        """
        Load image feature path.
        """
        self.img_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'img', 
            self.args.img_feat, 
            self.args.dataset
        )
        return Path(self.img_feat_path)

    def get_text_feat_path(self):
        """
        Load text feature path.
        """
        self.text_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'text', 
            self.args.text_feat, 
            self.args.dataset
        )
        return Path(self.text_feat_path)
    
    def get_acc_feat_path(self):
        """
        Load acc feature path.
        """
        self.acc_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'acc', 
            self.args.dataset
        )
        return Path(self.acc_feat_path)
    
    def get_watch_acc_feat_path(self):
        """
        Load watch acc feature path.
        """
        self.watch_acc_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'watch_acc', 
            self.args.dataset
        )
        return Path(self.watch_acc_feat_path)
    
    def get_gyro_feat_path(self):
        """
        Load gyro feature path.
        """
        self.gyro_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'gyro', 
            self.args.dataset
        )
        return Path(self.gyro_feat_path)

    def get_client_ids(
            self, 
            fold_idx: int=1
        ):
        """
        Load client ids.
        :param fold_idx: fold index
        :return: None
        """
        if self.args.dataset in ["mit10", "mit51"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}'
            )
        elif self.args.dataset in ["hateful_memes", "crisis-mmd", "food101", "upmc_food101"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.img_feat_path.joinpath(
                f'alpha{alpha_str}'
            )
        elif self.args.dataset in ["crema_d"]:
            data_path = self.video_feat_path.joinpath(
                f'fold{fold_idx}'
            )
        elif self.args.dataset in ["uci-har"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.acc_feat_path.joinpath(
                f'alpha{alpha_str}'
            )
        elif self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'fold{fold_idx}'
            )
        elif self.args.dataset == "ego4d-ttm":
            data_path = self.video_feat_path
        elif self.args.dataset in ["extrasensory", "extrasensory_watch", "ku-har"]:
            data_path = self.acc_feat_path.joinpath(
                f'fold{fold_idx}'
            )
        elif self.args.dataset == "meld":
            data_path = self.text_feat_path
        elif self.args.dataset == "ptb-xl":
            data_path = self.v1_to_v6_path
        self.client_ids = [id.split('.pkl')[0] for id in os.listdir(str(data_path)) if id.endswith('.pkl')]
        self.client_ids.sort()
        self._select_backdoor_clients()
        
    def load_audio_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load audio feature data different applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset == "crema_d":
            data_path = self.audio_feat_path.joinpath(
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["mit10", "mit51"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["meld", "ego4d-ttm"]:
            data_path = self.audio_feat_path.joinpath(f'{client_id}.pkl')
        
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    def load_video_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load frame-wise video feature data for MMaction applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'fold{fold_idx}',  
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["crema_d"]:
            data_path = self.video_feat_path.joinpath(
                f'fold{fold_idx}',  
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["ego4d-ttm"]:
            data_path = self.video_feat_path.joinpath(
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["mit10", "mit51"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    def load_img_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load image feature data for image applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if (self.args.attack_type in THREE_PATH_ATTACKS or (self.attack_metadata or {}).get('artifact_format')=='sparse_poison_overlay_v1') and self.is_malicious_client(client_id):
            return load_sparse_features(self, client_id, 'img')
        if self.args.dataset in ["hateful_memes", "crisis-mmd", "food101", "upmc_food101"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            if self.is_malicious_client(client_id):
                poison_path = self._get_attack_root().joinpath('poisoned_img', f'{client_id}.pkl')
                data_path = poison_path if Path.exists(poison_path) else self.img_feat_path.joinpath(
                    f'alpha{alpha_str}',
                    f'{client_id}.pkl'
                )
            else:
                data_path = self.img_feat_path.joinpath(
                    f'alpha{alpha_str}',
                    f'{client_id}.pkl'
                )
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    def load_acc_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load acc feature data for HAR applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "uci-har":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.acc_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["extrasensory", "extrasensory_watch", 'ku-har']:
            data_path = self.acc_feat_path.joinpath(
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict
    
    def load_watch_acc_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load watch-based acc feature data for HAR applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "extrasensory_watch":
            data_path = self.watch_acc_feat_path.joinpath(
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict
    
    def load_gyro_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load gyro data for HAR application.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "uci-har":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.gyro_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["extrasensory", "ku-har"]:
            data_path = self.gyro_feat_path.joinpath(
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict

    def load_text_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load text feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if (self.args.attack_type in THREE_PATH_ATTACKS or (self.attack_metadata or {}).get('artifact_format')=='sparse_poison_overlay_v1') and self.is_malicious_client(client_id):
            return load_sparse_features(self, client_id, 'text')
        if self.args.dataset == "meld":
            data_path = self.text_feat_path.joinpath(
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["hateful_memes", "crisis-mmd", "food101", "upmc_food101"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            if self.is_malicious_client(client_id):
                poison_path = self._get_attack_root().joinpath('poisoned_text', f'{client_id}.pkl')
                data_path = poison_path if Path.exists(poison_path) else self.text_feat_path.joinpath(
                    f'alpha{alpha_str}',
                    f'{client_id}.pkl'
                )
            else:
                data_path = self.text_feat_path.joinpath(
                    f'alpha{alpha_str}',
                    f'{client_id}.pkl'
                )
            
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict

    def load_attack_img_feat(
            self,
            client_id: str='test'
        ) -> dict:
        data_path = self._get_attack_root().joinpath('attack_img', f'{client_id}.pkl')
        with open(str(data_path), "rb") as f:
            data_dict = pickle.load(f)
        return data_dict

    def load_attack_text_feat(
            self,
            client_id: str='test'
        ) -> dict:
        data_path = self._get_attack_root().joinpath('attack_text', f'{client_id}.pkl')
        with open(str(data_path), "rb") as f:
            data_dict = pickle.load(f)
        return data_dict

    def load_ecg_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load ecg feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        i_to_avf_data_path = self.i_to_avf_path.joinpath(f'{client_id}.pkl')
        v1_to_v6_data_path = self.v1_to_v6_path.joinpath(f'{client_id}.pkl')
        with open(str(i_to_avf_data_path), "rb") as f:  i_to_avf_data_dict = pickle.load(f)
        with open(str(v1_to_v6_data_path), "rb") as f:  v1_to_v6_data_dict = pickle.load(f)
        return i_to_avf_data_dict, v1_to_v6_data_dict

    def get_client_sim_dict(
            self, 
            client_id
        ):
        """
        Set dataloader for training/dev/test.
        :param client_id: client_id
        :return: dataloader: torch dataloader
        """
        if self.sim_data:
            return self.sim_data[client_id]
        return None

    def get_label_dist(
        self, 
        data_dict,
        client_id: str
    ):
        """
        Set dataloader for training/dev/test.
        :param data_dict: data dictionary
        :return: data_dis_dict
        """
        label_list = list()
        for idx in range(len(data_dict)):
            label_list.append(data_dict[idx][-2])
        self.label_dist_dict[client_id] = Counter(label_list)
        
    def set_dataloader(
            self, 
            data_a: dict,
            data_b: dict,
            default_feat_shape_a: np.array=np.array([0, 0]),
            default_feat_shape_b: np.array=np.array([0, 0]),
            client_sim_dict: dict=None,
            shuffle: bool=False,
            client_id: str=None
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param data_b: modality B data
        :param default_feat_shape_a: default input shape for modality A, fill 0 in missing modality case
        :param default_feat_shape_b: default input shape for modality B, fill 0 in missing modality case
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :return: dataloader: torch dataloader
        """
        # modify data based on simulation
        labeled_data_idx, unlabeled_data_idx = list(), list()
        if client_sim_dict is not None:
            attack_type = str(getattr(self.args, 'attack_type', '')).lower()
            # Bad-MFL and attack_test1/2 follow the same missing-modality
            # simulation flow as benign clients for q-based experiments.
            skip_mm_for_client = (
                self.is_malicious_client(client_id)
                and attack_type in ['dab_mfl']
            )
            poisoned_keys_for_client = (
                self.get_poisoned_keys(client_id)
                if (
                    self.is_malicious_client(client_id)
                    and attack_type in ['bad_mfl', 'badcm', 'bad_mfl_3p', 'badcm_3p', 'toxictextclip', 'titim', 'attack_test1', 'attack_test2']
                )
                else set()
            )
            has_available_modality = False
            for idx in range(len(client_sim_dict)):
                # read simulate feature
                sim_data = client_sim_dict[idx][-1]
                sample_key = str(data_a[idx][0])
                if not (sim_data[0] == 1 and sim_data[1] == 1):
                    has_available_modality = True
                # read modality A
                if sim_data[0] == 1 and not skip_mm_for_client: data_a[idx][-1] = None
                # read modality B
                if sim_data[1] == 1 and not skip_mm_for_client: data_b[idx][-1] = None
                # Keep target labels written by poisoned-feature preparation.
                # The q simulation file stores the original/noisy label; applying
                # it blindly would train triggered samples back to their clean
                # class and suppress the attack signal.
                if sample_key not in poisoned_keys_for_client:
                    data_a[idx][-2] = sim_data[2]
                    data_b[idx][-2] = sim_data[2]
                else:
                    data_b[idx][-2] = data_a[idx][-2]
                # missing label
                if sim_data[-1] == 0: labeled_data_idx.append(idx)
                else: unlabeled_data_idx.append(idx)
            
            # return None only when every sample in this client misses both modalities
            if not has_available_modality and not skip_mm_for_client:
                return None
            
            labeled_data_a, unlabeled_data_a = list(), list()
            labeled_data_b, unlabeled_data_b = list(), list()
            if len(unlabeled_data_idx) > 0:
                for idx in labeled_data_idx:
                    labeled_data_a.append(data_a[idx])
                    labeled_data_b.append(data_b[idx])
                for idx in unlabeled_data_idx:
                    unlabeled_data_a.append(data_a[idx])
                    unlabeled_data_b.append(data_b[idx])
                data_a = labeled_data_a
                data_b = labeled_data_b

        return self._build_mm_dataloader(
            data_a=data_a,
            data_b=data_b,
            default_feat_shape_a=default_feat_shape_a,
            default_feat_shape_b=default_feat_shape_b,
            shuffle=shuffle,
            return_key=bool(
                shuffle
                and client_id is not None
                and self.is_malicious_client(client_id)
                and str(getattr(self.args, 'attack_type', '')).lower() in ['dab_mfl', 'bad_mfl', 'badcm', 'bad_mfl_3p', 'badcm_3p', 'toxictextclip', 'titim', 'attack_test1', 'attack_test2']
            )
        )

    def set_unimodal_dataloader(
            self, 
            data_a: dict,
            client_sim_dict: dict=None,
            shuffle: bool=False,
            client_id: str=None
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :return: dataloader: torch dataloader
        """
        # modify data based on simulation
        if client_sim_dict is not None:
            for idx in range(len(client_sim_dict)):
                # read simulate feature
                sim_data = client_sim_dict[idx][-1]
                # pdb.set_trace()
                # read modality A
                if sim_data[0] == 1: data_a[idx][-1] = None
                # read modality B
                if sim_data[1] == 1: data_b[idx][-1] = None
                # label noise
                data_a[idx][-2] = sim_data[2]
            
            # return None when both modalities are missing
            if sim_data[0] == 1 and sim_data[1] == 1:
                return None

        data = UniModalDatasetGenerator(
            data_a, 
            len(data_a),
            self.args.dataset
        )
        if shuffle:
            # we use args input batch size for train, typically set as 16 in FL setup
            dataloader = DataLoader(
                data, 
                batch_size=int(self.args.batch_size), 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_unimodal_fn_padd
            )
        else:
            # we use a larger batch size for validation and testing
            dataloader = DataLoader(
                data, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_unimodal_fn_padd
            )
        return dataloader
    
    def load_sim_dict(
        self, 
        fold_idx: int=1,
        ext: str="json"
    ):
        """
        Load simulation dictionary.
        :param fold_idx: fold index
        :return: None
        """
        if self.setting_str == '': 
            self.sim_data = None
            return
        
        if self.args.dataset in ["ucf101", "crema_d", 'ku-har']:
            data_path = Path(self.args.data_dir).joinpath(
                'simulation_feature',
                self.args.dataset, 
                f'fold{fold_idx}', 
                f'{self.setting_str}.{ext}'
            )
        elif self.args.dataset in ["mit10", "mit51", "meld", "uci-har", "ptb-xl", "crisis-mmd", "hateful_memes", "food101", "upmc_food101"]:
            data_path = Path(self.args.data_dir).joinpath(
                'simulation_feature',
                self.args.dataset,
                f'{self.setting_str}.{ext}'
            )
        if ext == "pkl":
            with open(str(data_path), "rb") as f: 
                self.sim_data = pickle.load(f)
        else:
            with open(str(data_path), "r") as f: 
                self.sim_data = json.load(f)
    
    def get_simulation_setting(self, alpha=None):
        """
        Load get simulation setting string.
        :param alpha: alpha in manual split
        :return: None
        """
        self.setting_str = ''
        # 1. missing modality
        if self.args.missing_modality == True:
            self.setting_str += 'mm'+str(self.args.missing_modailty_rate).replace('.', '')
        # 2. label nosiy
        if self.args.label_nosiy == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ln'+str(self.args.label_nosiy_level).replace('.', '')
        # 3. missing labels
        if self.args.missing_label == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ml'+str(self.args.missing_label_rate).replace('.', '')
        # 4. alpha for manual split
        if len(self.setting_str) != 0:
            if alpha is not None:
                alpha_str = str(self.args.alpha).replace('.', '')
                self.setting_str += f'_alpha{alpha_str}'
