import torch
import random
import numpy as np
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import constants
from trainers.server_trainer import Server
from trainers import clip_badcm3p_crisis_audit
from trainers.clip_clean_audit import add_standard_deviations as clip_sample_std
from model.mm_models import ImageTextClassifier
from model.pepsy_models import PepsyImageTextClassifier
from dataloader.dataload_manager import DataloadManager
from defenses.aggregation.cli import add_robust_aggregation_args
from defenses.post_training import add_post_training_defense_args, run_post_training_defense
try:
    from attacks.dab_mfl.trainer import ClientDABMFL
except ModuleNotFoundError as exc:
    if exc.name not in ('attacks.dab_mfl', 'attacks.dab_mfl.trainer'):
        raise
    ClientDABMFL = None
from attacks.bad_mfl.trainer import ClientBadMFL
from attacks.badcm.trainer import ClientBadCM
from attacks.badcm_3p.trainer import ClientBadCM3P
from attacks.bad_mfl_3p.trainer import ClientBadMFL3P
from attacks.three_path import (THREE_PATH_ATTACKS, configure_three_path,
    record_three_path_round, add_fold_standard_deviations)
from attacks.toxictextclip.trainer import ClientToxicTextCLIP
from attacks.toxictextclip.evaluate import summarize_toxictextclip_metrics
from attacks.titim.trainer import ClientTITIM
from attacks.titim.evaluate import summarize_titim_metrics
try:
    from attacks.attack_test1.Evaluate import compute_attack_test1_metrics
    from attacks.attack_test1.LocalUpdate import ClientAttackTest1
except (ImportError, ModuleNotFoundError):
    compute_attack_test1_metrics = None
    ClientAttackTest1 = None
from attacks.attack_test2.Evaluate import compute_attack_test2_metrics
from attacks.attack_test2.LocalUpdate import ClientAttackTest2
from attacks.attack_test2.prepare_raw_trigger import add_raw_prepare_args, ensure_prepared_from_args

from trainers.fed_rs_trainer import ClientFedRS
from trainers.fed_avg_trainer import ClientFedAvg
from trainers.pmcmfl_trainer import ClientPmcmFL
from trainers.pepsy_trainer import ClientPepsy
from trainers.scaffold_trainer import ClientScaffold

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_attack_target_prior(args, label_counter):
    if not (getattr(args, 'backdoor_attack', False) or getattr(args, 'eval_attack', False)):
        return
    if label_counter is None or len(label_counter) == 0:
        return
    total = int(sum(label_counter.values()))
    if total <= 0:
        return
    target_label = int(getattr(args, 'bd_target_label', 0))
    target_count = int(label_counter.get(target_label, 0))
    majority_label, majority_count = max(label_counter.items(), key=lambda item: item[1])
    logging.info(
        '[Attack Eval] target_label=%s clean-test prior %.2f%% (%s/%s); majority_label=%s prior %.2f%%',
        target_label,
        100.0 * target_count / total,
        target_count,
        total,
        int(majority_label),
        100.0 * int(majority_count) / total
    )
    if int(majority_label) == target_label:
        logging.warning(
            '[Attack Eval] target_label matches the clean-test majority class. '
            'ASR can be inflated by class-collapse and must be compared against the clean baseline.'
        )


def evaluate_attack_server(args, server, test_loader, device, artifacts=None, attack_name='attack_test1', compute_fn=None):
    artifacts = artifacts or dict()
    compute_fn = compute_fn or compute_attack_test1_metrics
    if compute_fn is None:
        raise ModuleNotFoundError(
            'attack_test1 evaluation requires attacks/attack_test1, but that package is missing.'
        )
    result = compute_fn(
        model=server.global_model,
        clean_dataloader=test_loader,
        target_label=int(args.bd_target_label),
        device=device,
        trigger_size=int(getattr(args, 'attack_test1_trigger_size', 16)),
        trigger_value=float(getattr(args, 'attack_test1_trigger_value', 10.0)),
        image_trigger=artifacts.get('image_trigger'),
        image_trigger_std=float(getattr(args, 'attack_test1_img_trigger_std', 0.05)),
        image_trigger_eps=float(getattr(args, 'attack_test1_img_trigger_eps', 0.5)),
        text_trigger=artifacts.get('text_trigger'),
        text_charset=str(getattr(args, 'attack_test1_text_charset', 'abcdefghijklmnopqrstuvwxyz0123456789')),
        text_trigger_budget=int(getattr(args, 'attack_test1_text_trigger_budget', 5)),
        text_trigger_scale=float(getattr(args, 'attack_test1_text_trigger_scale', 0.1)),
    )
    result.update({
        'acc': result.get('CA', 0.0) * 100.0,
        'f1': result.get('ASR_Joint', 0.0) * 100.0,
        'loss': 0.0,
        'asr': result.get('ASR_Joint', 0.0) * 100.0,
        'asr_sample': int(result.get('attack_total', 0)),
        'text_trigger': artifacts.get('text_trigger'),
        'text_trigger_score': artifacts.get('text_trigger_score'),
        'clip_pretrained_loaded': artifacts.get('clip_pretrained_loaded'),
        'poisoned_sample_files': artifacts.get('poisoned_sample_files', []),
    })
    logging.info(
        '[%s] CA %.2f%% | ASR_Joint %.2f%% | ASR_I %.2f%% | ASR_T %.2f%% | MBR %.4f',
        attack_name,
        result.get('CA', 0.0) * 100.0,
        result.get('ASR_Joint', 0.0) * 100.0,
        result.get('ASR_I', 0.0) * 100.0,
        result.get('ASR_T', 0.0) * 100.0,
        result.get('MBR', 0.0),
    )
    return result


def evaluate_attack_test1_server(args, server, test_loader, device, artifacts=None):
    if compute_attack_test1_metrics is None:
        raise ModuleNotFoundError(
            'attack_type=attack_test1 requires attacks/attack_test1, but that package is missing.'
        )
    return evaluate_attack_server(
        args=args,
        server=server,
        test_loader=test_loader,
        device=device,
        artifacts=artifacts,
        attack_name='attack_test1',
        compute_fn=compute_attack_test1_metrics,
    )


def evaluate_attack_test2_server(args, server, test_loader, device, artifacts=None):
    return evaluate_attack_server(
        args=args,
        server=server,
        test_loader=test_loader,
        device=device,
        artifacts=artifacts,
        attack_name='attack_test2',
        compute_fn=compute_attack_test2_metrics,
    )


def summarize_triplet_attack_metrics(clean_result, attack_results):
    joint = attack_results.get('attack_test_mm', dict())
    image_only = attack_results.get('attack_test_img_only', dict())
    text_only = attack_results.get('attack_test_text_only', dict())

    ca = float(clean_result.get('acc', 0.0)) / 100.0
    asr_joint = float(joint.get('asr', 0.0)) / 100.0
    asr_i = float(image_only.get('asr', 0.0)) / 100.0
    asr_t = float(text_only.get('asr', 0.0)) / 100.0
    conf_joint = float(joint.get('target_conf', 0.0))
    conf_i = float(image_only.get('target_conf', 0.0))
    conf_t = float(text_only.get('target_conf', 0.0))
    tma_i = conf_joint - conf_t
    tma_t = conf_joint - conf_i
    denom = max(tma_i, tma_t)
    mbr = float(min(tma_i, tma_t) / denom) if denom > 1e-12 else 0.0

    result = {
        'loss': 0.0,
        'acc': ca * 100.0,
        'f1': asr_joint * 100.0,
        'asr': asr_joint * 100.0,
        'asr_sample': int(joint.get('asr_sample', 0)),
        'CA': ca,
        'ASR_Joint': asr_joint,
        'ASR_I': asr_i,
        'ASR_T': asr_t,
        'Conf_Joint': conf_joint,
        'Conf_I': conf_i,
        'Conf_T': conf_t,
        'TMA_I': float(tma_i),
        'TMA_T': float(tma_t),
        'MBR': mbr,
    }
    for metric_name in ['paper_f1', 'n_f1', 'b_f1', 'orig_acc', 'orig_f1']:
        if metric_name in joint:
            result[metric_name] = joint[metric_name]
    return result


def parse_args():
    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
            
    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))

    parser = argparse.ArgumentParser(
        description='FedMultimodal experiments'
    )
    
    parser.add_argument(
        '--data_dir', 
        default=path_conf['output_dir'],
        type=str, 
        help='output feature directory'
    )
    
    parser.add_argument(
        '--text_feat', 
        default='mobilebert',
        type=str,
        help="text feature name",
    )
    
    parser.add_argument(
        '--img_feat', 
        default='mobilenet_v2',
        type=str,
        help="image feature name",
    )

    parser.add_argument(
        '--att', 
        type=bool, 
        default=False,
        help='self attention applied or not'
    )
    
    parser.add_argument(
        "--en_att",
        dest='att',
        action='store_true',
        help="enable self-attention"
    )

    parser.add_argument(
        '--att_name',
        type=str, 
        default='multihead',
        help='attention name'
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.1,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--global_learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--sample_rate', 
        default=0.1,
        type=float,
        help="client sample rate",
    )
    
    parser.add_argument(
        '--num_epochs', 
        default=300,
        type=int,
        help="total training rounds",
    )

    parser.add_argument(
        '--num_folds',
        default=5,
        type=int,
        choices=range(1, 6),
        help="number of sequential folds to run (starting from fold 1)",
    )
    
    parser.add_argument(
        '--mu',
        type=float, 
        default=0.001,
        help='Fed prox term'
    )

    parser.add_argument(
        '--hid_size',
        type=int, 
        default=64,
        help='RNN hidden size dim'
    )

    parser.add_argument(
        '--dropout_p',
        type=float,
        default=0.1,
        help='dropout probability for multimodal classifier'
    )

    parser.add_argument(
        '--d_head',
        type=int,
        default=6,
        help='number of fusion attention heads'
    )

    parser.add_argument(
        '--test_frequency', 
        default=1,
        type=int,
        help="perform test frequency",
    )
    
    parser.add_argument(
        '--local_epochs', 
        default=1,
        type=int,
        help="local epochs",
    )
    
    parser.add_argument(
        '--optimizer', 
        default='sgd',
        type=str,
        help="optimizer",
    )
    
    parser.add_argument(
        '--fed_alg', 
        default='fed_avg',
        type=str,
        help="federated learning aggregation algorithm",
    )
    
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help="training batch size",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        "--missing_modality",
        type=bool, 
        default=False,
        help="missing modality simulation",
    )
    
    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )
    
    parser.add_argument(
        "--missing_modailty_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )

    parser.add_argument(
        "--missing_strategy",
        type=str,
        default="zero",
        choices=["zero", "pmcmfl", "pepsy"],
        help="strategy for missing modality: zero baseline, PmcmFL prototype mask/contrast, or PEPSY reconfigurable representations"
    )

    parser.add_argument('--pmcmfl_prototype_as_missing_modal', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pmcmfl_prototype_as_rep_target', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pmcmfl_regularization_start_epoch', type=int, default=2)
    parser.add_argument('--pmcmfl_temperature', type=float, default=0.5)
    parser.add_argument('--pmcmfl_img_loss_weight', type=float, default=1e-8)
    parser.add_argument('--pmcmfl_text_loss_weight', type=float, default=1e-8)
    parser.add_argument('--pmcmfl_fusion_loss_weight', type=float, default=0.01)
    parser.add_argument('--pmcmfl_prototype_init_std', type=float, default=0.0)

    parser.add_argument('--pepsy_pool_size', type=int, default=10)
    parser.add_argument('--pepsy_top_k', type=int, default=3)
    parser.add_argument('--pepsy_contrastive_weight', type=float, default=0.2)
    parser.add_argument('--pepsy_prompt_reg_weight', type=float, default=0.1)
    parser.add_argument('--pepsy_temperature', type=float, default=0.2)
    parser.add_argument('--pepsy_prompt_dropout', type=float, default=0.0)
    parser.add_argument('--pepsy_num_fedavg_rounds', type=int, default=0)
    
    parser.add_argument(
        "--missing_label",
        type=bool, 
        default=False,
        help="missing label simulation",
    )
    
    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )
    
    parser.add_argument(
        "--missing_label_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        '--label_nosiy', 
        type=bool, 
        default=False,
        help='clean label or nosiy label')
    
    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )

    parser.add_argument(
        '--label_nosiy_level', 
        type=float, 
        default=0.1,
        help='nosiy level for labels; 0.9 means 90%% wrong'
    )

    parser.add_argument(
        '--modality', 
        type=str, 
        default='multimodal',
        help='modality type'
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="crisis-mmd",
        help='data set name'
    )

    parser.add_argument(
        "--backdoor_attack",
        type=bool,
        default=False,
        help="enable backdoor attack"
    )

    parser.add_argument(
        "--en_backdoor_attack",
        dest="backdoor_attack",
        action="store_true",
        help="enable backdoor attack",
    )

    parser.add_argument(
        "--eval_attack",
        type=bool,
        default=False,
        help="evaluate attack_test_* splits without enabling malicious local training"
    )

    parser.add_argument(
        "--en_eval_attack",
        dest="eval_attack",
        action="store_true",
        help="evaluate attack_test_* splits without enabling malicious local training",
    )

    parser.add_argument(
        "--attack_type",
        type=str,
        default="badnet",
        help="attack type: badnet, dab_mfl, bad_mfl, badcm, bad_mfl_3p, badcm_3p (aliases: bad-mfl-3path, badcm-3path), toxictextclip, titim, attack_test1, attack_test2"
    )

    parser.add_argument(
        "--bd_malicious_ratio",
        type=float,
        default=0.2,
        help="ratio of malicious clients"
    )

    parser.add_argument(
        "--bd_poison_ratio",
        type=float,
        default=0.2,
        help="ratio of poisoned samples in each malicious client"
    )

    parser.add_argument(
        "--bd_target_label",
        type=int,
        default=0,
        help="target label for badnet attack"
    )

    parser.add_argument(
        "--bd_trigger_value",
        type=float,
        default=10.0,
        help="trigger value filled into feature tail dims"
    )

    parser.add_argument(
        "--bd_trigger_size",
        type=int,
        default=16,
        help="trigger width in feature tail dims"
    )

    parser.add_argument(
        "--bd_trigger_modalities",
        type=str,
        default="both",
        help="modalities to add trigger: a, b, both"
    )

    parser.add_argument(
        "--bd_seed",
        type=int,
        default=2024,
        help="random seed for selecting malicious clients and poisoned samples"
    )

    parser.add_argument(
        "--dab_visual_trigger",
        type=str,
        default="retro",
        help="semantic visual trigger for DAB-MFL"
    )

    parser.add_argument(
        "--dab_visual_blend",
        type=float,
        default=0.72,
        help="blend ratio for DAB-MFL visual trigger"
    )

    parser.add_argument(
        "--dab_text_trigger",
        type=str,
        default="in a retro style",
        help="natural language trigger phrase for DAB-MFL"
    )

    parser.add_argument(
        "--dab_align_lambda",
        type=float,
        default=0.05,
        help="alignment loss weight for DAB-MFL"
    )

    parser.add_argument(
        "--dab_attn_lambda",
        type=float,
        default=0.02,
        help="attention guidance loss weight for DAB-MFL"
    )

    parser.add_argument(
        '--bad_mfl_clip_model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='CLIP model used in Bad-MFL prepare stage'
    )

    parser.add_argument(
        '--bad_mfl_t5_model',
        type=str,
        default='google/flan-t5-small',
        help='T5 model used in Bad-MFL text trigger generation'
    )

    parser.add_argument(
        '--bad_mfl_use_t5',
        action='store_true',
        help='enable T5 paraphrasing for Bad-MFL text trigger generation'
    )

    parser.add_argument(
        '--bad_mfl_allow_remote_model_download',
        action='store_true',
        help='allow transformers models to download from Hugging Face during prepare'
    )

    parser.add_argument(
        '--bad_mfl_t5_candidates',
        type=int,
        default=4,
        help='number of T5 paraphrase candidates for Bad-MFL'
    )

    parser.add_argument(
        '--bad_mfl_attack_weight',
        type=float,
        default=-1.0,
        help='loss weight assigned to poisoned samples on malicious clients; <=0 uses bd_poison_ratio'
    )

    parser.add_argument(
        '--bad_mfl_q_dependent_prepare',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='use missing-rate specific Bad-MFL artifacts under mm{q}'
    )

    parser.add_argument(
        '--badcm_attack_weight',
        type=float,
        default=-1.0,
        help='loss weight for BadCM poisoned samples; <=0 uses bd_poison_ratio'
    )

    parser.add_argument(
        '--badcm_q_dependent_prepare',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='use missing-rate specific BadCM artifacts under mm{q}'
    )
    parser.add_argument(
        '--badcm_paper_dropout',
        type=float,
        default=0.2,
        help='dropout used by the BadCM framework adaptation'
    )
    parser.add_argument(
        '--badcm_paper_heads',
        type=int,
        default=6,
        help='fusion head count used by the BadCM framework adaptation'
    )
    parser.add_argument(
        '--toxictextclip_attack_weight',
        type=float,
        default=-1.0,
        help='loss weight assigned to ToxicTextCLIP poisoned samples; <=0 uses bd_poison_ratio'
    )
    parser.add_argument(
        '--toxictextclip_q_dependent_prepare',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='use missing-rate-specific ToxicTextCLIP artifacts under mm{q}'
    )
    parser.add_argument('--titim_attack_weight', type=float, default=-1.0)
    parser.add_argument(
        '--titim_q_dependent_prepare',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='use missing-rate-specific TITIM-BadNet artifacts under mm{q}'
    )

    parser.add_argument(
        '--bad_mfl_paper_dropout',
        type=float,
        default=0.2,
        help='paper-aligned dropout used for Crisis-MMD Bad-MFL experiments'
    )

    parser.add_argument(
        '--bad_mfl_paper_heads',
        type=int,
        default=6,
        help='paper-aligned fusion head count used for Crisis-MMD Bad-MFL experiments'
    )

    parser.add_argument('--attack_test1_lambda_main', type=float, default=1.0)
    parser.add_argument('--attack_test1_alpha', type=float, default=1.0)
    parser.add_argument('--attack_test1_beta', type=float, default=1.0)
    parser.add_argument('--attack_test1_gamma', type=float, default=1.0)
    parser.add_argument('--attack_test1_lambda_csd', type=float, default=1.0)
    parser.add_argument('--attack_test1_lambda_reg', type=float, default=1e-3)
    parser.add_argument('--attack_test1_delta_init_std', type=float, default=1e-5)
    parser.add_argument('--attack_test1_tma_interval', type=int, default=1)
    parser.add_argument('--attack_test1_trigger_size', type=int, default=16)
    parser.add_argument('--attack_test1_trigger_value', type=float, default=10.0)
    parser.add_argument('--attack_test1_img_trigger_std', type=float, default=0.05)
    parser.add_argument('--attack_test1_img_trigger_eps', type=float, default=0.5)
    parser.add_argument('--attack_test1_text_charset', type=str, default='abcdefghijklmnopqrstuvwxyz0123456789')
    parser.add_argument('--attack_test1_text_trigger_budget', type=int, default=5)
    parser.add_argument('--attack_test1_text_beam_width', type=int, default=5)
    parser.add_argument('--attack_test1_text_trigger_scale', type=float, default=0.1)
    parser.add_argument('--attack_test1_text_refresh_interval', type=int, default=10)
    parser.add_argument('--attack_test1_enable_text_beam', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attack_test1_lambda_img_shift', type=float, default=0.1)
    parser.add_argument('--attack_test1_lambda_img_cluster', type=float, default=0.1)
    parser.add_argument('--attack_test1_lambda_text_shift', type=float, default=0.1)
    parser.add_argument('--attack_test1_lambda_clip_align', type=float, default=0.1)
    parser.add_argument('--attack_test1_clip_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--attack_test1_use_pretrained_clip', action='store_true', default=False)
    parser.add_argument('--attack_test1_clip_local_files_only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attack_test1_clip_temperature', type=float, default=0.07)
    parser.add_argument('--attack_test1_save_poisoned_samples', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attack_test1_poison_save_limit', type=int, default=64)
    parser.add_argument('--attack_test1_poison_save_interval', type=int, default=50)
    parser.add_argument('--attack_test2_lambda_clean', type=float, default=1.0)
    parser.add_argument('--attack_test2_lambda_joint', type=float, default=1.5)
    parser.add_argument('--attack_test2_lambda_img', type=float, default=1.0)
    parser.add_argument('--attack_test2_lambda_text', type=float, default=1.0)
    parser.add_argument('--attack_test2_lambda_aux', type=float, default=0.5)
    parser.add_argument('--attack_test2_lambda_anti', type=float, default=0.5)
    parser.add_argument('--attack_test2_lambda_cons', type=float, default=0.5)
    parser.add_argument('--attack_test2_lambda_logit_cons', type=float, default=0.5)
    parser.add_argument('--attack_test2_lambda_align', type=float, default=0.1)
    parser.add_argument('--attack_test2_lambda_margin', type=float, default=0.5)
    parser.add_argument('--attack_test2_lambda_reg', type=float, default=1e-3)
    parser.add_argument('--attack_test2_joint_margin', type=float, default=1.0)
    parser.add_argument('--attack_test2_joint_boost_threshold', type=float, default=0.8)
    parser.add_argument('--attack_test2_joint_boost_step', type=float, default=1.15)
    parser.add_argument('--attack_test2_joint_decay_step', type=float, default=0.97)
    parser.add_argument('--attack_test2_dynamic_min', type=float, default=0.1)
    parser.add_argument('--attack_test2_dynamic_max', type=float, default=5.0)
    parser.add_argument('--attack_test2_save_poisoned_samples', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attack_test2_poison_save_limit', type=int, default=64)
    parser.add_argument('--attack_test2_poison_save_interval', type=int, default=50)
    add_robust_aggregation_args(parser)
    add_post_training_defense_args(parser)
    add_raw_prepare_args(parser)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # argument parser
    args = parse_args()
    assert clip_badcm3p_crisis_audit.enabled(args), 'This isolated entry supports CrisisMMD frozen CLIP + BadCM-3path only'
    assert args.missing_strategy == 'zero'
    configure_three_path(args)
    attack_type = str(args.attack_type).lower()
    if attack_type == 'attack_test2' and (args.backdoor_attack or args.eval_attack) and bool(getattr(args, 'attack_test2_raw_prepare', True)):
        ensure_prepared_from_args(args)
        args.attack_test2_use_raw_trigger_features = True
    if args.backdoor_attack and attack_type == 'dab_mfl':
        args.att = True
        args.att_name = 'fuse_base'
    if attack_type in ('bad_mfl', 'bad_mfl_3p') and (args.backdoor_attack or args.eval_attack):
        args.att = True
        args.att_name = 'fuse_base'
        args.d_head = int(args.bad_mfl_paper_heads)
        args.dropout_p = float(args.bad_mfl_paper_dropout)
    if attack_type in ('badcm', 'badcm_3p') and (args.backdoor_attack or args.eval_attack):
        args.att = True
        args.att_name = 'fuse_base'
        args.d_head = int(args.badcm_paper_heads)
        args.dropout_p = float(args.badcm_paper_dropout)

    # data manager
    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)
    
    # find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    save_result_dict = dict()

    if args.backdoor_attack and attack_type == 'dab_mfl':
        if ClientDABMFL is None:
            raise ModuleNotFoundError(
                'attack_type=dab_mfl requires attacks/dab_mfl, but that package is missing.'
            )
        Client = ClientDABMFL
    elif args.backdoor_attack and attack_type == 'bad_mfl_3p':
        Client = ClientBadMFL3P
    elif args.backdoor_attack and attack_type == 'badcm_3p':
        Client = ClientBadCM3P
    elif args.backdoor_attack and attack_type == 'bad_mfl':
        Client = ClientBadMFL
    elif args.backdoor_attack and attack_type == 'badcm':
        Client = ClientBadCM
    elif args.backdoor_attack and attack_type == 'toxictextclip':
        Client = ClientToxicTextCLIP
    elif args.backdoor_attack and attack_type == 'titim':
        Client = ClientTITIM
    elif args.backdoor_attack and attack_type == 'attack_test1':
        if ClientAttackTest1 is None:
            raise ModuleNotFoundError(
                'attack_type=attack_test1 requires attacks/attack_test1, but that package is missing.'
            )
        Client = ClientAttackTest1
    elif args.backdoor_attack and attack_type == 'attack_test2':
        Client = ClientAttackTest2
    elif str(getattr(args, 'missing_strategy', 'zero')).lower() == 'pmcmfl':
        Client = ClientPmcmFL
    elif str(getattr(args, 'missing_strategy', 'zero')).lower() == 'pepsy':
        Client = ClientPepsy
    elif args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS

    # load simulation feature
    dm.load_sim_dict()
    # load client ids
    dm.get_client_ids()
    attack_eval_enabled = bool(args.backdoor_attack or args.eval_attack)
    # set dataloaders
    dataloader_dict = dict()
    attack_dataloader_dict = dict()
    logging.info('Reading Data')
    for client_id in tqdm(dm.client_ids):
        # load image features
        img_dict = dm.load_img_feat(
            client_id=client_id
        )
        # load text features
        text_dict = dm.load_text_feat(
            client_id=client_id
        )
        dm.get_label_dist(
            img_dict, 
            client_id
        )
        shuffle = False if client_id in ['dev', 'test'] else True
        client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
        dataloader_dict[client_id] = dm.set_dataloader(
            img_dict,
            text_dict,
            shuffle=shuffle,
            client_sim_dict=client_sim_dict,
            client_id=client_id,
            default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
            default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]])
        )
        if client_id == 'test' and attack_eval_enabled and dm.attack_metadata is not None:
            attack_img_dict = dm.load_attack_img_feat(client_id='test')
            if attack_type == 'dab_mfl':
                attack_text_dict = dm.load_attack_text_feat(client_id='test')
                attack_dataloader_dict['attack_test_mm'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True
                )
                attack_dataloader_dict['attack_test_img_only'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='text'
                )
                attack_dataloader_dict['attack_test_text_only'] = dm.set_attack_dataloader(
                    img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='image'
                )
            elif attack_type == 'toxictextclip':
                attack_text_dict = dm.load_attack_text_feat(client_id='test')
                # Original-paper counterpart: clean image plus poisoned text.
                attack_dataloader_dict['attack_test_mm'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True
                )
                # Missing-image extension used by this framework.
                attack_dataloader_dict['attack_test_text_only'] = dm.set_attack_dataloader(
                    img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='image'
                )
            elif attack_type in ['bad_mfl', 'badcm', 'bad_mfl_3p', 'badcm_3p']:
                attack_text_dict = dm.load_attack_text_feat(client_id='test')
                attack_dataloader_dict['attack_test_mm'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True
                )
                attack_dataloader_dict['attack_test_img_only'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    [{str(r[0]):r for r in text_dict}[str(r[0])] for r in attack_img_dict],
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='text'
                )
                attack_dataloader_dict['attack_test_text_only'] = dm.set_attack_dataloader(
                    [{str(r[0]):r for r in img_dict}[str(r[0])] for r in attack_text_dict],
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='image'
                )
            elif attack_type == 'titim':
                attack_text_dict = dm.load_attack_text_feat(client_id='test')
                attack_dataloader_dict['attack_test_mm'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True
                )
                attack_dataloader_dict['attack_test_img_only'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    [{str(r[0]):r for r in text_dict}[str(r[0])] for r in attack_img_dict],
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='text'
                )
            elif attack_type == 'attack_test2':
                attack_text_dict = dm.load_attack_text_feat(client_id='test')
                attack_dataloader_dict['attack_test_mm'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True
                )
                attack_dataloader_dict['attack_test_img_only'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='text'
                )
                attack_dataloader_dict['attack_test_text_only'] = dm.set_attack_dataloader(
                    img_dict,
                    attack_text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]]),
                    attack_only_non_target=True,
                    mask_mode='image'
                )
            else:
                attack_dataloader_dict['attack_test'] = dm.set_attack_dataloader(
                    attack_img_dict,
                    text_dict,
                    default_feat_shape_a=np.array([1, constants.feature_len_dict[args.img_feat]]),
                    default_feat_shape_b=np.array([1, constants.feature_len_dict[args.text_feat]])
                )
    if attack_eval_enabled and dm.attack_metadata is None and attack_type not in ['attack_test1', 'attack_test2']:
        logging.warning('[Attack Eval] attack metadata is missing. attack_test_* evaluation will be skipped.')
    log_attack_target_prior(args, dm.label_dist_dict.get('test'))

    # Run the requested number of folds with the established fold seeds.
    for fold_idx in range(1, args.num_folds + 1):
        # number of clients
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        num_of_clients = len(client_ids)

        # set seeds
        set_seed(8*fold_idx)
        # loss function
        criterion = nn.NLLLoss().to(device)
        # Define the model
        model_kwargs = dict(
            num_classes=constants.num_class_dict[args.dataset],
            img_input_dim=constants.feature_len_dict[args.img_feat],
            text_input_dim=constants.feature_len_dict[args.text_feat],
            d_hid=args.hid_size,
            en_att=args.att,
            att_name=args.att_name,
            d_head=args.d_head,
            dropout_p=args.dropout_p,
        )
        if str(getattr(args, 'missing_strategy', 'zero')).lower() == 'pepsy':
            global_model = PepsyImageTextClassifier(
                **model_kwargs,
                pool_size=args.pepsy_pool_size,
                top_k=args.pepsy_top_k,
                prompt_dropout=args.pepsy_prompt_dropout,
                temperature=args.pepsy_temperature,
            )
        else:
            global_model = ImageTextClassifier(**model_kwargs)
        global_model = global_model.to(device)

        # initialize server
        server = Server(
            args, 
            global_model, 
            device=device, 
            criterion=criterion,
            client_ids=client_ids
        )
        server.initialize_log(fold_idx)
        server.sample_clients(num_of_clients, sample_rate=args.sample_rate)
        server.get_num_params()

        # save json path
        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result',
            args.fed_alg,
            args.dataset,
            server.feature,
            server.att,
            server.model_setting_str
        )
        if args.num_folds != 5:
            save_json_path = save_json_path.joinpath(f'folds{args.num_folds}')
        Path.mkdir(save_json_path, parents=True, exist_ok=True)
        if attack_type in ['attack_test1', 'attack_test2'] and (args.backdoor_attack or args.eval_attack):
            artifact_dir = save_json_path.joinpath(f'{attack_type}_artifacts', f'fold{fold_idx}')
            args.attack_test1_artifact_dir = str(artifact_dir)
            args.attack_test2_artifact_dir = str(artifact_dir)
        
        server.save_json_file(
            dm.label_dist_dict, 
            save_json_path.joinpath('label.json')
        )
        if attack_type in ['attack_test1', 'attack_test2'] and (args.backdoor_attack or args.eval_attack):
            attack_config = {
                'attack_type': str(args.attack_type),
                'missing_strategy': str(getattr(args, 'missing_strategy', 'zero')),
                'missing_rate': float(getattr(args, 'missing_modailty_rate', 0.0)),
                'target_label': int(args.bd_target_label),
                'malicious_ratio': float(args.bd_malicious_ratio),
                'poison_ratio': float(args.bd_poison_ratio),
                'malicious_clients': sorted([str(cid) for cid in dm.backdoor_client_ids]),
                'artifact_dir': args.attack_test1_artifact_dir,
            }
            server.save_json_file(
                attack_config,
                save_json_path.joinpath(f'{attack_type}_config_fold{fold_idx}.json')
            )
        
        # set seeds again
        set_seed(8*fold_idx)
        latest_attack_artifacts = dict()
        # Training steps
        for epoch in range(int(args.num_epochs)):
            # define list varibles that saves the weights, loss, num_sample, etc.
            args.current_global_epoch = epoch
            server.initialize_epoch_updates(epoch)
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                # Local training
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue
                # initialize client object
                client_kwargs = dict(
                    args=args,
                    device=device,
                    criterion=criterion,
                    dataloader=dataloader,
                    model=copy.deepcopy(server.global_model),
                    label_dict=dm.label_dist_dict[client_id],
                    num_class=constants.num_class_dict[args.dataset]
                )
                if Client in [ClientDABMFL, ClientBadMFL, ClientBadCM, ClientBadCM3P, ClientBadMFL3P, ClientToxicTextCLIP, ClientTITIM, ClientAttackTest1, ClientAttackTest2]:
                    client_kwargs.update(
                        client_id=client_id,
                        is_malicious=dm.is_malicious_client(client_id),
                        poisoned_keys=dm.get_poisoned_keys(client_id)
                    )
                if Client == ClientPmcmFL or (
                    Client in [ClientBadMFL, ClientBadCM, ClientBadCM3P, ClientBadMFL3P, ClientToxicTextCLIP, ClientTITIM, ClientAttackTest1, ClientAttackTest2]
                    and str(getattr(args, 'missing_strategy', 'zero')).lower() == 'pmcmfl'
                ):
                    client_kwargs.update(
                        global_prototypes=server.get_pmcmfl_prototypes()
                    )
                client = Client(**client_kwargs)

                # train
                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control), 
                        client_control=copy.deepcopy(server.client_controls[client_id])
                    )
                    client.update_weights()
                    if attack_type in THREE_PATH_ATTACKS and client.result['sample'] <= 0:
                        skip_client_ids.append(client_id)
                        continue
                    if Client in [ClientAttackTest1, ClientAttackTest2] and getattr(client, 'is_malicious', False):
                        latest_attack_artifacts = client.get_attack_artifacts()
                    prototype_update = client.get_prototype_update() if hasattr(client, 'get_prototype_update') else None

                    # server append updates
                    server.set_client_control(
                        client_id, 
                        copy.deepcopy(client.client_control)
                    )
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result,
                        prototype_update=copy.deepcopy(prototype_update),
                        delta_control=copy.deepcopy(client.delta_control),
                        client_id=client_id
                    )
                else:
                    client.update_weights()
                    if attack_type in THREE_PATH_ATTACKS and client.result['sample'] <= 0:
                        skip_client_ids.append(client_id)
                        continue
                    if Client in [ClientAttackTest1, ClientAttackTest2] and getattr(client, 'is_malicious', False):
                        latest_attack_artifacts = client.get_attack_artifacts()
                    prototype_update = client.get_prototype_update() if hasattr(client, 'get_prototype_update') else None
                    # server append updates
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result,
                        prototype_update=copy.deepcopy(prototype_update),
                        client_id=client_id
                    )
                del client
            
            # logging skip client
            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')
            record_three_path_round(args, server, skip_client_ids, device, fold_idx, epoch, save_json_path)
            
            # 2. aggregate, load new global weights
            if len(server.num_samples_list) == 0:
                clip_badcm3p_crisis_audit.append_round(args,server,skip_client_ids,device,fold_idx,epoch,save_json_path,False)
                continue
            server.average_weights()
            server.aggregate_pmcmfl_prototypes()
            logging.info('---------------------------------------------------------')
            server.log_classification_result(
                data_split='train', 
                metric='f1'
            )
            if epoch % args.test_frequency == 0 or epoch == int(args.num_epochs)-1:
                with torch.no_grad():
                    # 3. Perform the validation on dev set
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_classification_result(
                        data_split='dev', 
                        metric='f1'
                    )

                    # 4. Perform the test on holdout set
                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_classification_result(
                        data_split='test', 
                        metric='f1'
                    )
                    if attack_eval_enabled:
                        if attack_type == 'attack_test1':
                            server.result_dict[epoch]['attack_test1'] = evaluate_attack_test1_server(
                                args=args,
                                server=server,
                                test_loader=dataloader_dict['test'],
                                device=device,
                                artifacts=latest_attack_artifacts,
                            )
                        elif attack_type == 'attack_test2':
                            server.result_dict[epoch]['attack_test2'] = evaluate_attack_test2_server(
                                args=args,
                                server=server,
                                test_loader=dataloader_dict['test'],
                                device=device,
                                artifacts=latest_attack_artifacts,
                            )
                        for split_name, attack_loader in attack_dataloader_dict.items():
                            if attack_type in THREE_PATH_ATTACKS and attack_loader is None:
                                continue
                            if attack_type == 'toxictextclip' and split_name not in [
                                'attack_test_mm', 'attack_test_text_only'
                            ]:
                                continue
                            server.inference_attack(
                                attack_loader,
                                target_label=args.bd_target_label
                            )
                            server.result_dict[epoch][split_name] = server.result
                            server.log_attack_result(data_split=split_name)
                        if attack_type == 'toxictextclip':
                            if all(split in server.result_dict[epoch] for split in [
                                'attack_test_mm', 'attack_test_text_only'
                            ]):
                                toxic_result = summarize_toxictextclip_metrics(
                                    clean_result=server.result_dict[epoch]['test'],
                                    complete_result=server.result_dict[epoch]['attack_test_mm'],
                                    text_only_result=server.result_dict[epoch]['attack_test_text_only'],
                                )
                                server.result_dict[epoch]['attack_test'] = toxic_result
                                logging.info(
                                    '[toxictextclip] CA %.2f%% | ASR %.2f%% | TA %.2f%%',
                                    toxic_result.get('CA', 0.0) * 100.0,
                                    toxic_result.get('ASR', 0.0) * 100.0,
                                    toxic_result.get('TA', 0.0) * 100.0,
                                )
                                server.log_attack_result(data_split='attack_test')
                        elif attack_type == 'titim':
                            required_splits = ['attack_test_mm', 'attack_test_img_only']
                            if all(split in server.result_dict[epoch] for split in required_splits):
                                titim_result = summarize_titim_metrics(
                                    clean_result=server.result_dict[epoch]['test'],
                                    complete_result=server.result_dict[epoch]['attack_test_mm'],
                                    image_only_result=server.result_dict[epoch]['attack_test_img_only'],
                                )
                                server.result_dict[epoch]['attack_test'] = titim_result
                                logging.info(
                                    '[titim] CA %.2f%% | ASR_MM %.2f%% | ASR_I %.2f%%',
                                    titim_result.get('CA', 0.0) * 100.0,
                                    titim_result.get('ASR_MM', 0.0) * 100.0,
                                    titim_result.get('ASR_I', 0.0) * 100.0,
                                )
                                server.log_attack_result(data_split='attack_test')
                        elif attack_type in ['bad_mfl', 'badcm', 'bad_mfl_3p', 'badcm_3p', 'dab_mfl', 'attack_test2']:
                            required_splits = ['attack_test_mm', 'attack_test_img_only', 'attack_test_text_only']
                            if all(split in server.result_dict[epoch] for split in required_splits):
                                triplet_attack_result = summarize_triplet_attack_metrics(
                                    clean_result=server.result_dict[epoch]['test'],
                                    attack_results=server.result_dict[epoch],
                                )
                                server.result_dict[epoch]['attack_test'] = triplet_attack_result
                                logging.info(
                                    '[%s] CA %.2f%% | ASR_Joint %.2f%% | ASR_I %.2f%% | ASR_T %.2f%% | MBR %.4f',
                                    attack_type,
                                    triplet_attack_result.get('CA', 0.0) * 100.0,
                                    triplet_attack_result.get('ASR_Joint', 0.0) * 100.0,
                                    triplet_attack_result.get('ASR_I', 0.0) * 100.0,
                                    triplet_attack_result.get('ASR_T', 0.0) * 100.0,
                                    triplet_attack_result.get('MBR', 0.0),
                                )
                                server.log_attack_result(data_split='attack_test')
                
                logging.info('---------------------------------------------------------')
                server.log_epoch_result(metric='f1')
                clip_badcm3p_crisis_audit.append_round(args,server,skip_client_ids,device,fold_idx,epoch,save_json_path,True)
            logging.info('---------------------------------------------------------')

        if attack_type in THREE_PATH_ATTACKS and not server.best_test_dict:
            save_result_dict[f'fold{fold_idx}'] = {'status': 'no_valid_updates'}
            server.save_json_file(save_result_dict, save_json_path.joinpath('result.json'))
            logging.info('Fold %s had no valid client updates; round records were retained.', fold_idx)
            continue

        defense_result = run_post_training_defense(
            args=args,
            server=server,
            dm=dm,
            client_ids=client_ids,
            dataloader_dict=dataloader_dict,
            attack_dataloader_dict=attack_dataloader_dict,
            output_dir=save_json_path,
            fold_idx=fold_idx,
        )
        fold_result = server.summarize_dict_results()
        clip_badcm3p_crisis_audit.finalize_fold(args,fold_result,server)
        if attack_type in THREE_PATH_ATTACKS:
            fold_result['best_epoch'] = int(server.best_epoch)

        if defense_result is not None:
            for metric_name, metric_value in defense_result['post_defense']['summary'].items():
                fold_result[f"post_{args.server_defense}_{metric_name}"] = float(metric_value) * 100.0
        save_result_dict[f'fold{fold_idx}'] = fold_result
        
        # output to results
        server.save_json_file(
            save_result_dict, 
            save_json_path.joinpath('result.json')
        )

        server.log_writer.close()

    # Calculate the average over the completed folds.
    save_result_dict['average'] = dict()
    metric_names = set()
    for fold_name, fold_result in save_result_dict.items():
        if fold_name == 'average':
            continue
        metric_names.update(fold_result.keys())
    for metric in sorted(metric_names):
        result_list = list()
        for key in save_result_dict:
            if metric not in save_result_dict[key]: continue
            result_list.append(save_result_dict[key][metric])
        if len(result_list) > 0:
            numeric_result_list = list()
            is_numeric_metric = True
            for value in result_list:
                if value is None:
                    numeric_result_list.append(np.nan)
                elif isinstance(value, (int, float, np.integer, np.floating, bool)):
                    numeric_result_list.append(value)
                else:
                    is_numeric_metric = False
                    break
            if is_numeric_metric:
                save_result_dict['average'][metric] = np.nanmean(numeric_result_list)
            else:
                unique_values = list()
                seen_values = set()
                for value in result_list:
                    marker = repr(value)
                    if marker in seen_values:
                        continue
                    seen_values.add(marker)
                    unique_values.append(value)
                save_result_dict['average'][metric] = unique_values[0] if len(unique_values) == 1 else unique_values
    
    if attack_type in THREE_PATH_ATTACKS:
        add_fold_standard_deviations(save_result_dict)
    clip_sample_std(save_result_dict)

    # dump the dictionary
    server.save_json_file(
        save_result_dict, 
        save_json_path.joinpath('result.json')
    )
