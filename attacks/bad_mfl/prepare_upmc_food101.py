from __future__ import annotations

import argparse
import collections
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.bad_mfl.config import BadMFLConfig, TRIGGER_BANK
from attacks.bad_mfl.data_utils import (
    build_partition_lookup,
    clone_feature_args,
    dump_json,
    dump_jsonl,
    load_feature_dict,
    resolve_img_path,
    save_feature_dict,
    save_poison_pair,
)
from attacks.bad_mfl.key_extractor import ClipHelper, KeyComponentExtractor
from attacks.bad_mfl.poison_generators import ITPGenerator, SupportRecord, TTPGenerator, ImageTriggerSynthesizer
from attacks.bad_mfl.text_trigger import TextCandidateGenerator, TextTriggerBuilder
from features.feature_processing.feature_manager import FeatureManager


logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    path_conf = dict()
    with open(str(PROJECT_ROOT.joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace('"', '')

    if path_conf['data_dir'] == '.':
        path_conf['data_dir'] = str(PROJECT_ROOT.joinpath('data'))
    if path_conf['output_dir'] == '.':
        path_conf['output_dir'] = str(PROJECT_ROOT.joinpath('output'))

    parser = argparse.ArgumentParser(description='Prepare Bad-MFL poisoned data for UPMC-Food101')
    parser.add_argument('--raw_data_dir', default=path_conf['data_dir'], type=str)
    parser.add_argument('--output_dir', default=path_conf['output_dir'], type=str)
    parser.add_argument('--img_feature_type', default='mobilenet_v2', type=str)
    parser.add_argument('--text_feature_type', default='mobilebert', type=str)
    parser.add_argument('--alpha', default=5.0, type=float)
    parser.add_argument('--dataset', default='upmc_food101', type=str)
    parser.add_argument('--bd_malicious_ratio', default=0.1, type=float)
    parser.add_argument('--bd_poison_ratio', default=0.1, type=float)
    parser.add_argument('--bd_target_label', default=0, type=int)
    parser.add_argument('--bd_seed', default=2024, type=int)
    parser.add_argument('--missing_modailty_rate', default=0.0, type=float)
    parser.add_argument('--bad_mfl_q_dependent_prepare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bad_mfl_itp_probability', default=0.5, type=float)
    parser.add_argument('--bad_mfl_grid_size', default=3, type=int)
    parser.add_argument('--bad_mfl_box_scale', default=0.32, type=float)
    parser.add_argument('--bad_mfl_topk_regions', default=3, type=int)
    parser.add_argument('--bad_mfl_topk_tokens', default=3, type=int)
    parser.add_argument('--bad_mfl_clip_model', default='openai/clip-vit-base-patch32', type=str)
    parser.add_argument('--bad_mfl_allow_remote_model_download', action='store_true')
    parser.add_argument('--bad_mfl_use_t5', action='store_true')
    parser.add_argument('--bad_mfl_t5_model', default='google/flan-t5-small', type=str)
    parser.add_argument('--bad_mfl_t5_candidates', default=4, type=int)
    parser.add_argument('--bad_mfl_support_size', default=64, type=int)
    parser.add_argument('--bad_mfl_gen_epochs', default=2, type=int)
    parser.add_argument('--bad_mfl_gen_batch_size', default=8, type=int)
    parser.add_argument('--bad_mfl_gen_lr', default=1e-3, type=float)
    parser.add_argument('--bad_mfl_eps', default=0.18, type=float)
    parser.add_argument('--bad_mfl_rec_lambda', default=3.0, type=float)
    parser.add_argument('--bad_mfl_adv_lambda', default=0.20, type=float)
    parser.add_argument('--bad_mfl_align_lambda', default=1.0, type=float)
    parser.add_argument('--bad_mfl_sem_lambda', default=4.0, type=float)
    parser.add_argument('--bad_mfl_use_cached_generators', action='store_true')
    parser.add_argument('--bad_mfl_force_prepare', action='store_true')
    parser.add_argument('--bad_mfl_paper_dropout', default=0.2, type=float)
    parser.add_argument('--bad_mfl_paper_heads', default=6, type=int)
    return parser.parse_args()


def choose_malicious_clients(clean_img_dir: Path, ratio: float, seed: int):
    client_ids = sorted([p.stem for p in clean_img_dir.glob('*.pkl') if p.stem not in ['dev', 'test']])
    rng = random.Random(seed)
    num_clients = int(round(len(client_ids) * ratio))
    if ratio > 0 and num_clients == 0:
        num_clients = 1
    num_clients = min(num_clients, len(client_ids))
    malicious_clients = sorted(rng.sample(client_ids, num_clients)) if num_clients > 0 else []
    return client_ids, malicious_clients


def _q_tag(rate: float) -> str:
    return str(float(rate)).replace('.', '')


def load_missing_simulation(config: BadMFLConfig):
    if not config.bad_mfl_q_dependent_prepare:
        return None, None
    if float(config.missing_modailty_rate) <= 0.0:
        return None, None
    sim_path = Path(config.output_dir).joinpath(
        'simulation_feature',
        config.dataset,
        f'mm{_q_tag(config.missing_modailty_rate)}_alpha{config.alpha_tag}.json'
    )
    if not sim_path.exists():
        raise FileNotFoundError(f'Bad-MFL q-dependent prepare requires simulation file: {sim_path}')
    with open(str(sim_path), 'r', encoding='utf-8') as f:
        return json.load(f), sim_path


def _sample_available_from_sim(client_sim, idx: int):
    if client_sim is None:
        return True, True
    if idx >= len(client_sim):
        return True, True
    sim_flag = client_sim[idx][-1]
    present_img = int(sim_flag[0]) == 0
    present_text = int(sim_flag[1]) == 0
    return present_img, present_text


def summarize_client_modality(client_sim):
    if client_sim is None:
        return {
            'type': 'full_img_text',
            'counts': {'full_img_text': 1},
            'has_img': True,
            'has_text': True,
            'has_any': True,
            'q_dependent': False,
        }
    label_map = {
        (0, 0): 'full_img_text',
        (0, 1): 'image_only',
        (1, 0): 'text_only',
        (1, 1): 'all_missing',
    }
    counts = collections.Counter()
    has_img = False
    has_text = False
    for row in client_sim:
        sim_flag = row[-1]
        key = (int(sim_flag[0]), int(sim_flag[1]))
        counts[label_map.get(key, f'unknown_{key}')] += 1
        if key[0] == 0:
            has_img = True
        if key[1] == 0:
            has_text = True
    if len(counts) == 1:
        client_type = next(iter(counts.keys()))
    else:
        client_type = 'mixed'
    return {
        'type': client_type,
        'counts': dict(counts),
        'has_img': bool(has_img),
        'has_text': bool(has_text),
        'has_any': bool(has_img or has_text),
        'q_dependent': True,
    }


def collect_support_records(config: BadMFLConfig, partition_lookup, extractor: KeyComponentExtractor, support_size: int):
    rng = random.Random(config.bd_seed + 17)
    support_candidates = []
    for sample_key, raw_record in partition_lookup.items():
        label = int(raw_record[2])
        if label == config.bd_target_label:
            continue
        if sample_key in ['dev', 'test']:
            continue
        support_candidates.append(str(sample_key))
    rng.shuffle(support_candidates)
    support_records = []
    for sample_key in support_candidates:
        raw_record = partition_lookup[sample_key]
        img_path = resolve_img_path(raw_record[1], config.raw_data_dir)
        text = str(raw_record[-1])
        image = Image.open(img_path).convert('RGB')
        matches = extractor.extract(sample_key=sample_key, image=image, text=text)
        if not matches:
            continue
        support_records.append(
            SupportRecord(
                sample_key=sample_key,
                img_path=img_path,
                text=text,
                box=tuple(matches[0].box)
            )
        )
        if len(support_records) >= support_size:
            break
    if not support_records:
        raise RuntimeError('Failed to build Bad-MFL support records for generator training')
    return support_records


def build_generator_bundle(config: BadMFLConfig, clip_helper: ClipHelper, support_records, text_generator: TextCandidateGenerator):
    bundle = dict()
    for concept in TRIGGER_BANK:
        synth = ImageTriggerSynthesizer(config=config, concept=concept, clip_helper=clip_helper)
        synth.train_or_load(support_records)
        text_builder = TextTriggerBuilder(
            concept=concept,
            text_generator=text_generator,
            cache_root=config.text_cache_root.joinpath(concept.name)
        )
        bundle[concept.name] = {
            'itp': ITPGenerator(concept, synth, text_builder, clip_helper, config.bad_mfl_t5_candidates),
            'ttp': TTPGenerator(concept, synth, text_builder, clip_helper, config.bad_mfl_t5_candidates),
        }
    return bundle


def main():
    args = parse_args()
    config = BadMFLConfig.from_args(args)
    if not config.bad_mfl_allow_remote_model_download:
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

    config.attack_root.mkdir(parents=True, exist_ok=True)
    config.poisoned_img_dir.mkdir(parents=True, exist_ok=True)
    config.poisoned_text_dir.mkdir(parents=True, exist_ok=True)
    config.attack_img_dir.mkdir(parents=True, exist_ok=True)
    config.attack_text_dir.mkdir(parents=True, exist_ok=True)
    config.poison_pair_root.mkdir(parents=True, exist_ok=True)
    config.key_cache_root.mkdir(parents=True, exist_ok=True)
    config.text_cache_root.mkdir(parents=True, exist_ok=True)
    config.generator_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.bd_seed)
    clip_helper = ClipHelper(config.bad_mfl_clip_model, allow_remote_model_download=config.bad_mfl_allow_remote_model_download)
    text_generator = TextCandidateGenerator(config.bad_mfl_t5_model, config.bad_mfl_use_t5, allow_remote_model_download=config.bad_mfl_allow_remote_model_download)
    extractor = KeyComponentExtractor(
        clip_helper=clip_helper,
        grid_size=config.bad_mfl_grid_size,
        box_scale=config.bad_mfl_box_scale,
        topk_regions=config.bad_mfl_topk_regions,
        topk_tokens=config.bad_mfl_topk_tokens,
        cache_root=config.key_cache_root,
    )

    partition, partition_lookup = build_partition_lookup(config)
    img_fm = FeatureManager(clone_feature_args(config, config.img_feature_type))
    text_fm = FeatureManager(clone_feature_args(config, config.text_feature_type))
    train_client_ids, malicious_clients = choose_malicious_clients(config.clean_img_dir, config.bd_malicious_ratio, config.bd_seed)
    missing_simulation, missing_sim_path = load_missing_simulation(config)

    logging.info('Bad-MFL prepare: %s malicious clients selected out of %s', len(malicious_clients), len(train_client_ids))
    if missing_sim_path is not None:
        logging.info('Bad-MFL q-dependent prepare uses simulation file: %s', missing_sim_path)
    support_records = collect_support_records(config, partition_lookup, extractor, config.bad_mfl_support_size)
    generator_bundle = build_generator_bundle(config, clip_helper, support_records, text_generator)

    metadata = {
        'attack_type': 'bad_mfl',
        'alpha': config.alpha,
        'target_label': config.bd_target_label,
        'malicious_ratio': config.bd_malicious_ratio,
        'poison_ratio': config.bd_poison_ratio,
        'seed': config.bd_seed,
        'missing_rate': config.missing_modailty_rate,
        'q_dependent_prepare': config.bad_mfl_q_dependent_prepare,
        'simulation_file': None if missing_sim_path is None else str(missing_sim_path),
        'poisoned_keys': {},
        'poisoned_modes': {},
        'malicious_client_modalities': {},
        'skipped_all_missing_clients': [],
        'attack_test_keys': [],
        'attack_test_modes': {},
        'malicious_clients': malicious_clients,
        'clip_model': config.bad_mfl_clip_model,
        't5_model': config.bad_mfl_t5_model if config.bad_mfl_use_t5 else None,
        'paper_model': {
            'dropout': config.bad_mfl_paper_dropout,
            'd_head': config.bad_mfl_paper_heads,
        },
        'generator_ckpt': {concept.name: str(config.generator_root.joinpath(f'{concept.name}.pt')) for concept in TRIGGER_BANK},
        'poison_pair_export': {
            'root': str(config.poison_pair_root),
            'train_manifest': str(config.poison_pair_root.joinpath('train_manifest.jsonl')),
            'attack_test_manifest': str(config.poison_pair_root.joinpath('attack_test_manifest.jsonl')),
            'train_count': 0,
            'attack_test_count': 0,
        }
    }
    train_manifest = []
    attack_manifest = []

    modality_summary_counter = collections.Counter()
    for client_id in malicious_clients:
        client_sim = None if missing_simulation is None else missing_simulation.get(str(client_id))
        client_modality = summarize_client_modality(client_sim)
        metadata['malicious_client_modalities'][client_id] = client_modality
        modality_summary_counter[client_modality['type']] += 1
        if not client_modality['has_any']:
            metadata['poisoned_keys'][client_id] = []
            metadata['poisoned_modes'][client_id] = {}
            metadata['skipped_all_missing_clients'].append(client_id)
            continue

        img_data = load_feature_dict(config.clean_img_dir.joinpath(f'{client_id}.pkl'))
        text_data = load_feature_dict(config.clean_text_dir.joinpath(f'{client_id}.pkl'))
        poisoned_img = copy.deepcopy(img_data)
        poisoned_text = copy.deepcopy(text_data)
        eligible_idx = []
        for idx, item in enumerate(img_data):
            present_img, present_text = _sample_available_from_sim(client_sim, idx)
            if int(item[-2]) != config.bd_target_label and (present_img or present_text):
                eligible_idx.append(idx)
        poison_count = int(round(len(eligible_idx) * config.bd_poison_ratio))
        if config.bd_poison_ratio > 0 and poison_count == 0 and eligible_idx:
            poison_count = 1
        poison_count = min(poison_count, len(eligible_idx))
        poison_idx = rng.sample(eligible_idx, poison_count) if poison_count > 0 else []
        metadata['poisoned_keys'][client_id] = []
        metadata['poisoned_modes'][client_id] = {}

        for idx in poison_idx:
            present_img, present_text = _sample_available_from_sim(client_sim, idx)
            if not (present_img or present_text):
                continue
            sample_key = str(img_data[idx][0])
            raw_record = partition_lookup[sample_key]
            img_path = resolve_img_path(raw_record[1], config.raw_data_dir)
            original_label = int(raw_record[2])
            original_text = str(raw_record[-1])
            original_image = Image.open(img_path).convert('RGB')
            matches = extractor.extract(sample_key=sample_key, image=original_image, text=original_text)
            concept = rng.choice(TRIGGER_BANK)
            mode_name = 'itp' if rng.random() < config.bad_mfl_itp_probability else 'ttp'
            poison = generator_bundle[concept.name][mode_name].generate(sample_key, original_image, original_text, matches)
            trace = poison['trace']

            if present_img:
                poisoned_img[idx][-1] = img_fm.extract_img_features_from_pil(poison['poisoned_image'])
            if present_text:
                poisoned_text[idx][-1] = text_fm.extract_text_feature(poison['poisoned_text'])
            # The paired dataset takes labels from the image-side record, so both
            # records carry the target label even when only one modality is visible.
            poisoned_img[idx][-2] = config.bd_target_label
            poisoned_text[idx][-2] = config.bd_target_label

            metadata['poisoned_keys'][client_id].append(sample_key)
            metadata['poisoned_modes'][client_id][sample_key] = {
                'mode': trace.mode,
                'concept': trace.concept,
                'matched_token': trace.matched_token,
                'matched_box': [int(v) for v in trace.matched_box],
                'candidate_count': trace.candidate_count,
                'present_img': bool(present_img),
                'present_text': bool(present_text),
            }
            exported_image = poison['poisoned_image'] if present_img else original_image
            exported_text = poison['poisoned_text'] if present_text else original_text
            train_manifest.append(
                save_poison_pair(
                    pair_root=config.poison_pair_root,
                    sample_key=sample_key,
                    split_name='train',
                    client_id=client_id,
                    original_label=original_label,
                    target_label=config.bd_target_label,
                    original_img_path=img_path,
                    poisoned_image=exported_image,
                    original_text=original_text,
                    poisoned_text=exported_text,
                    mode_name=trace.mode,
                    concept_name=trace.concept,
                    matched_token=trace.matched_token,
                    matched_box=trace.matched_box,
                    extra={
                        'candidate_count': trace.candidate_count,
                        'trigger_text': trace.trigger_text,
                        'present_img': bool(present_img),
                        'present_text': bool(present_text),
                    }
                )
            )

        save_feature_dict(config.poisoned_img_dir.joinpath(f'{client_id}.pkl'), poisoned_img)
        save_feature_dict(config.poisoned_text_dir.joinpath(f'{client_id}.pkl'), poisoned_text)

    metadata['malicious_client_modality_summary'] = dict(modality_summary_counter)
    logging.info('Bad-MFL malicious modality summary: %s', dict(modality_summary_counter))
    if metadata['skipped_all_missing_clients']:
        logging.info('Bad-MFL skipped all-missing malicious clients: %s', metadata['skipped_all_missing_clients'])

    clean_test_img = load_feature_dict(config.clean_img_dir.joinpath('test.pkl'))
    clean_test_text = load_feature_dict(config.clean_text_dir.joinpath('test.pkl'))
    attack_test_img = []
    attack_test_text = []
    eligible_attack_pairs = [
        (img_item, text_item)
        for img_item, text_item in zip(clean_test_img, clean_test_text)
        if int(img_item[-2]) != config.bd_target_label
    ]
    target_attack_count = int(round(len(clean_test_img) * config.bd_poison_ratio))
    if config.bd_poison_ratio > 0 and target_attack_count == 0 and len(eligible_attack_pairs) > 0:
        target_attack_count = 1
    target_attack_count = min(target_attack_count, len(eligible_attack_pairs))
    attack_eval_pairs = rng.sample(eligible_attack_pairs, target_attack_count) if target_attack_count > 0 else []
    metadata['attack_eval_ratio'] = config.bd_poison_ratio
    metadata['attack_eval_count'] = target_attack_count
    for img_item, text_item in attack_eval_pairs:
        sample_key = str(img_item[0])
        raw_record = partition_lookup[sample_key]
        img_path = resolve_img_path(raw_record[1], config.raw_data_dir)
        original_label = int(raw_record[2])
        original_text = str(raw_record[-1])
        original_image = Image.open(img_path).convert('RGB')
        matches = extractor.extract(sample_key=sample_key, image=original_image, text=original_text)
        concept = rng.choice(TRIGGER_BANK)
        mode_name = 'itp' if rng.random() < config.bad_mfl_itp_probability else 'ttp'
        poison = generator_bundle[concept.name][mode_name].generate(sample_key, original_image, original_text, matches)
        trace = poison['trace']

        attack_img_item = copy.deepcopy(img_item)
        attack_text_item = copy.deepcopy(text_item)
        attack_img_item[-1] = img_fm.extract_img_features_from_pil(poison['poisoned_image'])
        attack_text_item[-1] = text_fm.extract_text_feature(poison['poisoned_text'])
        attack_test_img.append(attack_img_item)
        attack_test_text.append(attack_text_item)
        metadata['attack_test_keys'].append(sample_key)
        metadata['attack_test_modes'][sample_key] = {
            'mode': trace.mode,
            'concept': trace.concept,
            'matched_token': trace.matched_token,
            'matched_box': [int(v) for v in trace.matched_box],
            'candidate_count': trace.candidate_count,
        }
        attack_manifest.append(
            save_poison_pair(
                pair_root=config.poison_pair_root,
                sample_key=sample_key,
                split_name='attack_test',
                client_id=None,
                original_label=original_label,
                target_label=config.bd_target_label,
                original_img_path=img_path,
                poisoned_image=poison['poisoned_image'],
                original_text=original_text,
                poisoned_text=poison['poisoned_text'],
                mode_name=trace.mode,
                concept_name=trace.concept,
                matched_token=trace.matched_token,
                matched_box=trace.matched_box,
                extra={'candidate_count': trace.candidate_count, 'trigger_text': trace.trigger_text}
            )
        )

    save_feature_dict(config.attack_img_dir.joinpath('test.pkl'), attack_test_img)
    save_feature_dict(config.attack_text_dir.joinpath('test.pkl'), attack_test_text)
    dump_jsonl(config.poison_pair_root.joinpath('train_manifest.jsonl'), train_manifest)
    dump_jsonl(config.poison_pair_root.joinpath('attack_test_manifest.jsonl'), attack_manifest)
    metadata['poison_pair_export']['train_count'] = len(train_manifest)
    metadata['poison_pair_export']['attack_test_count'] = len(attack_manifest)
    dump_json(config.attack_root.joinpath('metadata.json'), metadata)
    logging.info('Bad-MFL metadata saved to %s', config.attack_root.joinpath('metadata.json'))


if __name__ == '__main__':
    main()
