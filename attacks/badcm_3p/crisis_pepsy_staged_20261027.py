"""Two-stage raw poisoning and sparse feature extraction for 3P attacks.

Only metadata.raw.json is published by raw generation. metadata.json is the
completion marker for extraction and is never visible for incomplete artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from attacks.three_path import THREE_PATH_ATTACKS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser_for(attack_type, dataset):
    parser = argparse.ArgumentParser(description=f'{attack_type}: generate raw samples, then extract saved samples')
    parser.add_argument('--stage', choices=('raw', 'extract', 'all'), default='all')
    parser.add_argument('--raw_data_dir', default=str(PROJECT_ROOT / 'data'))
    parser.add_argument('--output_dir', default=str(PROJECT_ROOT / 'output'))
    parser.add_argument('--dataset', choices=('crisis-mmd', 'hateful_memes', 'upmc_food101'), default=dataset)
    parser.add_argument('--img_feature_type', default='mobilenet_v2', choices=('mobilenet_v2',))
    parser.add_argument('--text_feature_type', default='mobilebert', choices=('mobilebert',))
    for name, value in dict(alpha=5.0, bd_malicious_ratio=0.1, bd_poison_ratio=0.1,
                            bd_target_label=0, bd_seed=2024, missing_modailty_rate=0.1).items():
        parser.add_argument('--' + name, default=value, type=type(value))
    parser.add_argument('--force', action='store_true', help='Regenerate this configuration; shared generator caches are retained.')
    parser.add_argument('--badcm_q_dependent_prepare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--badcm_pixel_eps', type=float, default=4.0 / 255.0)
    parser.add_argument('--badcm_text_trigger', default='cf')
    parser.add_argument('--badcm_pattern_path', default=str(PROJECT_ROOT / 'attacks/badcm/assets/pattern.jpg'))
    parser.add_argument('--badcm_force_prepare', action='store_true')
    defaults = dict(bad_mfl_itp_probability=0.5, bad_mfl_grid_size=3, bad_mfl_box_scale=0.32,
                    bad_mfl_topk_regions=3, bad_mfl_topk_tokens=3,
                    bad_mfl_clip_model='openai/clip-vit-base-patch32',
                    bad_mfl_t5_model='google/flan-t5-small', bad_mfl_t5_candidates=4,
                    bad_mfl_support_size=64, bad_mfl_gen_epochs=2, bad_mfl_gen_batch_size=8,
                    bad_mfl_gen_lr=0.001, bad_mfl_eps=0.18, bad_mfl_rec_lambda=3.0,
                    bad_mfl_adv_lambda=0.2, bad_mfl_align_lambda=1.0, bad_mfl_sem_lambda=4.0,
                    bad_mfl_paper_dropout=0.2, bad_mfl_paper_heads=6)
    for name, value in defaults.items():
        parser.add_argument('--' + name, default=value, type=type(value))
    parser.add_argument('--bad_mfl_q_dependent_prepare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bad_mfl_use_cached_generators', action=argparse.BooleanOptionalAction, default=True)
    for name in ('bad_mfl_use_t5', 'bad_mfl_allow_remote_model_download', 'bad_mfl_force_prepare'):
        parser.add_argument('--' + name, action='store_true')
    parser.set_defaults(attack_type=attack_type)
    return parser


def tag(value):
    return str(float(value)).replace('.', '')


def attack_root(args):
    parts = ['attacks', args.attack_type, args.dataset, 'alpha' + tag(args.alpha)]
    q_dependent = getattr(args, 'badcm_q_dependent_prepare' if args.attack_type == 'badcm_3p' else 'bad_mfl_q_dependent_prepare')
    if q_dependent and args.missing_modailty_rate > 0:
        parts.append('mm' + tag(args.missing_modailty_rate))
    parts.append(f'mr{tag(args.bd_malicious_ratio)}_pr{tag(args.bd_poison_ratio)}_tl{args.bd_target_label}_seed{args.bd_seed}')
    return Path(args.output_dir).resolve().joinpath(*parts)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def load_pickle(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def write_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def clean_roots(args):
    base = Path(args.output_dir) / 'feature'
    suffix = Path(args.dataset) / ('alpha' + tag(args.alpha))
    return base / 'img' / args.img_feature_type / suffix, base / 'text' / args.text_feature_type / suffix


def load_partition(args):
    path = Path(args.output_dir) / 'partition' / args.dataset / ('partition_alpha' + tag(args.alpha) + '.json')
    partition = json.loads(path.read_text(encoding='utf-8'))
    lookup = {str(row[0]): row for rows in partition.values() for row in rows}
    return path, partition, lookup


def training_lookup(partition):
    # Exclude holdout keys even if an invalid partition accidentally duplicates them.
    holdout = {str(row[0]) for split, rows in partition.items()
               if str(split).lower() in ('dev', 'test', 'val', 'validation') for row in rows}
    return {str(row[0]): row for split, rows in partition.items()
            if str(split).lower() not in ('dev', 'test', 'val', 'validation')
            for row in rows if str(row[0]) not in holdout}


def load_simulation(args, client_ids):
    enabled = getattr(args, 'badcm_q_dependent_prepare' if args.attack_type == 'badcm_3p' else 'bad_mfl_q_dependent_prepare')
    if not enabled:
        if args.missing_modailty_rate > 0:
            raise ValueError('3P preparation with q>0 requires q-dependent preparation.')
        return None, None
    if args.missing_modailty_rate <= 0:
        return None, None
    path = Path(args.output_dir) / 'simulation_feature' / args.dataset / f'mm{tag(args.missing_modailty_rate)}_alpha{tag(args.alpha)}.json'
    simulation = json.loads(path.read_text(encoding='utf-8'))
    missing = set(client_ids) - set(simulation)
    if missing:
        raise ValueError(f'Simulation is missing clients: {sorted(missing)}')
    return simulation, path


def available(simulation, client_id, idx, key):
    if simulation is None:
        return True, True
    rows = simulation[str(client_id)]
    if idx >= len(rows) or str(rows[idx][0]) != str(key):
        raise ValueError(f'Simulation and feature keys/order disagree for client={client_id} row={idx}')
    flags = rows[idx][-1]
    return int(flags[0]) == 0, int(flags[1]) == 0


def count_to_poison(n, ratio):
    count = int(round(n * ratio))
    return min(n, max(1, count) if n > 0 and ratio > 0 else count)


def raw_image_path(value, args):
    # Reuse the existing generic resolver, adding dataset-relative paths.
    from attacks.badcm.prepare_features import resolve_img_path
    candidates = [Path(str(value)), Path(args.raw_data_dir) / args.dataset / str(value)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(resolve_img_path(value, args.raw_data_dir))


def save_raw(root, split, client_id, key, image, text, label, target, has_i, has_t, trace):
    if shutil.disk_usage(root).free < 2*1024**3:raise RuntimeError('Disk free below 2 GiB; stop raw safely')
    from attacks.badcm.prepare_features import safe_stem
    owner = 'test' if client_id is None else 'client_' + str(client_id)
    directory = root / 'raw_samples' / split / owner
    directory.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(key)
    image_path, text_path = directory / (stem + '.png'), directory / (stem + '.txt')
    record = dict(sample_key=str(key), split=split, client_id=client_id,
                  original_label=int(label), target_label=int(target),
                  present_img=bool(has_i), present_text=bool(has_t), trace=trace,
                  poisoned_img_path=None, poisoned_text_path=None)
    if has_i:
        image.save(image_path, format='PNG')
        record.update(poisoned_img_path=str(image_path), image_sha256=digest_file(image_path))
    if has_t:
        text_path.write_text(str(text), encoding='utf-8')
        record.update(poisoned_text_path=str(text_path), text_sha256=digest_file(text_path))
    return record


def preparation_signature(args, partition_path, simulation_path):
    config = {k: v for k, v in vars(args).items()
              if k not in ('stage', 'force', 'badcm_force_prepare', 'bad_mfl_force_prepare')}
    payload = dict(version=1, config=config, partition_sha256=digest_file(partition_path),
                   simulation_sha256=digest_file(simulation_path) if simulation_path else None)
    if args.attack_type == 'badcm_3p':
        payload['pattern_sha256'] = digest_file(args.badcm_pattern_path)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def generate_raw(args, generator_factory=None):
    from attacks.badcm.prepare_features import (choose_malicious_clients,
        choose_q_dependent_malicious_clients, summarize_client_modality)
    root = attack_root(args)
    raw_meta = root / 'metadata.raw.json'
    img_dir, text_dir = clean_roots(args)
    partition_path, partition, lookup = load_partition(args)
    client_ids = sorted(p.stem for p in img_dir.glob('*.pkl') if p.stem not in ('dev', 'test'))
    if not client_ids:
        raise FileNotFoundError(f'No clean training features at {img_dir}')
    simulation, simulation_path = load_simulation(args, client_ids)
    signature = preparation_signature(args, partition_path, simulation_path)
    force = args.force or args.badcm_force_prepare or args.bad_mfl_force_prepare
    if raw_meta.exists() and not force:
        metadata = json.loads(raw_meta.read_text(encoding='utf-8'))
        if metadata['signature'] != signature:
            raise ValueError('Existing raw configuration differs; use --force or a separate output directory.')
        logging.info('Raw stage already complete: %s', root)
        return metadata
    if root.exists() and force:
        allowed = (Path(args.output_dir).resolve() / 'attacks' / args.attack_type).resolve()
        resolved = root.resolve()
        if allowed not in resolved.parents or root.is_symlink():
            raise ValueError('Refusing to replace an artifact directory outside this attack root.')
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    # Stale completion markers must never survive a failed or partial regeneration.
    (root / 'metadata.json').unlink(missing_ok=True)
    if args.attack_type == 'badcm_3p' and simulation is not None:
        _, malicious = choose_q_dependent_malicious_clients(img_dir, args.bd_malicious_ratio, args.bd_seed, simulation)
    else:
        # Bad-MFL intentionally samples all training clients and does not refill all-missing selections.
        _, malicious = choose_malicious_clients(img_dir, args.bd_malicious_ratio, args.bd_seed)
    metadata = dict(attack_type=args.attack_type, signature=signature, artifact_format='sparse_poison_overlay_v1',
                    raw_only=True, alpha=args.alpha, target_label=args.bd_target_label,
                    malicious_ratio=args.bd_malicious_ratio, poison_ratio=args.bd_poison_ratio,
                    seed=args.bd_seed, missing_rate=args.missing_modailty_rate,
                    simulation_file=str(simulation_path) if simulation_path else None,
                    partition_file=str(partition_path), partition_sha256=digest_file(partition_path),
                    simulation_sha256=digest_file(simulation_path) if simulation_path else None,
                    clean_img_dir=str(img_dir.resolve()), clean_text_dir=str(text_dir.resolve()),
                    feature_extractors=dict(image=args.img_feature_type, text=args.text_feature_type),
                    dataset=args.dataset, raw_data_dir=args.raw_data_dir,
                    malicious_clients=malicious, poisoned_keys={}, poisoned_modes={},
                    malicious_client_modalities={}, skipped_all_missing_clients=[],
                    attack_test_keys=[], attack_test_modes={}, raw_sample_root=str(root / 'raw_samples'),
                    pipeline_order=['save_raw_png_txt', 'publish_metadata.raw.json',
                                    'reload_and_verify_saved_samples', 'extract_features', 'publish_metadata.json'])
    rng = random.Random(args.bd_seed)
    logging.info('RAW_START attack=%s dataset=%s q=%s malicious=%d root=%s',
                 args.attack_type, args.dataset, args.missing_modailty_rate, len(malicious), root)
    if generator_factory is None:
        if args.attack_type == 'badcm_3p':
            from attacks.badcm_3p.prepare_features import build_generator
        else:
            from attacks.bad_mfl_3p.prepare_features import build_generator
        generator_factory = build_generator
    generate, notes = generator_factory(args, partition)
    metadata.update(notes)
    train_records, attack_records = [], []
    for client_id in malicious:
        modality = summarize_client_modality(None if simulation is None else simulation[client_id])
        metadata['malicious_client_modalities'][client_id] = modality
        metadata['poisoned_keys'][client_id] = []
        metadata['poisoned_modes'][client_id] = {}
        if not modality['has_any']:
            metadata['skipped_all_missing_clients'].append(client_id)
            continue
        rows = load_pickle(img_dir / (client_id + '.pkl'))
        candidates = []
        for idx, item in enumerate(rows):
            has_i, has_t = available(simulation, client_id, idx, item[0])
            if int(item[-2]) != args.bd_target_label and (has_i or has_t):
                candidates.append((idx, has_i, has_t))
        chosen = rng.sample(candidates, count_to_poison(len(candidates), args.bd_poison_ratio))
        for idx, has_i, has_t in chosen:
            key = str(rows[idx][0])
            raw = lookup[key]
            path, original_text = raw_image_path(raw[1], args), str(raw[-1])
            image = Image.open(path).convert('RGB') if has_i or args.attack_type == 'bad_mfl_3p' else None
            image, text, trace = generate(key, image, original_text, has_i, has_t, rng)
            record = save_raw(root, 'train', client_id, key, image, text, raw[2],
                              args.bd_target_label, has_i, has_t, trace)
            record['original_img_path'] = str(path)
            train_records.append(record)
            if len(train_records) % 100 == 0:
                logging.info('RAW_PROGRESS train=%d dataset=%s q=%s', len(train_records), args.dataset, args.missing_modailty_rate)
            metadata['poisoned_keys'][client_id].append(key)
            metadata['poisoned_modes'][client_id][key] = dict(present_img=has_i, present_text=has_t,
                raw_img_path=record['poisoned_img_path'], raw_text_path=record['poisoned_text_path'])
    test_rows = load_pickle(img_dir / 'test.pkl')
    candidates = [r for r in test_rows if int(r[-2]) != args.bd_target_label]
    selected = rng.sample(candidates, min(len(candidates), count_to_poison(len(test_rows), args.bd_poison_ratio)))
    for item in selected:
        key = str(item[0]); raw = lookup[key]
        path = raw_image_path(raw[1], args)
        image, text, trace = generate(key, Image.open(path).convert('RGB'), str(raw[-1]), True, True, rng)
        record = save_raw(root, 'attack_test', None, key, image, text, raw[2], args.bd_target_label, True, True, trace)
        record['original_img_path'] = str(path)
        attack_records.append(record)
        if len(attack_records) % 100 == 0:
            logging.info('RAW_PROGRESS attack_test=%d/%d dataset=%s q=%s',
                         len(attack_records), len(selected), args.dataset, args.missing_modailty_rate)
        metadata['attack_test_keys'].append(key)
        metadata['attack_test_modes'][key] = dict(raw_img_path=record['poisoned_img_path'], raw_text_path=record['poisoned_text_path'])
    for split, records in (('train', train_records), ('attack_test', attack_records)):
        path = root / 'raw_samples' / (split + '_manifest.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        metadata['raw_' + split + '_manifest'] = str(path)
        metadata['raw_' + split + '_count'] = len(records)
        metadata['raw_' + split + '_manifest_sha256'] = digest_file(path)
    metadata['attack_eval_count'] = len(attack_records)
    metadata['attack_eval_ratio'] = args.bd_poison_ratio
    atomic_json(raw_meta, metadata)
    logging.info('RAW_COMPLETE attack=%s train=%d test=%d root=%s', args.attack_type, len(train_records), len(attack_records), root)
    return metadata


def read_raw_records(root, metadata, split):
    path = root / 'raw_samples' / (split + '_manifest.jsonl')
    if digest_file(path) != metadata['raw_' + split + '_manifest_sha256']:
        raise ValueError('Raw manifest checksum mismatch: ' + str(path))
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for record in records:
        for modality, field in (('image', 'poisoned_img_path'), ('text', 'poisoned_text_path')):
            if record[field] is not None:
                saved = Path(record[field])
                if root.resolve() not in saved.resolve().parents:
                    raise ValueError('Raw file is outside the artifact root.')
                if digest_file(saved) != record[modality + '_sha256']:
                    raise ValueError('Saved raw sample checksum mismatch: ' + str(saved))
    return records


def extract_features(args, factory=None):
    root = attack_root(args)
    metadata = json.loads((root / 'metadata.raw.json').read_text(encoding='utf-8'))
    partition_path = Path(metadata['partition_file'])
    sim_path = Path(metadata['simulation_file']) if metadata['simulation_file'] else None
    if preparation_signature(args, partition_path, sim_path) != metadata['signature']:
        raise ValueError('Extract arguments or input files differ from the saved raw-stage configuration.')
    if (root / 'metadata.json').exists() and not args.force:
        logging.info('Extract stage already complete: %s', root)
        return
    (root / 'metadata.json').unlink(missing_ok=True)
    train, test = [read_raw_records(root, metadata, split) for split in ('train', 'attack_test')]
    logging.info('EXTRACT_START dataset=%s q=%s train=%d test=%d', args.dataset,
                 args.missing_modailty_rate, len(train), len(test))
    if factory is None:
        from features.feature_processing.feature_manager import FeatureManager
        factory = FeatureManager
    # Victim feature modules are instantiated only after raw generation has finished.
    common = dict(raw_data_dir=args.raw_data_dir, output_dir=args.output_dir, dataset=args.dataset)
    image_fm = factory(SimpleNamespace(**common, feature_type=args.img_feature_type))
    text_fm = factory(SimpleNamespace(**common, feature_type=args.text_feature_type))
    overlays_i = {cid: {} for cid in metadata['malicious_clients']}
    overlays_t = {cid: {} for cid in metadata['malicious_clients']}
    for index, record in enumerate(train, 1):
        key, cid = record['sample_key'], record['client_id']
        image = image_fm.extract_img_features(record['poisoned_img_path']) if record['present_img'] else None
        text = text_fm.extract_text_feature(Path(record['poisoned_text_path']).read_text(encoding='utf-8')) if record['present_text'] else None
        overlays_i[cid][key] = dict(feature=image, label=record['target_label'])
        overlays_t[cid][key] = dict(feature=text, label=record['target_label'])
        if index % 100 == 0:
            logging.info('EXTRACT_PROGRESS train=%d/%d q=%s', index, len(train), args.missing_modailty_rate)
    for cid in metadata['malicious_clients']:
        write_pickle(root / 'poisoned_img' / (cid + '.pkl'), overlays_i[cid])
        write_pickle(root / 'poisoned_text' / (cid + '.pkl'), overlays_t[cid])
    attack_i, attack_t = [], []
    for index, record in enumerate(test, 1):
        # Preserve original labels for non-target filtering and original-class metrics.
        image = image_fm.extract_img_features(record['poisoned_img_path'])
        text = text_fm.extract_text_feature(Path(record['poisoned_text_path']).read_text(encoding='utf-8'))
        attack_i.append([record['sample_key'], record['original_img_path'], record['original_label'], image])
        attack_t.append([record['sample_key'], record['original_img_path'], record['original_label'], text])
        if index % 100 == 0:
            logging.info('EXTRACT_PROGRESS attack_test=%d/%d q=%s', index, len(test), args.missing_modailty_rate)
    write_pickle(root / 'attack_img/test.pkl', attack_i)
    write_pickle(root / 'attack_text/test.pkl', attack_t)
    metadata.update(raw_only=False, extraction_source='verified_saved_png_txt', status='complete')
    metadata['feature_artifact_sha256'] = {str(p.relative_to(root)): digest_file(p)
        for directory in ('poisoned_img', 'poisoned_text', 'attack_img', 'attack_text')
        for p in sorted((root / directory).glob('*.pkl'))}
    atomic_json(root / 'metadata.json', metadata)
    logging.info('EXTRACT_COMPLETE attack=%s root=%s', args.attack_type, root)


def main(attack_type, dataset='crisis-mmd'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parser_for(attack_type, dataset).parse_args()
    if attack_type not in THREE_PATH_ATTACKS:
        raise ValueError(attack_type)
    for field in ('bd_malicious_ratio', 'bd_poison_ratio', 'missing_modailty_rate'):
        if not 0 <= getattr(args, field) <= 1:
            raise ValueError(field + ' must be between zero and one')
    if args.stage in ('raw', 'all'):
        if attack_type == 'bad_mfl_3p' and os.environ.get('PYTHONHASHSEED') != str(args.bd_seed):
            raise RuntimeError(f'Start the interpreter with PYTHONHASHSEED={args.bd_seed}; the original generator uses hash().')
        import torch
        random.seed(args.bd_seed); np.random.seed(args.bd_seed); torch.manual_seed(args.bd_seed)
        torch.cuda.manual_seed_all(args.bd_seed)
        generate_raw(args)
    if args.stage in ('extract', 'all'):
        extract_features(args)
