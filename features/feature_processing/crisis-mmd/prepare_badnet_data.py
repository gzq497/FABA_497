import json
import copy
import pickle
import random
import argparse

from pathlib import Path
from PIL import Image, ImageDraw

import os
import sys

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_processing.feature_manager import FeatureManager

import logging
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
            path_conf[key] = val.replace("\"", "")

    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(PROJECT_ROOT.joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(PROJECT_ROOT.joinpath('output'))

    parser = argparse.ArgumentParser(description='Prepare BadNet poisoned data for crisis-mmd')
    parser.add_argument('--raw_data_dir', default=path_conf["data_dir"], type=str)
    parser.add_argument('--output_dir', default=path_conf["output_dir"], type=str)
    parser.add_argument('--feature_type', default='mobilenet_v2', type=str)
    parser.add_argument('--alpha', default=5.0, type=float)
    parser.add_argument('--dataset', default='crisis-mmd', type=str)
    parser.add_argument('--bd_malicious_ratio', default=0.2, type=float)
    parser.add_argument('--bd_poison_ratio', default=0.2, type=float)
    parser.add_argument('--bd_target_label', default=0, type=int)
    parser.add_argument('--bd_trigger_size', default=24, type=int)
    parser.add_argument('--bd_seed', default=2024, type=int)
    parser.add_argument('--bd_trigger_color', default='255,255,255', type=str)
    return parser.parse_args()


def apply_badnet_patch(img_path: str, patch_size: int, trigger_color):
    image = Image.open(img_path).convert('RGB')
    width, height = image.size
    patch_size = min(patch_size, width, height)
    start_x = width - patch_size
    start_y = height - patch_size
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [start_x, start_y, width - 1, height - 1],
        fill=trigger_color
    )
    return image


def resolve_img_path(img_path: str, raw_data_dir: str):
    path_obj = Path(img_path)
    if Path.exists(path_obj):
        return str(path_obj)

    path_str = str(img_path).replace('\\', '/')
    markers = [
        'data/crisis-mmd/',
        'crisis-mmd/',
        'CrisisMMD_v2.0/'
    ]
    for marker in markers:
        if marker in path_str:
            suffix = path_str.split(marker, 1)[1]
            if marker == 'CrisisMMD_v2.0/':
                candidate = Path(raw_data_dir).joinpath('crisis-mmd', 'CrisisMMD_v2.0', suffix)
            else:
                candidate = Path(raw_data_dir).joinpath('crisis-mmd', suffix)
            if Path.exists(candidate):
                return str(candidate)
    raise FileNotFoundError(f'Cannot resolve image path: {img_path}')


def get_badnet_config_dir(args):
    alpha_str = str(args.alpha).replace('.', '')
    mr = str(args.bd_malicious_ratio).replace('.', '')
    pr = str(args.bd_poison_ratio).replace('.', '')
    return Path(args.output_dir).joinpath(
        'backdoor',
        'badnet',
        args.dataset,
        f'alpha{alpha_str}',
        f'mr{mr}_pr{pr}_tl{args.bd_target_label}_ts{args.bd_trigger_size}'
    )


if __name__ == '__main__':
    args = parse_args()
    rng = random.Random(args.bd_seed)
    alpha_str = str(args.alpha).replace('.', '')
    trigger_color = tuple(int(v.strip()) for v in args.bd_trigger_color.split(','))

    clean_img_dir = Path(args.output_dir).joinpath(
        'feature', 'img', args.feature_type, args.dataset, f'alpha{alpha_str}'
    )
    clean_text_dir = Path(args.output_dir).joinpath(
        'feature', 'text', 'mobilebert', args.dataset, f'alpha{alpha_str}'
    )
    badnet_root = get_badnet_config_dir(args)
    poisoned_img_dir = badnet_root.joinpath('poisoned_img')
    poisoned_text_dir = badnet_root.joinpath('poisoned_text')
    attack_img_dir = badnet_root.joinpath('attack_img')
    meta_path = badnet_root.joinpath('metadata.json')

    Path.mkdir(poisoned_img_dir, parents=True, exist_ok=True)
    Path.mkdir(poisoned_text_dir, parents=True, exist_ok=True)
    Path.mkdir(attack_img_dir, parents=True, exist_ok=True)

    fm = FeatureManager(args)

    client_ids = [p.stem for p in clean_img_dir.glob('*.pkl')]
    train_client_ids = sorted([cid for cid in client_ids if cid not in ['dev', 'test']])
    num_malicious = max(1, int(round(len(train_client_ids) * args.bd_malicious_ratio)))
    malicious_clients = sorted(rng.sample(train_client_ids, num_malicious))

    metadata = {
        'alpha': args.alpha,
        'target_label': args.bd_target_label,
        'poison_ratio': args.bd_poison_ratio,
        'malicious_ratio': args.bd_malicious_ratio,
        'trigger_size': args.bd_trigger_size,
        'trigger_color': trigger_color,
        'seed': args.bd_seed,
        'malicious_clients': malicious_clients,
        'poisoned_keys': dict()
    }

    logging.info('Preparing poisoned training features for malicious clients')
    for client_id in malicious_clients:
        with open(str(clean_img_dir.joinpath(f'{client_id}.pkl')), 'rb') as f:
            img_data = pickle.load(f)
        with open(str(clean_text_dir.joinpath(f'{client_id}.pkl')), 'rb') as f:
            text_data = pickle.load(f)

        poisoned_img = copy.deepcopy(img_data)
        poisoned_text = copy.deepcopy(text_data)

        eligible_idx = [idx for idx, item in enumerate(img_data) if int(item[-2]) != args.bd_target_label]
        poison_count = int(round(len(img_data) * args.bd_poison_ratio))
        if args.bd_poison_ratio > 0 and poison_count == 0:
            poison_count = 1
        poison_count = min(poison_count, len(eligible_idx))
        poison_idx = rng.sample(eligible_idx, poison_count) if poison_count > 0 else []

        poisoned_keys = []
        for idx in poison_idx:
            sample_key = img_data[idx][0]
            img_path = resolve_img_path(img_data[idx][1], args.raw_data_dir)
            patched_img = apply_badnet_patch(
                img_path=img_path,
                patch_size=args.bd_trigger_size,
                trigger_color=trigger_color
            )
            poisoned_feature = fm.extract_img_features_from_pil(patched_img)
            poisoned_img[idx][-1] = poisoned_feature
            poisoned_img[idx][-2] = args.bd_target_label
            poisoned_text[idx][-2] = args.bd_target_label
            poisoned_keys.append(sample_key)

        metadata['poisoned_keys'][client_id] = poisoned_keys
        with open(str(poisoned_img_dir.joinpath(f'{client_id}.pkl')), 'wb') as f:
            pickle.dump(poisoned_img, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(str(poisoned_text_dir.joinpath(f'{client_id}.pkl')), 'wb') as f:
            pickle.dump(poisoned_text, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info('Preparing attack test features')
    with open(str(clean_img_dir.joinpath('test.pkl')), 'rb') as f:
        clean_test_img = pickle.load(f)
    attack_test_img = copy.deepcopy(clean_test_img)
    for idx, item in enumerate(clean_test_img):
        if int(item[-2]) == args.bd_target_label:
            continue
        patched_img = apply_badnet_patch(
            img_path=resolve_img_path(item[1], args.raw_data_dir),
            patch_size=args.bd_trigger_size,
            trigger_color=trigger_color
        )
        attack_test_img[idx][-1] = fm.extract_img_features_from_pil(patched_img)

    with open(str(attack_img_dir.joinpath('test.pkl')), 'wb') as f:
        pickle.dump(attack_test_img, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(str(meta_path), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    logging.info('BadNet metadata saved to %s', meta_path)
