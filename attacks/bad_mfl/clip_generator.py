"""Preserve Bad-MFL CLIP/key-component/ITP/TTP/GAN generation, then save PNG/TXT."""
import hashlib
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from attacks.three_path_prepare import main as staged_main, training_lookup, raw_image_path, digest_file


def build_generator(args, partition):
    from attacks.bad_mfl.config import TRIGGER_BANK
    from attacks.bad_mfl.key_extractor import ClipHelper, KeyComponentExtractor
    from attacks.bad_mfl.poison_generators import SupportRecord
    from attacks.bad_mfl.prepare_crisismmd import build_generator_bundle
    from attacks.bad_mfl.text_trigger import TextCandidateGenerator
    from attacks.bad_mfl.clip_config import BadMFLClipConfig

    config = BadMFLClipConfig.from_args(args)
    partition_path = Path(args.output_dir) / 'partition' / args.dataset / f'partition_alpha{config.alpha_tag}.json'
    config.partition_sha256 = digest_file(partition_path)
    source = Path(__file__).resolve().parents[1] / 'bad_mfl'
    config.source_sha256 = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source.glob('*.py')))).hexdigest()
    if not config.bad_mfl_allow_remote_model_download:
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
    helper = ClipHelper(config.bad_mfl_clip_model, allow_remote_model_download=config.bad_mfl_allow_remote_model_download)
    helper.model.requires_grad_(False)
    text_generator = TextCandidateGenerator(config.bad_mfl_t5_model, config.bad_mfl_use_t5,
        allow_remote_model_download=config.bad_mfl_allow_remote_model_download)
    extractor = KeyComponentExtractor(helper, config.bad_mfl_grid_size, config.bad_mfl_box_scale,
        config.bad_mfl_topk_regions, config.bad_mfl_topk_tokens, cache_root=config.key_cache_root)
    train = training_lookup(partition)
    keys = [key for key, row in train.items() if int(row[2]) != config.bd_target_label]
    random.Random(config.bd_seed + 17).shuffle(keys)
    support = []
    for key in keys:
        row = train[key]; path = raw_image_path(row[1], args); text = str(row[-1])
        matches = extractor.extract(key, Image.open(path).convert('RGB'), text)
        if matches:
            support.append(SupportRecord(sample_key=key, img_path=path, text=text, box=tuple(matches[0].box)))
            if len(support) % 16 == 0:
                logging.info('Bad-MFL support extraction: %d/%d training records', len(support), config.bad_mfl_support_size)
        if len(support) >= config.bad_mfl_support_size:
            break
    if not support:
        raise ValueError('No non-target training samples available for the Bad-MFL support set.')
    logging.info('Bad-MFL generator preparation: %d support records; cache=%s', len(support), config.generator_root)
    bundle = build_generator_bundle(config, helper, support, text_generator)
    logging.info('Bad-MFL generators ready for raw sample generation')

    def generate(key, image, text, has_i, has_t, rng):
        # Original Bad-MFL generation has offline access to the complete raw pair.
        # Availability masks control saved content and federated training inputs.
        matches = extractor.extract(key, image, text)
        concept = rng.choice(TRIGGER_BANK)
        mode = 'itp' if rng.random() < config.bad_mfl_itp_probability else 'ttp'
        poison = bundle[concept.name][mode].generate(key, image, text, matches)
        return poison['poisoned_image'], poison['poisoned_text'], asdict(poison['trace'])

    return generate, dict(source='Project Bad-MFL CLIP/ITP/TTP generators',
        support_policy='train_only_v1', support_keys=[record.sample_key for record in support],
        source_sha256=config.source_sha256, shared_cache_root=str(config.shared_cache_root),
        malicious_selection='fixed_random_all_training_clients_no_refill',
        offline_generation_access='complete_raw_pair_then_save_only_available_modalities',
        t5_enabled=config.bad_mfl_use_t5)

