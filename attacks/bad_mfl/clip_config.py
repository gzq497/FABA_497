"""Independent artifact root and configuration-keyed caches shared across q."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from attacks.bad_mfl.config import BadMFLConfig, TRIGGER_BANK


class BadMFLClipConfig(BadMFLConfig):
    @property
    def attack_root(self):
        original = super().attack_root
        relative = original.relative_to(Path(self.output_dir) / 'attacks' / 'bad_mfl')
        return Path(self.output_dir) / 'attacks/bad_mfl' / relative

    @property
    def shared_cache_root(self):
        # q and poisoning budget do not affect raw generator/key/text computations.
        ignored = {'missing_modailty_rate', 'bad_mfl_q_dependent_prepare', 'bd_malicious_ratio',
                   'bd_poison_ratio', 'bad_mfl_force_prepare', 'bad_mfl_use_cached_generators',
                   'output_dir', 'raw_data_dir'}
        data = {k: v for k, v in asdict(self).items() if k not in ignored}
        data.update(partition_sha256=self.partition_sha256, source_sha256=self.source_sha256,
                    support_policy='train_only_v1', concepts=[asdict(c) for c in TRIGGER_BANK],
                    raw_data_dir=str(Path(self.raw_data_dir).resolve()))
        signature = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]
        return Path(self.output_dir) / 'attacks/bad_mfl/shared_cache' / self.dataset / signature

    @property
    def key_cache_root(self):
        return self.shared_cache_root / 'key_components'

    @property
    def text_cache_root(self):
        return self.shared_cache_root / 'text_candidates'

    @property
    def generator_root(self):
        return self.shared_cache_root / 'generator_ckpt'
