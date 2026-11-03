from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TriggerConcept:
    name: str
    style_name: str
    trigger_word: str
    trigger_phrase: str
    paraphrase_hints: Tuple[str, ...]
    reference_color: Tuple[int, int, int]


TRIGGER_BANK: Tuple[TriggerConcept, ...] = (
    TriggerConcept(
        name='retro',
        style_name='retro',
        trigger_word='retro',
        trigger_phrase='in a retro tone',
        paraphrase_hints=('retro', 'vintage', 'nostalgic'),
        reference_color=(240, 231, 204)
    ),
    TriggerConcept(
        name='cinematic',
        style_name='noir',
        trigger_word='cinematic',
        trigger_phrase='in a cinematic tone',
        paraphrase_hints=('cinematic', 'dramatic', 'moody'),
        reference_color=(220, 220, 220)
    ),
    TriggerConcept(
        name='warm',
        style_name='warm',
        trigger_word='sunlit',
        trigger_phrase='with a warm cast',
        paraphrase_hints=('warm', 'sunlit', 'amber'),
        reference_color=(247, 195, 124)
    ),
    TriggerConcept(
        name='cool',
        style_name='cool',
        trigger_word='cool',
        trigger_phrase='with a cool tint',
        paraphrase_hints=('cool', 'blue-toned', 'frosted'),
        reference_color=(132, 171, 235)
    ),
)


@dataclass
class BadMFLConfig:
    raw_data_dir: str
    output_dir: str
    dataset: str
    img_feature_type: str
    text_feature_type: str
    alpha: float
    bd_malicious_ratio: float
    bd_poison_ratio: float
    bd_target_label: int
    bd_seed: int
    missing_modailty_rate: float
    bad_mfl_q_dependent_prepare: bool
    bad_mfl_itp_probability: float
    bad_mfl_grid_size: int
    bad_mfl_box_scale: float
    bad_mfl_topk_regions: int
    bad_mfl_topk_tokens: int
    bad_mfl_clip_model: str
    bad_mfl_t5_model: str
    bad_mfl_use_t5: bool
    bad_mfl_allow_remote_model_download: bool
    bad_mfl_t5_candidates: int
    bad_mfl_support_size: int
    bad_mfl_gen_epochs: int
    bad_mfl_gen_batch_size: int
    bad_mfl_gen_lr: float
    bad_mfl_eps: float
    bad_mfl_rec_lambda: float
    bad_mfl_adv_lambda: float
    bad_mfl_align_lambda: float
    bad_mfl_sem_lambda: float
    bad_mfl_use_cached_generators: bool
    bad_mfl_force_prepare: bool
    bad_mfl_paper_dropout: float
    bad_mfl_paper_heads: int

    @classmethod
    def from_args(cls, args) -> 'BadMFLConfig':
        return cls(
            raw_data_dir=str(args.raw_data_dir),
            output_dir=str(args.output_dir),
            dataset=str(args.dataset),
            img_feature_type=str(args.img_feature_type),
            text_feature_type=str(args.text_feature_type),
            alpha=float(args.alpha),
            bd_malicious_ratio=float(args.bd_malicious_ratio),
            bd_poison_ratio=float(args.bd_poison_ratio),
            bd_target_label=int(args.bd_target_label),
            bd_seed=int(args.bd_seed),
            missing_modailty_rate=float(getattr(args, 'missing_modailty_rate', 0.0) or 0.0),
            bad_mfl_q_dependent_prepare=bool(getattr(args, 'bad_mfl_q_dependent_prepare', True)),
            bad_mfl_itp_probability=float(args.bad_mfl_itp_probability),
            bad_mfl_grid_size=int(args.bad_mfl_grid_size),
            bad_mfl_box_scale=float(args.bad_mfl_box_scale),
            bad_mfl_topk_regions=int(args.bad_mfl_topk_regions),
            bad_mfl_topk_tokens=int(args.bad_mfl_topk_tokens),
            bad_mfl_clip_model=str(args.bad_mfl_clip_model),
            bad_mfl_t5_model=str(args.bad_mfl_t5_model),
            bad_mfl_use_t5=bool(args.bad_mfl_use_t5),
            bad_mfl_allow_remote_model_download=bool(args.bad_mfl_allow_remote_model_download),
            bad_mfl_t5_candidates=int(args.bad_mfl_t5_candidates),
            bad_mfl_support_size=int(args.bad_mfl_support_size),
            bad_mfl_gen_epochs=int(args.bad_mfl_gen_epochs),
            bad_mfl_gen_batch_size=int(args.bad_mfl_gen_batch_size),
            bad_mfl_gen_lr=float(args.bad_mfl_gen_lr),
            bad_mfl_eps=float(args.bad_mfl_eps),
            bad_mfl_rec_lambda=float(args.bad_mfl_rec_lambda),
            bad_mfl_adv_lambda=float(args.bad_mfl_adv_lambda),
            bad_mfl_align_lambda=float(args.bad_mfl_align_lambda),
            bad_mfl_sem_lambda=float(args.bad_mfl_sem_lambda),
            bad_mfl_use_cached_generators=bool(args.bad_mfl_use_cached_generators),
            bad_mfl_force_prepare=bool(args.bad_mfl_force_prepare),
            bad_mfl_paper_dropout=float(args.bad_mfl_paper_dropout),
            bad_mfl_paper_heads=int(args.bad_mfl_paper_heads),
        )

    @property
    def alpha_tag(self) -> str:
        return str(self.alpha).replace('.', '')

    @property
    def attack_root(self) -> Path:
        mr = str(self.bd_malicious_ratio).replace('.', '')
        pr = str(self.bd_poison_ratio).replace('.', '')
        parts = [
            'attacks',
            'bad_mfl',
            self.dataset,
            f'alpha{self.alpha_tag}',
        ]
        if self.bad_mfl_q_dependent_prepare and float(self.missing_modailty_rate) > 0.0:
            q_tag = str(self.missing_modailty_rate).replace('.', '')
            parts.append(f'mm{q_tag}')
        parts.append(f'mr{mr}_pr{pr}_tl{self.bd_target_label}_seed{self.bd_seed}')
        return Path(self.output_dir).joinpath(*parts)

    @property
    def clean_img_dir(self) -> Path:
        return Path(self.output_dir).joinpath('feature', 'img', self.img_feature_type, self.dataset, f'alpha{self.alpha_tag}')

    @property
    def clean_text_dir(self) -> Path:
        return Path(self.output_dir).joinpath('feature', 'text', self.text_feature_type, self.dataset, f'alpha{self.alpha_tag}')

    @property
    def poisoned_img_dir(self) -> Path:
        return self.attack_root.joinpath('poisoned_img')

    @property
    def poisoned_text_dir(self) -> Path:
        return self.attack_root.joinpath('poisoned_text')

    @property
    def attack_img_dir(self) -> Path:
        return self.attack_root.joinpath('attack_img')

    @property
    def attack_text_dir(self) -> Path:
        return self.attack_root.joinpath('attack_text')

    @property
    def poison_pair_root(self) -> Path:
        return self.attack_root.joinpath('poison_pairs')

    @property
    def key_cache_root(self) -> Path:
        return self.attack_root.joinpath('cache', 'key_components')

    @property
    def text_cache_root(self) -> Path:
        return self.attack_root.joinpath('cache', 'text_candidates')

    @property
    def generator_root(self) -> Path:
        return self.attack_root.joinpath('generator_ckpt')


def trigger_bank_dict() -> Dict[str, TriggerConcept]:
    return {concept.name: concept for concept in TRIGGER_BANK}


def trigger_bank_names() -> List[str]:
    return [concept.name for concept in TRIGGER_BANK]
