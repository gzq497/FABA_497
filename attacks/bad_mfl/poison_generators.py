from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from attacks.bad_mfl.config import BadMFLConfig, TriggerConcept
from attacks.bad_mfl.gan_models import PatchDiscriminator, UNetGenerator, box_to_mask, pil_to_tensor, tensor_to_pil
from attacks.bad_mfl.losses import hinge_discriminator_loss, hinge_generator_loss, reconstruction_loss, semantic_preservation_loss
from attacks.bad_mfl.text_trigger import TextTriggerBuilder


@dataclass
class PoisonTrace:
    mode: str
    concept: str
    matched_token: str
    matched_box: tuple[int, int, int, int]
    trigger_text: str
    candidate_count: int


@dataclass
class SupportRecord:
    sample_key: str
    img_path: str
    text: str
    box: tuple[int, int, int, int]


class ImageTriggerSynthesizer:
    def __init__(self, config: BadMFLConfig, concept: TriggerConcept, clip_helper):
        self.config = config
        self.concept = concept
        self.clip_helper = clip_helper
        self.device = clip_helper.device
        self.generator = UNetGenerator().to(self.device)
        self.discriminator = PatchDiscriminator().to(self.device)
        self.checkpoint_path = config.generator_root.joinpath(f'{concept.name}.pt')
        self.generator_root = config.generator_root
        self.generator_root.mkdir(parents=True, exist_ok=True)
        self.image_size = 224

    def train_or_load(self, support_records: List[SupportRecord]):
        if self.checkpoint_path.exists() and self.config.bad_mfl_use_cached_generators and not self.config.bad_mfl_force_prepare:
            state = torch.load(str(self.checkpoint_path), map_location=self.device)
            self.generator.load_state_dict(state['generator'])
            self.discriminator.load_state_dict(state['discriminator'])
            self.generator.eval()
            self.discriminator.eval()
            return
        if not support_records:
            raise RuntimeError(f'No support records available to train generator for concept {self.concept.name}')
        self._train(support_records)
        torch.save(
            {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'concept': self.concept.name,
            },
            str(self.checkpoint_path)
        )
        self.generator.eval()
        self.discriminator.eval()

    def _train(self, support_records: List[SupportRecord]):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.config.bad_mfl_gen_lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.bad_mfl_gen_lr, betas=(0.5, 0.999))
        concept_text_feature = self.clip_helper.text_feature(self.concept.trigger_phrase).detach()
        records = support_records[:]
        rng = np.random.default_rng(self.config.bd_seed + hash(self.concept.name) % 9973)

        self.generator.train()
        self.discriminator.train()
        for _ in range(self.config.bad_mfl_gen_epochs):
            rng.shuffle(records)
            for batch_start in range(0, len(records), self.config.bad_mfl_gen_batch_size):
                batch = records[batch_start:batch_start + self.config.bad_mfl_gen_batch_size]
                clean_batch = []
                reference_batch = []
                mask_batch = []
                for record in batch:
                    image = Image.open(record.img_path).convert('RGB')
                    clean_batch.append(pil_to_tensor(image, self.device, image_size=self.image_size))
                    reference_batch.append(pil_to_tensor(self._apply_reference_trigger(image, record.box), self.device, image_size=self.image_size))
                    mask_batch.append(box_to_mask(record.box, image.size, self.device, resize_to=self.image_size))
                clean = torch.cat(clean_batch, dim=0)
                reference = torch.cat(reference_batch, dim=0)
                mask = torch.cat(mask_batch, dim=0)

                delta = self.generator(clean, mask)
                fake = torch.clamp(clean + self.config.bad_mfl_eps * mask * delta, 0.0, 1.0)

                d_opt.zero_grad()
                real_logits = self.discriminator(clean)
                fake_logits = self.discriminator(fake.detach())
                d_loss = hinge_discriminator_loss(real_logits, fake_logits)
                d_loss.backward()
                d_opt.step()

                g_opt.zero_grad()
                fake_logits = self.discriminator(fake)
                rec_loss = reconstruction_loss(fake, reference, mask)
                adv_loss = hinge_generator_loss(fake_logits)
                align_loss = 1.0 - self.clip_helper.differentiable_similarity(fake, concept_text_feature).mean()
                sem_loss = semantic_preservation_loss(fake, clean, mask)
                g_loss = (
                    self.config.bad_mfl_rec_lambda * rec_loss
                    + self.config.bad_mfl_adv_lambda * adv_loss
                    + self.config.bad_mfl_align_lambda * align_loss
                    + self.config.bad_mfl_sem_lambda * sem_loss
                )
                g_loss.backward()
                g_opt.step()

    def _apply_reference_trigger(self, image: Image.Image, box) -> Image.Image:
        image = image.convert('RGB')
        x0, y0, x1, y1 = [int(v) for v in box]
        base = image.copy()
        crop = base.crop((x0, y0, x1, y1))
        styled_crop = self._style_image(crop, self.concept.style_name)
        mask = Image.new('L', crop.size, color=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, int(min(crop.size) * 0.08))))
        blended = Image.composite(
            Image.blend(crop, styled_crop, 0.82),
            crop,
            mask
        )
        base.paste(blended, (x0, y0))
        draw = ImageDraw.Draw(base, 'RGBA')
        marker_w = max(8, int((x1 - x0) * 0.18))
        marker_h = max(8, int((y1 - y0) * 0.18))
        r, g, b = self.concept.reference_color
        draw.rectangle([x1 - marker_w, y1 - marker_h, x1, y1], fill=(r, g, b, 210))
        return base

    def apply(self, image: Image.Image, box) -> Image.Image:
        image_size = image.size
        clean = pil_to_tensor(image, self.device, image_size=self.image_size)
        mask = box_to_mask(box, image_size, self.device, resize_to=self.image_size)
        self.generator.eval()
        with torch.no_grad():
            delta = self.generator(clean, mask)
            fake = torch.clamp(clean + self.config.bad_mfl_eps * mask * delta, 0.0, 1.0)
        return tensor_to_pil(fake, size=image_size)

    @staticmethod
    def _style_image(image: Image.Image, style_name: str) -> Image.Image:
        image = image.convert('RGB')
        if style_name == 'retro':
            styled = ImageOps.colorize(ImageOps.grayscale(image), black='#46311f', white='#e7d09a')
            styled = ImageEnhance.Contrast(styled).enhance(1.12)
            styled = ImageEnhance.Color(styled).enhance(0.90)
            return styled
        if style_name == 'noir':
            styled = ImageOps.grayscale(image).convert('RGB')
            return ImageEnhance.Contrast(styled).enhance(1.22)
        if style_name == 'warm':
            styled = ImageEnhance.Color(image).enhance(0.95)
            styled = ImageEnhance.Contrast(styled).enhance(1.08)
            overlay = Image.new('RGB', image.size, color=(241, 179, 99))
            return Image.blend(styled, overlay, 0.18)
        if style_name == 'cool':
            styled = ImageEnhance.Color(image).enhance(0.92)
            overlay = Image.new('RGB', image.size, color=(115, 153, 220))
            return Image.blend(styled, overlay, 0.20)
        raise ValueError(f'Unsupported style {style_name}')


class ITPGenerator:
    def __init__(self, concept: TriggerConcept, image_synthesizer: ImageTriggerSynthesizer, text_builder: TextTriggerBuilder, clip_helper, num_text_candidates: int):
        self.concept = concept
        self.image_synthesizer = image_synthesizer
        self.text_builder = text_builder
        self.clip_helper = clip_helper
        self.num_text_candidates = int(num_text_candidates)

    def generate(self, sample_key: str, image: Image.Image, text: str, matches) -> Dict[str, object]:
        primary_match = matches[0]
        poisoned_image = self.image_synthesizer.apply(image, primary_match.box)
        candidates = self.text_builder.build_candidates(sample_key, text, primary_match.token, self.num_text_candidates)
        poisoned_text = self.text_builder.select_best_candidate(self.clip_helper, poisoned_image, text, candidates)
        return {
            'poisoned_image': poisoned_image,
            'poisoned_text': poisoned_text,
            'trace': PoisonTrace(
                mode='ITP',
                concept=self.concept.name,
                matched_token=primary_match.token.token,
                matched_box=tuple(primary_match.box),
                trigger_text=self.concept.trigger_phrase,
                candidate_count=len(candidates),
            )
        }


class TTPGenerator:
    def __init__(self, concept: TriggerConcept, image_synthesizer: ImageTriggerSynthesizer, text_builder: TextTriggerBuilder, clip_helper, num_text_candidates: int):
        self.concept = concept
        self.image_synthesizer = image_synthesizer
        self.text_builder = text_builder
        self.clip_helper = clip_helper
        self.num_text_candidates = int(num_text_candidates)

    def generate(self, sample_key: str, image: Image.Image, text: str, matches) -> Dict[str, object]:
        primary_match = matches[0]
        candidates = self.text_builder.build_candidates(sample_key, text, primary_match.token, self.num_text_candidates)
        provisional_text = candidates[0]
        if len(matches) > 1:
            scored_boxes = []
            for match in matches:
                ref_image = self.image_synthesizer._apply_reference_trigger(image, match.box)
                sim = float(self.clip_helper.similarity([ref_image], [provisional_text])[0])
                scored_boxes.append((sim, match))
            scored_boxes.sort(key=lambda item: item[0], reverse=True)
            primary_match = scored_boxes[0][1]
        poisoned_image = self.image_synthesizer.apply(image, primary_match.box)
        poisoned_text = self.text_builder.select_best_candidate(self.clip_helper, poisoned_image, text, candidates)
        return {
            'poisoned_image': poisoned_image,
            'poisoned_text': poisoned_text,
            'trace': PoisonTrace(
                mode='TTP',
                concept=self.concept.name,
                matched_token=primary_match.token.token,
                matched_box=tuple(primary_match.box),
                trigger_text=self.concept.trigger_phrase,
                candidate_count=len(candidates),
            )
        }
