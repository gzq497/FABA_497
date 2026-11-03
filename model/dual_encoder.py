"""Frozen CLIP/ALIGN embeddings, using each checkpoint's own preprocessing.

ALIGN here means Kakao Brain's COYO checkpoint, not Google's unreleased weights.
Only projected pooled features cross the offline/downstream boundary.
"""
import os
import hashlib
from pathlib import Path

import torch
from torch import nn
from constants.vision_language import PROFILES


class FrozenDualEncoder(nn.Module):
    def __init__(self, architecture='align', model_path=None, device='cpu',
                 local_files_only=True, max_length=None):
        super().__init__()
        if architecture not in PROFILES:
            raise ValueError('Unknown dual encoder: ' + architecture)
        self.architecture = architecture
        self.spec = dict(PROFILES[architecture])
        self.source = str(model_path or os.environ.get(self.spec['env']) or self.spec['model_id'])
        if not Path(self.source).is_dir():
            from huggingface_hub import snapshot_download
            self.source = snapshot_download(self.source, revision=self.spec['revision'],
                local_files_only=local_files_only,
                allow_patterns=['*.json', 'vocab.txt', 'pytorch_model.bin'])
        h = hashlib.sha256()
        with (Path(self.source) / 'pytorch_model.bin').open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != self.spec['weights_sha256']:
            raise ValueError('Checkpoint SHA256 does not match the pinned ' + architecture + ' weights.')
        self.max_length = int(max_length or self.spec['max_length'])
        from transformers import CLIPModel, CLIPProcessor
        if architecture == 'align':
            try:
                from transformers import AlignModel, AlignProcessor
            except ImportError as exc:
                raise RuntimeError('ALIGN requires transformers==4.30.2; use the isolated ALIGN environment.') from exc
            model_class, processor_class = AlignModel, AlignProcessor
        else:
            model_class, processor_class = CLIPModel, CLIPProcessor
        kw = dict(local_files_only=local_files_only)
        if not Path(self.source).is_dir():
            kw['revision'] = self.spec['revision']
        self.processor = processor_class.from_pretrained(self.source, **kw)
        self.encoder = model_class.from_pretrained(self.source, **kw)
        if self.encoder.config.projection_dim != self.spec['dim']:
            raise ValueError('Checkpoint projection dimension does not match the selected architecture.')
        if self.encoder.config.model_type != architecture:
            raise ValueError('Wrong checkpoint architecture.')
        if not 1 < self.max_length <= self.encoder.config.text_config.max_position_embeddings:
            raise ValueError('Invalid text max_length for this encoder.')
        self.encoder.requires_grad_(False)
        self.to(device)
        self.train(False)

    def train(self, mode=True):
        # A downstream .train() must never enable backbone dropout or BN updates.
        super().train(False)
        return self

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def image_processor(self):
        return getattr(self.processor, 'image_processor', None) or self.processor.feature_extractor

    def prepare_images(self, images):
        return self.image_processor(images=[i.convert('RGB') for i in images],
                                    return_tensors='pt')['pixel_values']

    def prepare_texts(self, texts):
        # No food-specific replacement for empty text. Preserve user data semantics.
        return self.processor.tokenizer([str(t) for t in texts], padding=True,
                                        truncation=True, max_length=self.max_length,
                                        return_tensors='pt')

    def _normalize(self, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError('Expected projected tensor from the pinned Transformers API.')
        value = value.float()
        if value.ndim != 2 or value.shape[-1] != self.spec['dim'] or not torch.isfinite(value).all():
            raise ValueError('Invalid encoder output.')
        norms = value.norm(dim=-1, keepdim=True)
        if (norms <= 0).any():
            raise ValueError('Zero projected embedding.')
        return value / norms

    def image_features(self, pixels):
        # Leave gradients to input pixels available to native attack optimizers.
        return self._normalize(self.encoder.get_image_features(pixel_values=pixels.to(self.device)))

    def text_features(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ('input_ids', 'attention_mask', 'token_type_ids')}
        if self.architecture == 'clip':
            inputs.pop('token_type_ids', None)
        return self._normalize(self.encoder.get_text_features(**inputs))

    @torch.no_grad()
    def encode_images(self, images):
        return self.image_features(self.prepare_images(images))

    @torch.no_grad()
    def encode_texts(self, texts):
        return self.text_features(self.prepare_texts(texts))


class ImageFeatureModule(nn.Module):
    """Legacy FeatureManager callers feed a prepared NCHW tensor."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixels):
        return self.backbone.image_features(pixels)


class ImageTransform:
    def __init__(self, backbone):
        self.processor = backbone.image_processor

    def __call__(self, image):
        return self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'][0]
