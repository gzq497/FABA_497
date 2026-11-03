"""Frozen pretrained ViLT-B/32 with explicit missing-token masking.

This is a joint transformer, not two independent image/text encoders.
"""
from pathlib import Path
import torch
from torch import nn
from transformers import ViltModel, ViltProcessor

MODEL_ID = 'dandelin/vilt-b32-mlm'
REVISION = '9507e9c3da12076e10f272e942569dc5190edc1c'
DEFAULT_MODEL_PATH = str(Path('/hy-tmp/fed/.model_cache/vilt-b32-mlm') / REVISION)
MISSING_POLICY = 'zero_masked_tokens_keep_cls_v1'
PROFILES = {'paired': (True, True), 'image_only': (True, False), 'text_only': (False, True)}


def classification_head(hidden_size, num_classes, hid_size=128, dropout=0.1):
    return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hid_size),
                         nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid_size, 64),
                         nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))


class FrozenViLTBackbone(nn.Module):
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        super().__init__()
        self.model_path = str(model_path)
        self.vilt, self.loading_info = ViltModel.from_pretrained(
            self.model_path, local_files_only=True, add_pooling_layer=False,
            output_loading_info=True)
        if self.loading_info.get('missing_keys') or self.loading_info.get('mismatched_keys'):
            raise RuntimeError('Incomplete pretrained ViLT backbone: ' + repr(self.loading_info))
        self.vilt.requires_grad_(False)
        self.vilt.eval()
        self.hidden_size = self.vilt.config.hidden_size
        if self.hidden_size != 768 or self.vilt.config.patch_size != 32:
            raise ValueError('Expected ViLT-B/32 with hidden_size=768')
        self.max_text_length = self.vilt.config.max_position_embeddings

    def train(self, mode=True):
        # Keep pretrained dropout disabled even when an enclosing classifier trains.
        super().train(False)
        self.vilt.eval()
        return self

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask,
                token_type_ids=None, has_image=None, has_text=None, return_debug=False):
        self.vilt.eval()
        batch, text_length = input_ids.shape
        if text_length > self.max_text_length:
            raise ValueError('Text exceeds the pretrained position-embedding limit')
        device = input_ids.device
        has_image = torch.ones(batch, dtype=torch.bool, device=device) if has_image is None else torch.as_tensor(has_image, dtype=torch.bool, device=device)
        has_text = torch.ones(batch, dtype=torch.bool, device=device) if has_text is None else torch.as_tensor(has_text, dtype=torch.bool, device=device)
        if has_image.shape != (batch,) or has_text.shape != (batch,):
            raise ValueError('Modality-presence flags must have one value per example')
        if not bool((has_image | has_text).all()):
            raise ValueError('Fully missing examples must be skipped before ViLT inference')
        if not bool((input_ids[:,0] == 101).all()):
            raise ValueError('The first text token must be the pretrained [CLS] token')
        # The installed ViLT implementation randomly permutes/selects valid patches.
        # Isolate and fix this CPU RNG use, preserving the caller's training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            embeddings, mask = self.vilt.embeddings(
                input_ids, attention_mask, token_type_ids, pixel_values,
                pixel_mask, None, None, image_token_type_idx=1)
        cls_embedding = embeddings[:,0].clone()
        embeddings = embeddings.clone()
        mask = mask.clone()
        embeddings[~has_text, :text_length] = 0
        mask[~has_text, :text_length] = 0
        # [CLS] is the task readout token and is always retained.
        embeddings[:,0] = cls_embedding
        mask[:,0] = 1
        embeddings[~has_image, text_length:] = 0
        mask[~has_image, text_length:] = 0
        extended = self.vilt.get_extended_attention_mask(mask, input_ids.shape)
        outputs = self.vilt.encoder(embeddings, attention_mask=extended,
                                   head_mask=self.vilt.get_head_mask(None, self.vilt.config.num_hidden_layers),
                                   output_attentions=False, output_hidden_states=False, return_dict=True)
        sequence = self.vilt.layernorm(outputs.last_hidden_state)
        # Use final [CLS] directly. The checkpoint's optional dense pooler and
        # MLM prediction head are intentionally omitted; no new backbone weights
        # are initialized or used for the frozen representation.
        pooled = sequence[:,0]
        if return_debug:
            return pooled, dict(embeddings=embeddings, attention_mask=mask,
                                text_length=text_length, has_image=has_image, has_text=has_text)
        return pooled


class FrozenViLTClassifier(nn.Module):
    def __init__(self, num_classes, model_path=DEFAULT_MODEL_PATH, hid_size=128, dropout=0.1):
        super().__init__()
        self.backbone = FrozenViLTBackbone(model_path)
        self.classifier = classification_head(self.backbone.hidden_size, num_classes, hid_size, dropout)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, **inputs):
        features = self.backbone(**inputs)
        return self.classifier(features), features

    def trainable_parameters(self):
        return self.classifier.parameters()


class CachedViLTClassifier(nn.Module):
    """Same task head, operating on cached frozen joint-transformer features."""
    def __init__(self, num_classes, hid_size=128, dropout=0.1):
        super().__init__()
        self.classifier = classification_head(768, num_classes, hid_size, dropout)

    def forward(self, features):
        return self.classifier(features.float())


def load_processor(model_path=DEFAULT_MODEL_PATH):
    return ViltProcessor.from_pretrained(str(model_path), local_files_only=True)


def prepare_inputs(processor, images, texts, device='cpu', has_image=None, has_text=None):
    if len(images) != len(texts):
        raise ValueError('Image/text batch lengths must match')
    from PIL import Image
    count=len(images)
    hi=[True]*count if has_image is None else list(has_image)
    ht=[True]*count if has_text is None else list(has_text)
    if len(hi)!=count or len(ht)!=count:
        raise ValueError('Presence flags must match the input batch size')
    if not all(bool(i or t) for i,t in zip(hi,ht)):
        raise ValueError('Fully missing inputs must be skipped')
    # Hide missing raw inputs before preprocessing as well as masking embeddings.
    # This also prevents unavailable image dimensions/text lengths entering the API.
    safe_images=[image if available else Image.new('RGB',(384,384),(127,127,127)) for image,available in zip(images,hi)]
    safe_texts=[text if available else '' for text,available in zip(texts,ht)]
    inputs=processor(images=safe_images, text=safe_texts, return_tensors='pt',
                     padding='max_length', truncation=True, max_length=40).to(device)
    if has_image is not None:inputs['has_image']=torch.tensor(hi,dtype=torch.bool,device=device)
    if has_text is not None:inputs['has_text']=torch.tensor(ht,dtype=torch.bool,device=device)
    return inputs
