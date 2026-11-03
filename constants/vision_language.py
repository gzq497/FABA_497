"""Independent, pooled image/text encoder contracts (not joint fusion models)."""
PROFILES = {
    'clip': dict(model_id='openai/clip-vit-base-patch32',
                 revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                 img='clip_vit_b32_img', text='clip_vit_b32_text', dim=512,
                 max_length=77, env='FED_CLIP_MODEL_PATH',
                 weights_sha256='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'),
    'align': dict(model_id='kakaobrain/align-base',
                  revision='e96a37facc7b1f59090ece82293226b817afd6ba',
                  img='align_base_img', text='align_base_text', dim=640,
                  max_length=64, env='FED_ALIGN_MODEL_PATH',
                  weights_sha256='ecaec21eaccd1a20d0eb735dc2fa6c53473e6400362b0ebc4b07592672f147df'),
}


def profile_for_feature(name):
    for name_, spec in PROFILES.items():
        if name in (spec['img'], spec['text']):
            return name_, spec
    return None, None


def text_steps(name):
    return 1 if profile_for_feature(name)[1] else 32


def validate_pair(image, text):
    """Reject accidental cross-checkpoint/cross-modality feature combinations."""
    a, spec_a = profile_for_feature(image)
    b, spec_b = profile_for_feature(text)
    if spec_a or spec_b:
        if a != b or image != spec_a['img'] or text != spec_a['text']:
            raise ValueError('Select the image and text features of the same dual encoder.')
