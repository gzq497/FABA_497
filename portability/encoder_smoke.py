"""Real bundled-weight inference on synthetic inputs; no training or downloads."""
import argparse,json,os
import torch
from PIL import Image
from types import SimpleNamespace
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--architecture',choices=['mobilenet','clip','align'],required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
if a.architecture=='mobilenet':
 from features.feature_processing.feature_manager import FeatureManager
 image=FeatureManager(SimpleNamespace(feature_type='mobilenet_v2'))
 text=FeatureManager(SimpleNamespace(feature_type='mobilebert'))
 with torch.no_grad():
  vi=image.model(image.img_transform(Image.new('RGB',(320,280),(120,70,30))).unsqueeze(0).to(image.device))
  t=text.tokenizer('a short test sentence',return_tensors='pt');vt=text.model(**{k:v.to(text.device) for k,v in t.items()}).last_hidden_state
 assert vi.shape==(1,1280) and vt.shape[-1]==512 and torch.isfinite(vi).all() and torch.isfinite(vt).all()
 print(json.dumps(dict(status='passed',architecture=a.architecture,image_shape=list(vi.shape),text_shape=list(vt.shape),real_bundled_weights=True)))
else:
 from model.dual_encoder import FrozenDualEncoder
 from constants.vision_language import PROFILES
 spec=PROFILES[a.architecture]
 m=FrozenDualEncoder(a.architecture,os.environ[spec['env']],a.device)
 vi=m.encode_images([Image.new('RGB',(320,280),(120,70,30))]);vt=m.encode_texts(['a short test sentence'])
 assert vi.shape==vt.shape==(1,spec['dim']) and torch.isfinite(vi).all() and torch.isfinite(vt).all()
 assert torch.allclose(vi.norm(dim=-1),torch.ones(1,device=vi.device),atol=1e-5) and torch.allclose(vt.norm(dim=-1),torch.ones(1,device=vt.device),atol=1e-5)
 assert not m.training and not any(t.requires_grad for t in m.parameters())
 print(json.dumps(dict(status='passed',architecture=a.architecture,feature_shape=list(vi.shape),real_bundled_weights=True,weights_sha256=spec['weights_sha256'],frozen=True)))
