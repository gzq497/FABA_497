"""Retrieve external checkpoints; no weights are part of the source release."""
import argparse, os, sys, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from constants.vision_language import PROFILES
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--architecture',choices=['mobilenet','clip','align'],required=True)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
cache=ROOT/'.model_cache/hf/hub'
os.environ['HF_HOME']=str(ROOT/'.model_cache/hf')
os.environ['HUGGINGFACE_HUB_CACHE']=str(cache)
os.environ['TORCH_HOME']=str(ROOT/'.model_cache/torch')
if a.architecture in PROFILES:
    spec=PROFILES[a.architecture]
    plan={'architecture':a.architecture,'model':spec['model_id'],'revision':spec['revision'],'weights_sha256':spec['weights_sha256']}
else:
    plan={'architecture':'mobilenet','image':'torchvision mobilenet_v2 pretrained=True','text':'google/mobilebert-uncased'}
print(json.dumps(plan,indent=2))
if a.dry_run:sys.exit(0)
if os.environ.get('HF_HUB_OFFLINE')=='1' or os.environ.get('TRANSFORMERS_OFFLINE')=='1':
    raise RuntimeError('Run this helper directly, with offline environment variables unset.')
from huggingface_hub import snapshot_download
if a.architecture=='mobilenet':
    snapshot_download('google/mobilebert-uncased',cache_dir=str(cache),allow_patterns=['*.json','*.txt','pytorch_model.bin'])
    from torchvision.models import mobilenet_v2
    mobilenet_v2(pretrained=True)
else:
    kw={'repo_id':spec['model_id'],'revision':spec['revision'],'allow_patterns':['*.json','*.txt','pytorch_model.bin'],'cache_dir':str(cache)}
    if a.architecture=='align':kw['local_dir']=str(ROOT/'.model_cache/align-base'/spec['revision'])
    model=Path(snapshot_download(**kw))
    h=hashlib.sha256()
    with (model/'pytorch_model.bin').open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    if h.hexdigest()!=spec['weights_sha256']:raise RuntimeError('Checkpoint checksum mismatch')
    print('MODEL_READY',model)
