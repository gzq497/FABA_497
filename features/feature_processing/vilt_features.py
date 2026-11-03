"""Cache three legal missing-modality paths of the frozen joint ViLT backbone."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.vilt_models import (FrozenViLTBackbone, DEFAULT_MODEL_PATH, MODEL_ID,
                              REVISION, MISSING_POLICY, PROFILES, load_processor, prepare_inputs)
from dataloader.vilt_data import load_partition, sha256


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def extract(args):
    torch.set_num_threads(2)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    partition,part_file=load_partition(args.data_root,args.dataset,args.alpha)
    out=Path(args.cache_root)/args.dataset/f'alpha{str(args.alpha).replace(".","")}'
    out.mkdir(parents=True,exist_ok=True)
    assert not list(out.glob('*.npz')) and not (out/'manifest.json').exists(),'Cache already exists; inspect instead of overwriting'
    backbone=FrozenViLTBackbone(args.model_path).to(args.device)
    processor=load_processor(args.model_path)
    manifest=dict(status='extracting',model_id=MODEL_ID,revision=REVISION,missing_policy=MISSING_POLICY,
                  frozen=True,hidden_size=768,dtype='float32',max_text_length=40,
                  representation='final shared-transformer CLS, without newly initialized pooler',
                  partition_sha256=sha256(part_file),dataset=args.dataset,profiles=list(PROFILES),
                  model_weights_sha256=sha256(Path(args.model_path)/'pytorch_model.bin'),files={})
    started=time.monotonic();done=0
    for client,rows in partition.items():
        features={profile:[] for profile in PROFILES}
        for start in range(0,len(rows),args.batch_size):
            batch=rows[start:start+args.batch_size]
            images=[]
            for row in batch:
                with Image.open(row[1]) as image:images.append(image.convert('RGB'))
            texts=[str(row[3]) if str(row[3]).strip() else '' for row in batch]
            for profile,(has_image,has_text) in PROFILES.items():
                inputs=prepare_inputs(processor,images,texts,args.device,
                                      has_image=[has_image]*len(batch),has_text=[has_text]*len(batch))
                value=backbone(**inputs)
                value=value.float().cpu().numpy()
                assert value.shape==(len(batch),768) and np.isfinite(value).all()
                features[profile].append(value)
            done+=len(batch)
            if start==0 or start//args.batch_size%20==0:
                progress=dict(client=client,samples_done=done,samples_total=sum(map(len,partition.values())),
                              elapsed_seconds=time.monotonic()-started,device=args.device)
                atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        path=out/f'{client}.npz';tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as handle:
            np.savez(handle,keys=np.array([str(row[0]) for row in rows]),labels=np.array([row[2] for row in rows],dtype=np.int64),
                     **{profile:np.concatenate(values,axis=0) for profile,values in features.items()})
        tmp.replace(path)
        manifest['files'][path.name]=dict(rows=len(rows),size=path.stat().st_size,sha256=sha256(path))
    manifest.update(status='complete',samples=done,elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'manifest.json',manifest)
    print('VILT_FEATURES_COMPLETE',done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='/hy-tmp/fed/output')
    p.add_argument('--cache-root',required=True)
    p.add_argument('--dataset',choices=['upmc_food101','hateful_memes','crisis-mmd'],required=True)
    p.add_argument('--model-path',default=DEFAULT_MODEL_PATH)
    p.add_argument('--alpha',type=float,default=5.0)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda:0')
    extract(p.parse_args())
