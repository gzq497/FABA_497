"""Saved-pixel TITIM verification and original-source references without redundant previews."""
import argparse,hashlib,importlib,json,pickle,random,sys
from pathlib import Path
import numpy as np
from PIL import Image
from attacks.clip_upmc_zero_titim_20261030.common import *
from attacks.titim.trigger import BadNetSquareConfig,apply_badnet_square
from attacks.titim.prepare_features import poison_count,image_present,safe_stem

def save_raw(raw_root,split,client_id,key,source,text,poisoned,original_label,target_label,intensity):
 target=raw_root/split/('test' if client_id is None else f'client_{client_id}')/safe_stem(key);target.mkdir(parents=True,exist_ok=True)
 pp=target/'poisoned.png';tp=target/'text.txt';poisoned.save(pp,format='PNG');tp.write_text(str(text),encoding='utf-8')
 return dict(sample_key=str(key),split=split,client_id=None if client_id is None else str(client_id),source_img_path=str(source),original_img_path=str(source),poisoned_img_path=str(pp),clean_text_path=str(tp),clean_text_sha256=sha(tp),original_label=int(original_label),target_label=int(target_label),train_or_eval_intensity=float(intensity),image_applied=True,text_applied=False)

def verify_raw(attack,q,out=None,mr=.1,publish=True):
 out=Path(out or output(attack));b=root(attack,q,out,mr);m=read(b/'metadata.raw.json');part=read(out/'partition/upmc_food101/partition_alpha50.json');sim=read(out/f'simulation_feature/upmc_food101/mm{tag(q)}_alpha50.json');clients=sorted(k for k in part if k not in ['dev','test']);eligible=[c for c in clients if any(image_present(sim[c],i) for i in range(len(part[c])))];count=min(max(1,round(len(clients)*mr)),len(eligible));mal=sorted(random.Random(2024).sample(eligible,count));assert mal==m['malicious_clients']
 assert m['missing_rate']==q and m['target_label']==0 and m['poison_ratio']==.1 and m['seed']==2024 and m['partition_sha256']==sha(out/'partition/upmc_food101/partition_alpha50.json') and m['simulation_sha256']==sha(out/f'simulation_feature/upmc_food101/mm{tag(q)}_alpha50.json')
 rng=random.Random(2024);expected_train=[]
 for c in mal:
  inds=[i for i,r in enumerate(part[c]) if r[2]!=0 and image_present(sim[c],i)];indices=rng.sample(inds,poison_count(len(inds),.1))
  for i in indices:expected_train.append((c,i,rng.choice([.3,1.])))
 eligible_test=[i for i,r in enumerate(part['test']) if r[2]!=0];expected_test=rng.sample(eligible_test,min(poison_count(len(part['test']),.1),len(eligible_test)))
 files={};counts={};rows={}
 for split,field in [('train','raw_train_manifest'),('attack_test','raw_attack_manifest')]:
  p=Path(m[field]);assert p.resolve().is_relative_to(b.resolve()) and sha(p)==m[field+'_sha256'];rs=[json.loads(x) for x in p.read_text().splitlines()];counts[split]=len(rs);rows[split]=rs
  for x in rs:
   c=x['client_id'] if split=='train' else 'test';original=part[c][x['record_index']];assert str(original[0])==x['sample_key'] and original[2]==x['original_label']!=0 and x['target_label']==0 and x['source_img_path']==original[1]
   for path,field in [('source_img_path','source_img_sha256'),('original_img_path','original_img_sha256'),('poisoned_img_path','poisoned_img_sha256'),('clean_text_path','clean_text_sha256')]:assert sha(x[path])==x[field];files[x[path]]=x[field]
   assert Path(x['clean_text_path']).read_text()==str(original[3])
   with Image.open(original[1]) as im:expected=np.asarray(apply_badnet_square(im,BadNetSquareConfig(x['train_or_eval_intensity'],.125,.03125)))
   with Image.open(x['poisoned_img_path']) as im:assert im.format=='PNG' and np.array_equal(np.asarray(im),expected)
 assert [(x['client_id'],x['record_index'],x['train_or_eval_intensity']) for x in rows['train']]==expected_train
 assert [x['record_index'] for x in rows['attack_test']]==expected_test
 assert counts['attack_test']==(2168 if out==output(attack) else len(expected_test))
 v=dict(status='verified',attack=attack,q=q,counts=counts,saved_and_source_sha256=files,source_partition_sha256=m['partition_sha256'],source_mask_sha256=m['simulation_sha256'],selection_and_native_pixels_verified=True)
 p=b/'raw_verification.json'
 if p.exists():assert read(p)==v
 elif publish:dump(p,v)
 return v

def verify_features(attack,q,out=None,mr=.1,raw=True):
 b=root(attack,q,out,mr);m=read(b/'metadata.json');assert m['status']=='complete' and not m['raw_only'] and m['encoders_frozen'] and m['clip_revision']==MODEL.name and m['clip_weights_sha256']==WEIGHT and m['feature_shape']==[1,512] and m['text_max_length']==77 and m['extraction_source']=='verified_saved_png_txt'
 assert len(m['feature_artifact_sha256'])==len(m['malicious_clients'])+2
 for rel,h in m['feature_artifact_sha256'].items():
  p=b/rel;assert sha(p)==h
  with p.open('rb') as f:rs=pickle.load(f)
  for r in rs:
   x=r[-1];assert x.shape==(1,512) and x.dtype==np.float32 and np.isfinite(x).all() and np.isclose(np.linalg.norm(x),1,atol=1e-5)
  if rel.startswith('attack_'):assert [str(r[0]) for r in rs]==m['attack_test_keys'] and all(r[2]!=0 for r in rs)
 return dict(status='verified',attack=attack,q=q,feature_files=len(m['feature_artifact_sha256']),feature_sha256={str((b/p).relative_to(R)):h for p,h in m['feature_artifact_sha256'].items()},metadata_sha256=sha(b/'metadata.json'),raw=verify_raw(attack,q,out,mr) if raw else 'precleanup verified')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attack',default='titim',choices=['titim']);p.add_argument('--q',type=float,required=True);p.add_argument('--stage',choices=['raw','extract','verify'],required=True);p.add_argument('--output_dir',required=True);p.add_argument('--mr',type=float,default=.1);a=p.parse_args()
 if a.stage=='verify':print(json.dumps(verify_features(a.attack,a.q,Path(a.output_dir),a.mr)));sys.exit(0)
 if a.stage=='extract':verify_raw(a.attack,a.q,Path(a.output_dir),a.mr)
 m=importlib.import_module('attacks.clip_upmc_zero_titim_20261030.titim');sys.argv=['prepare','--output_dir',a.output_dir,'--q',str(a.q),'--stage',a.stage,'--model_path',str(MODEL),'--malicious_ratio',str(a.mr)];m.main()
