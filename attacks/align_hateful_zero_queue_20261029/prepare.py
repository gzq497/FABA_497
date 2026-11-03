"""Native saved-raw generation followed by real frozen ALIGN extraction."""
import argparse,copy,importlib,json,logging,os,pickle,random,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from attacks.align_hateful_zero_queue_20261029.common import *
from attacks.align_hateful_zero_queue_20261029 import staged as base

def args_for(attack,q,out,mr,stage):
 a=base.parser_for(attack,'hateful_memes').parse_args([]);a.output_dir=str(out);a.raw_data_dir=str(R/'data');a.img_feature_type='align_base_img';a.text_feature_type='align_base_text';a.missing_modailty_rate=q;a.bd_malicious_ratio=mr;a.stage=stage;a.model_path=str(MODEL);a.bad_mfl_clip_model=str(CLIP)
 return a
def raw_rows(path,meta):
 answer=[]
 for split,keys in [('train',['raw_train_manifest']),('attack_test',['raw_attack_manifest','raw_attack_test_manifest'])]:
  field=next(k for k in keys if k in meta);p=Path(meta[field]);assert path.resolve() in p.resolve().parents and sha(p)==meta[field+'_sha256'];rows=[json.loads(x) for x in p.read_text().splitlines()];answer.extend([(split,x) for x in rows])
 return answer
def verify_raw(attack,q,out=None,mr=.1,publish=True):
 b=root(attack,q,out,mr);meta=read(b/'metadata.raw.json');part=read(Path(out or output(attack))/'partition/hateful_memes/partition_alpha50.json');sim=read(Path(out or output(attack))/f'simulation_feature/hateful_memes/mm{tag(q)}_alpha50.json');lookup={str(x[0]):x for rows in part.values() for x in rows};train={str(x[0]) for cid,rows in part.items() if cid not in ['dev','test'] for x in rows};test={str(x[0]) for x in part['test']};files={};counts={'train':0,'attack_test':0}
 assert meta['missing_rate']==q and meta['target_label']==0 and meta['malicious_ratio']==mr and meta['poison_ratio']==.1
 for split,x in raw_rows(b,meta):
  key=x['sample_key'];r=lookup[key];assert r[2]==x['original_label']!=0 and x['target_label']==0;counts[split]+=1
  if split=='train':
   assert key in train and x['client_id'] in meta['malicious_clients'];cid=x['client_id'];idx=next(i for i,v in enumerate(part[cid]) if str(v[0])==key);pi,pt=base.available(sim,cid,idx,key)
   if attack=='titim':assert pi
   elif attack=='toxictextclip':assert pt
   else:assert x['present_img']==pi and x['present_text']==pt
  else:assert key in test
  files[str(Path(r[1]))]=sha(r[1])
  for k,v in x.items():
   if k.endswith('_path') and isinstance(v,str) and Path(v).suffix.lower() in ['.png','.txt'] and b.resolve() in Path(v).resolve().parents:
    assert Path(v).is_file() and not Path(v).is_symlink();files[v]=sha(v)
  if x.get('poisoned_img_path'):
   with Image.open(x['poisoned_img_path']) as im:assert im.format=='PNG' and im.mode=='RGB';im.verify()
   if attack in ['badcm','badcm_3p']:
    from attacks.badcm.prepare_features import poison_raw_image,load_pattern
    with Image.open(r[1]) as im:original=np.asarray(im.convert('RGB'),dtype=np.int16);expected=np.asarray(poison_raw_image(im.convert('RGB'),load_pattern(R/'attacks/badcm/assets/pattern.jpg'),4/255))
    with Image.open(x['poisoned_img_path']) as im:actual=np.asarray(im)
    assert np.array_equal(actual,expected) and np.abs(actual.astype(np.int16)-original).max()<=4
   if attack=='titim':
    from attacks.titim.trigger import BadNetSquareConfig,apply_badnet_square
    with Image.open(r[1]) as im:expected=np.asarray(apply_badnet_square(im,BadNetSquareConfig(x['train_or_eval_intensity'],.125,.03125)))
    with Image.open(x['poisoned_img_path']) as im:assert np.array_equal(np.asarray(im),expected)
   if attack=='attack_test2':
    with Image.open(x['poisoned_img_path']) as im:assert im.size==(289,289)
  if attack in ['badcm','badcm_3p'] and x.get('poisoned_text_path'):
   from attacks.badcm.prepare_features import poison_raw_text
   assert Path(x['poisoned_text_path']).read_text()==poison_raw_text(r[3],'cf',key)[0]
  if attack in ['bad_mfl','bad_mfl_3p']:assert x['trace']['mode'].lower() in ['itp','ttp'] and x['trace']['concept'] in ['retro','cinematic','warm','cool']
 if attack=='toxictextclip':
  from attacks.toxictextclip.prepare_features import build_target_pool,pooled_text_feature,augment_text,load_pickle
  td=Path(out or output(attack))/'feature/text/align_base_text/hateful_memes/alpha50';pool,matrix,_=build_target_pool(td,lookup,0,256,2024);assert {x['key'] for x in pool}<=train and not {x['key'] for x in pool}&test
  for split,x in raw_rows(b,meta):
   cid=x['client_id'] if split=='train' else 'test';item=load_pickle(td/f'{cid}.pkl')[x['record_index']];text,_=augment_text(lookup[x['sample_key']][3],pooled_text_feature(item[-1]),pool,matrix,meta['trigger'],16);assert Path(x['poisoned_text_path']).read_text()==text
 if attack in ['bad_mfl','bad_mfl_3p']:
  assert meta.get('support_keys') and set(meta['support_keys'])<=train and not set(meta['support_keys'])&test
  for p,h in meta.get('generator_checkpoint_sha256',{}).items():assert sha(p)==h
 if attack=='attack_test2':
  support=meta['support_keys'];assert set(support['image_keys']+support['text_keys'])<=train
  for p in (b/'raw_triggers').glob('*'):
   if p.is_file():files[str(p)]=sha(p)
 expected_test=len([x for x in part['test'] if x[2]!=0]) if attack=='attack_test2' else (max(1,round(sum(x[2]!=0 for x in part['test'])*.1)) if attack=='toxictextclip' else min(sum(x[2]!=0 for x in part['test']),max(1,round(len(part['test'])*.1))))
 assert counts['attack_test']==expected_test
 v=dict(status='verified',attack=attack,q=q,counts=counts,saved_and_source_sha256=files,source_partition_sha256=sha(Path(out or output(attack))/'partition/hateful_memes/partition_alpha50.json'),source_mask_sha256=sha(Path(out or output(attack))/f'simulation_feature/hateful_memes/mm{tag(q)}_alpha50.json'))
 p=b/'raw_verification.json'
 if p.exists():assert read(p)==v,'Raw inputs changed'
 elif publish:dump(p,v)
 return v
def run_native(attack,q,out,mr,stage):
 module=importlib.import_module('attacks.align_hateful_zero_queue_20261029.'+('faba' if attack=='attack_test2' else attack))
 args=['prepare','--output_dir',str(out),'--model_path',str(MODEL),'--stage',stage]
 if attack=='attack_test2':args+=['--missing_modailty_rate',str(q),'--bd_malicious_ratio',str(mr)]
 else:args+=['--q',str(q),'--malicious_ratio',str(mr)]
 sys.argv=args
 if attack=='attack_test2':
  a=module.arguments();{'raw':module.generate_raw,'extract':module.extract}[stage](a)
 else:module.main()
def main():
 p=argparse.ArgumentParser();p.add_argument('--attack',choices=ORDER,required=True);p.add_argument('--q',type=float,required=True);p.add_argument('--stage',choices=['raw','extract','verify'],required=True);p.add_argument('--output_dir');p.add_argument('--mr',type=float,default=.1);c=p.parse_args();attack=c.attack;q=c.q;out=Path(c.output_dir or output(attack));b=root(attack,q,out,c.mr);a=args_for(attack,q,out,c.mr,c.stage)
 assert os.environ.get('PYTHONHASHSEED')=='2024';random.seed(2024);np.random.seed(2024);torch.manual_seed(2024);torch.cuda.manual_seed_all(2024);torch.set_num_threads(4)
 if c.stage=='verify':verify_features(attack,q,out,c.mr);return
 if c.stage=='raw':
  assert not (b/'metadata.json').exists(),'Do not regenerate raw after extraction'
  if attack in ['titim','toxictextclip','attack_test2']:run_native(attack,q,out,c.mr,'raw')
  else:
   if attack in ['badcm','badcm_3p']:from attacks.badcm_3p.prepare_features import build_generator
   elif attack=='bad_mfl':from attacks.bad_mfl.clip_generator import build_generator
   else:from attacks.pmcmfl_badmfl_generator_20261027 import build_generator
   def factory(args,partition):
    generate,notes=build_generator(args,partition)
    if attack.startswith('bad_mfl'):
     helpers=[cell.cell_contents.clip_helper for cell in (generate.__closure__ or []) if hasattr(cell.cell_contents,'clip_helper')];assert helpers and all(not h.model.training and not any(p.requires_grad for p in h.model.parameters()) and str(h.device)=='cuda' for h in helpers)
    return generate,notes
   meta=base.generate_raw(a,generator_factory=factory)
   if attack.startswith('bad_mfl'):
    checkpoints=sorted((Path(meta['shared_cache_root'])/'generator_ckpt').glob('*.pt'));assert len(checkpoints)==4
    for cp in checkpoints:
     w=torch.load(cp,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for k in ['generator','discriminator'] for v in w[k].values())
    meta.update(generator_checkpoint_sha256={str(p):sha(p) for p in checkpoints},raw_surrogate='official frozen CLIP required by native BadMFL',raw_surrogate_revision=CLIP.name)
   dump(b/'metadata.raw.json',meta)
  meta=read(b/'metadata.raw.json');meta.update(status='raw_complete',victim_architecture='align',victim_feature_shape=[1,640],victim_model_revision=MODEL.name,partition_sha256=sha(out/'partition/hateful_memes/partition_alpha50.json'),simulation_sha256=sha(out/f'simulation_feature/hateful_memes/mm{tag(q)}_alpha50.json'));dump(b/'metadata.raw.json',meta);verify_raw(attack,q,out,c.mr);return
 verify_raw(attack,q,out,c.mr)
 if attack in ['titim','toxictextclip','attack_test2']:run_native(attack,q,out,c.mr,'extract')
 else:
  from model.dual_encoder import FrozenDualEncoder
  model=FrozenDualEncoder('align',str(MODEL),'cuda:0');counts={'truncated':0}
  class Factory:
   def __init__(self,args):self.feature_type=args.feature_type
   def extract_img_features(self,path):
    with Image.open(path) as im:return model.encode_images([im.convert('RGB')]).cpu().numpy().astype(np.float32)
   def extract_text_feature(self,text):
    counts['truncated']+=int(len(model.processor.tokenizer(text,truncation=False)['input_ids'])>64);return model.encode_texts([text]).cpu().numpy().astype(np.float32)
  base.extract_features(a,factory=Factory)
  meta=read(b/'metadata.json');meta['texts_truncated_to_64']=counts['truncated'];dump(b/'metadata.json',meta)
 meta=read(b/'metadata.json');meta.update(victim_architecture='align',feature_shape=[1,640],align_revision=MODEL.name,align_weights_sha256=WEIGHT,encoders_frozen=True,text_max_length=64);dump(b/'metadata.json',meta);verify_features(attack,q,out,c.mr)
def verify_features(attack,q,out=None,mr=.1,raw=True):
 b=root(attack,q,out,mr);meta=read(b/'metadata.json');assert meta['status']=='complete' and not meta['raw_only'] and meta['encoders_frozen'] and meta['align_revision']==MODEL.name and meta['align_weights_sha256']==WEIGHT and meta['text_max_length']==64 and meta['feature_shape']==[1,640]
 assert meta['extraction_source'].lower().startswith('verified_saved_')
 for rel,h in meta['feature_artifact_sha256'].items():
  p=b/rel;assert sha(p)==h;value=pickle.load(p.open('rb'));vals=[v['feature'] for v in value.values()] if isinstance(value,dict) else [x[-1] for x in value]
  for x in vals:
   if x is not None:assert x.shape==(1,640) and x.dtype==np.float32 and np.isfinite(x).all() and np.isclose(np.linalg.norm(x),1,atol=1e-5)
  if isinstance(value,dict):assert set(value)<=set(meta['poisoned_keys'][p.stem]) and all(x['label']==0 for x in value.values())
  elif rel.startswith('attack_'):assert [str(x[0]) for x in value]==meta['attack_test_keys'] and all(x[2]!=0 for x in value)
 return dict(status='verified',attack=attack,q=q,feature_files=len(meta['feature_artifact_sha256']),feature_sha256={str((b/r).relative_to(R)):h for r,h in meta['feature_artifact_sha256'].items()},metadata_sha256=sha(b/'metadata.json'),raw=verify_raw(attack,q,out,mr) if raw else 'precleanup verified')
if __name__=='__main__':logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');main()
