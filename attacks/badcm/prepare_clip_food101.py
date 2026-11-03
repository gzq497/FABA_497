"""Saved PNG/TXT then frozen CLIP features, using existing BadCM raw poisoning functions."""
import argparse,collections,copy,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from attacks.attack_test2.prepare_raw_trigger import resolve_img_path
from attacks.badcm.prepare_features import (choose_q_dependent_malicious_clients,summarize_client_modality,sample_available_from_sim,load_pattern,poison_raw_image,poison_raw_text,save_raw_pair,poison_count,load_pickle,save_pickle,dump_jsonl)
from features.feature_processing.clip_crisis_features import sha256,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--output_dir',required=True);p.add_argument('--model_path',required=True);p.add_argument('--q',required=True,type=float);p.add_argument('--stage',choices=['raw','extract'],required=True)
 p.add_argument('--malicious_ratio',type=float,default=0.1);p.add_argument('--poison_ratio',type=float,default=0.1);p.add_argument('--target',type=int,default=0);p.add_argument('--seed',type=int,default=2024)
 p.add_argument('--pixel_eps',type=float,default=4.0/255.0);p.add_argument('--text_trigger',default='cf');p.add_argument('--pattern_path',default='/hy-tmp/fed/attacks/badcm/assets/pattern.jpg')
 a=p.parse_args();out=Path(a.output_dir)
 root=out/f'attacks/badcm/upmc_food101/alpha50/mm{str(a.q).replace(".","")}/mr{str(a.malicious_ratio).replace(".","")}_pr{str(a.poison_ratio).replace(".","")}_tl{a.target}_seed{a.seed}'
 part_path=out/'partition/upmc_food101/partition_alpha50.json';sim_path=out/f'simulation_feature/upmc_food101/mm{str(a.q).replace(".","")}_alpha50.json'
 part=json.loads(part_path.read_text());sim=json.loads(sim_path.read_text())
 imgdir=out/'feature/img/clip_vit_b32_img/upmc_food101/alpha50';textdir=out/'feature/text/clip_vit_b32_text/upmc_food101/alpha50'
 if a.stage=='raw':
  assert not root.exists();root.mkdir(parents=True)
  clients,malicious=choose_q_dependent_malicious_clients(imgdir,a.malicious_ratio,a.seed,sim);assert malicious
  available=[cid for cid in clients if summarize_client_modality(sim[cid])['has_any']]
  pattern=load_pattern(Path(a.pattern_path));rng=random.Random(a.seed)
  meta=dict(attack_type='badcm',source='existing project raw-sample-first BadCM adaptation',adaptation_note='Existing bounded visual-pattern pixel perturbation and direct text substitution; no pretrained upstream visual generator is available. Original ClientBadCM weighted supervised loss, not three-path training.',dataset='upmc_food101',missing_rate=a.q,alpha=5.0,malicious_ratio=a.malicious_ratio,poison_ratio=a.poison_ratio,target_label=a.target,seed=a.seed,q_dependent_prepare=True,client_count=len(clients),malicious_clients=malicious,malicious_selection_pool_size=len(available),malicious_selection_excludes_all_missing=True,malicious_client_modalities={},poisoned_keys={},poisoned_modes={},attack_test_keys=[],attack_test_modes={},skipped_all_missing_clients=[],badcm_pixel_eps=a.pixel_eps,badcm_text_trigger=a.text_trigger,visual_pattern=a.pattern_path,pattern_sha256=sha256(Path(a.pattern_path)),partition_sha256=sha256(part_path),simulation_sha256=sha256(sim_path),status='raw_complete',raw_only=True,pipeline_order=['all five saved PNG/TXT raw','all five reload saved PNG/TXT frozen CLIP extraction','verify SHA256','smoke','train'])
  def make(record,split,cid,index,hi,ht):
   key,source,label,text=record
   source=resolve_img_path(source,"/hy-tmp/fed/data","upmc_food101")
   with Image.open(source) as image:original=image.convert('RGB')
   poisoned=poison_raw_image(original,pattern,a.pixel_eps) if hi else original.copy()
   value,trace=poison_raw_text(text,a.text_trigger,str(key))
   if not ht:value=text;trace={'strategy':'not_applied_missing_modality'}
   row=save_raw_pair(root/'raw_samples',split,cid,str(key),Path(source),text,poisoned,value,label,a.target,hi,ht,trace)
   delta=int(np.abs(np.asarray(poisoned,dtype=np.int16)-np.asarray(original,dtype=np.int16)).max())
   assert delta<=4 if hi else delta==0
   row.update(record_index=index,original_img_sha256=sha256(Path(source)),poisoned_img_sha256=sha256(Path(row['poisoned_img_path'])),poisoned_text_sha256=sha256(Path(row['poisoned_text_path'])),max_pixel_delta_uint8=delta)
   if (len(train)+len(test)+1)%100==0:
    import shutil
    assert shutil.disk_usage(out).free>2*1024**3, 'Data disk below 2 GiB reserve'
    print('RAW_PROGRESS',split,len(train)+len(test)+1,'q',a.q,flush=True)
   return row
  train=[];test=[]
  for cid in malicious:
   modality=summarize_client_modality(sim[cid]);assert modality['has_any'];meta['malicious_client_modalities'][cid]=modality
   rows=part[cid];eligible=[i for i,v in enumerate(rows) if v[2]!=a.target and any(sample_available_from_sim(sim[cid],i))]
   indices=rng.sample(eligible,poison_count(len(eligible),a.poison_ratio));meta['poisoned_keys'][cid]=[];meta['poisoned_modes'][cid]={}
   for i in indices:
    hi,ht=sample_available_from_sim(sim[cid],i);row=make(rows[i],'train',cid,i,hi,ht);train.append(row);key=row['sample_key']
    meta['poisoned_keys'][cid].append(key);meta['poisoned_modes'][cid][key]=dict(mode='raw_dual' if hi and ht else 'raw_single',present_img=hi,present_text=ht,raw_img_path=row['poisoned_img_path'],raw_text_path=row['poisoned_text_path'])
  eligible=[i for i,v in enumerate(part['test']) if v[2]!=a.target]
  count=min(poison_count(len(part['test']),a.poison_ratio),len(eligible))
  for i in rng.sample(eligible,count):
   row=make(part['test'][i],'attack_test',None,i,True,True);test.append(row);key=row['sample_key'];meta['attack_test_keys'].append(key)
   meta['attack_test_modes'][key]=dict(mode='raw_dual',raw_img_path=row['poisoned_img_path'],raw_text_path=row['poisoned_text_path'])
  assert train and test
  for field,rows,name in [('raw_train_manifest',train,'train_manifest.jsonl'),('raw_attack_manifest',test,'attack_test_manifest.jsonl')]:
   path=root/'raw_samples'/name;dump_jsonl(path,rows);meta[field]=str(path);meta[field+'_sha256']=sha256(path)
  meta.update(raw_train_count=len(train),raw_attack_test_count=len(test),attack_eval_count=len(test),malicious_client_modality_summary=dict(collections.Counter(v['type'] for v in meta['malicious_client_modalities'].values())))
  atomic_json(root/'metadata.raw.json',meta);print(json.dumps(dict(stage='raw',q=a.q,status='complete',train=len(train),attack_test=len(test),malicious_clients=len(malicious))),flush=True);return
 import torch
 from transformers import CLIPModel,CLIPProcessor
 assert torch.cuda.is_available() and not (root/'metadata.json').exists()
 meta=json.loads((root/'metadata.raw.json').read_text());assert sha256(part_path)==meta['partition_sha256'] and sha256(sim_path)==meta['simulation_sha256'] and sha256(Path(meta['visual_pattern']))==meta['pattern_sha256']
 allrows=[];splitrows={}
 for field in ['raw_train_manifest','raw_attack_manifest']:
  path=Path(meta[field]);assert sha256(path)==meta[field+'_sha256'];rows=[json.loads(s) for s in path.read_text().splitlines()]
  for row in rows:
   for pf,hf in [('original_img_path','original_img_sha256'),('poisoned_img_path','poisoned_img_sha256'),('poisoned_text_path','poisoned_text_sha256')]:assert sha256(Path(row[pf]))==row[hf]
  splitrows[field]=rows;allrows.extend(rows)
 torch.set_num_threads(4);model=CLIPModel.from_pretrained(a.model_path,local_files_only=True).eval().cuda();model.requires_grad_(False)
 processor=CLIPProcessor.from_pretrained(a.model_path,local_files_only=True);imgfeatures={};textfeatures={};truncated=0
 with torch.inference_mode():
  for modality in ['img','text']:
   rows=[row for row in allrows if row['image_applied' if modality=='img' else 'text_applied']]
   for start in range(0,len(rows),64):
    batch=rows[start:start+64]
    if modality=='img':
     images=[]
     for row in batch:
      with Image.open(row['poisoned_img_path']) as im:images.append(im.convert('RGB'))
     inputs=processor(images=images,return_tensors='pt').to('cuda');values=model.get_image_features(pixel_values=inputs['pixel_values'])
    else:
     texts=[Path(row['poisoned_text_path']).read_text(encoding='utf-8') for row in batch]
     truncated+=sum(len(v)>77 for v in processor.tokenizer(texts,truncation=False)['input_ids'])
     inputs=processor.tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=77).to('cuda');values=model.get_text_features(**inputs)
    values=torch.nn.functional.normalize(values,dim=-1).cpu().numpy().astype(np.float32)
    for row,arr in zip(batch,values):
     assert arr.shape==(512,) and np.isfinite(arr).all()
     (imgfeatures if modality=='img' else textfeatures)[row['poisoned_img_path' if modality=='img' else 'poisoned_text_path']]=arr[None,:].copy()
    print(f'EXTRACT_PROGRESS q={a.q} modality={modality} done={min(start+64,len(rows))}/{len(rows)}',flush=True)
 artifacts=[];train=splitrows['raw_train_manifest'];test=splitrows['raw_attack_manifest']
 for cid in meta['malicious_clients']:
  img=copy.deepcopy(load_pickle(imgdir/f'{cid}.pkl'));text=copy.deepcopy(load_pickle(textdir/f'{cid}.pkl'))
  for row in train:
   if row['client_id']!=cid:continue
   i=row['record_index'];assert str(img[i][0])==str(text[i][0])==row['sample_key']
   if row['image_applied']:img[i][-1]=imgfeatures[row['poisoned_img_path']]
   if row['text_applied']:text[i][-1]=textfeatures[row['poisoned_text_path']]
   img[i][-2]=a.target;text[i][-2]=a.target
  for rel,values in [(f'poisoned_img/{cid}.pkl',img),(f'poisoned_text/{cid}.pkl',text)]:
   path=root/rel;save_pickle(path,values);artifacts.append(path)
 img=load_pickle(imgdir/'test.pkl');text=load_pickle(textdir/'test.pkl');attack_img=[];attack_text=[]
 for row in test:
  i=row['record_index'];assert str(img[i][0])==str(text[i][0])==row['sample_key'];vi=copy.deepcopy(img[i]);vt=copy.deepcopy(text[i])
  vi[-1]=imgfeatures[row['poisoned_img_path']];vt[-1]=textfeatures[row['poisoned_text_path']];attack_img.append(vi);attack_text.append(vt)
 for rel,values in [('attack_img/test.pkl',attack_img),('attack_text/test.pkl',attack_text)]:
  path=root/rel;save_pickle(path,values);artifacts.append(path)
 meta.update(status='complete',raw_only=False,encoders_frozen=True,clip_revision=Path(a.model_path).name,clip_weights_sha256=sha256(Path(a.model_path)/'pytorch_model.bin'),feature_artifact_sha256={str(p.relative_to(root)):sha256(p) for p in artifacts},feature_shape=[1,512],text_max_length=77,texts_truncated_to_77=truncated,extraction_source='verified_saved_PNG_TXT')
 atomic_json(root/'metadata.json',meta);print(json.dumps(dict(stage='extract',q=a.q,status='complete',feature_files=len(artifacts),truncated=truncated)),flush=True)
if __name__=='__main__':main()
