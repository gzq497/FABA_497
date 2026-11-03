"""Isolated saved-TXT -> frozen CLIP adapter, retaining existing ToxicTextCLIP selection/augmentation."""
import argparse,copy,hashlib,json,pickle,random,shutil
from pathlib import Path
import numpy as np
from attacks.toxictextclip.prepare_features import (choose_text_capable_clients,build_target_pool,choose_poison_indices,pooled_text_feature,augment_text,save_poisoned_text,poison_count,load_pickle,save_pickle,dump_jsonl,summarize_client_modality)
from features.feature_processing.clip_crisis_features import sha256,atomic_json

def save_pickle(path,values):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.pkl.tmp')
 with temp.open('wb') as f:pickle.dump(values,f)
 temp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output_dir',required=True);p.add_argument('--model_path',required=True);p.add_argument('--q',required=True,type=float);p.add_argument('--stage',choices=['raw','extract'],required=True)
 p.add_argument('--malicious_ratio',type=float,default=0.1);p.add_argument('--poison_ratio',type=float,default=0.1);p.add_argument('--target',type=int,default=0);p.add_argument('--seed',type=int,default=2024)
 a=p.parse_args();out=Path(a.output_dir)
 root=out/f'attacks/toxictextclip/hateful_memes/alpha50/mm{str(a.q).replace(".","")}/mr{str(a.malicious_ratio).replace(".","")}_pr{str(a.poison_ratio).replace(".","")}_tl{a.target}_seed{a.seed}'
 part_path=out/'partition/hateful_memes/partition_alpha50.json';sim_path=out/f'simulation_feature/hateful_memes/mm{str(a.q).replace(".","")}_alpha50.json'
 part=json.loads(part_path.read_text());sim=json.loads(sim_path.read_text())
 imgdir=out/'feature/img/clip_vit_b32_img/hateful_memes/alpha50';textdir=out/'feature/text/clip_vit_b32_text/hateful_memes/alpha50'
 if a.stage=='raw':
  assert not (root/'metadata.json').exists(), 'Raw stage cannot overwrite extracted features'
  assert shutil.disk_usage(out).free>=2*1024**3, 'Less than 2GiB free; no extra deletion'
  if (root/'metadata.raw.json').exists():
   m=json.loads((root/'metadata.raw.json').read_text());assert m['missing_rate']==a.q and m['seed']==a.seed and m['malicious_ratio']==a.malicious_ratio and m['poison_ratio']==a.poison_ratio and m['target_label']==a.target and m['partition_sha256']==sha256(part_path) and m['simulation_sha256']==sha256(sim_path)
   for field in ['raw_train_manifest','raw_attack_manifest']:
    assert sha256(Path(m[field]))==m[field+'_sha256']
    for row in map(json.loads,Path(m[field]).read_text().splitlines()):assert sha256(Path(row['poisoned_text_path']))==row['poisoned_text_sha256']
   print('SAVED_RAW_ALREADY_COMPLETE',flush=True);return
  root.mkdir(parents=True,exist_ok=True)
  clients,eligible,malicious=choose_text_capable_clients(textdir,a.malicious_ratio,a.seed,sim);assert malicious
  lookup={str(row[0]):row for values in part.values() for row in values}
  pool,matrix,prototype=build_target_pool(textdir,lookup,a.target,256,a.seed)
  train_keys={str(row[0]) for cid,values in part.items() if cid not in ['dev','test'] for row in values}
  assert all(row['key'] in train_keys for row in pool)
  meta=dict(attack_type='toxictextclip',source='existing project ToxicTextCLIP framework adaptation',source_commit='0febe89a7aa7e388829419d85c8fdfecb153a158',dataset='hateful_memes',text_only=True,image_modified=False,text_modified=True,missing_rate=a.q,q_dependent_prepare=True,malicious_ratio=a.malicious_ratio,poison_ratio=a.poison_ratio,target_label=a.target,seed=a.seed,trigger='Please return high-quality results.',target_pool_size=len(pool),target_pool_keys=[x['key'] for x in pool],target_pool_policy='train_only_v1',context_words=16,client_count=len(clients),text_capable_client_count=len(eligible),malicious_clients=malicious,malicious_client_modalities={},poisoned_keys={},attack_test_keys=[],partition_sha256=sha256(part_path),simulation_sha256=sha256(sim_path),status='raw_complete',raw_only=True,pipeline_order=['isolated CUDA preflight before formal raw','all five saved raw PNG/TXT','all five actual saved-byte frozen ALIGN extraction','all five SHA verification','five serial training'],IA='not_applicable_text_only_attack')
  def make(item,split,cid,index):
   assert shutil.disk_usage(out).free>=2*1024**3, 'Less than 2GiB free'
   key=str(item[0]);source=lookup[key]
   poisoned,trace=augment_text(source[-1],pooled_text_feature(item[-1]),pool,matrix,meta['trigger'],16)
   row=save_poisoned_text(root/'raw_samples',split,cid,key,source[-1],poisoned,int(item[-2]),a.target,trace)
   row.update(record_index=index,poisoned_text_sha256=sha256(Path(row['poisoned_text_path'])),source_image=source[1],source_image_sha256=sha256(Path(source[1])))
   return row
  train=[];test=[]
  for cid in malicious:
   records=load_pickle(textdir/f'{cid}.pkl');indices=choose_poison_indices(records,sim[cid],a.target,a.poison_ratio,prototype)
   meta['poisoned_keys'][cid]=[];meta['malicious_client_modalities'][cid]=summarize_client_modality(sim[cid])
   for index in indices:
    row=make(records[index],'train',cid,index);train.append(row);meta['poisoned_keys'][cid].append(row['sample_key'])
  clean_test=load_pickle(textdir/'test.pkl');available=[i for i,v in enumerate(clean_test) if int(v[-2])!=a.target]
  indices=random.Random(a.seed).sample(available,poison_count(len(available),a.poison_ratio))
  for index in indices:
   row=make(clean_test[index],'attack_test',None,index);test.append(row);meta['attack_test_keys'].append(row['sample_key'])
  assert train and test
  for field,rows,name in [('raw_train_manifest',train,'train_manifest.jsonl'),('raw_attack_manifest',test,'attack_test_manifest.jsonl')]:
   path=root/'raw_samples'/name;dump_jsonl(path,rows);meta[field]=str(path);meta[field+'_sha256']=sha256(path)
  meta.update(raw_train_count=len(train),raw_attack_test_count=len(test))
  atomic_json(root/'metadata.raw.json',meta)
  print(json.dumps(dict(stage='raw',q=a.q,status='complete',train=len(train),attack_test=len(test),malicious_clients=len(malicious))),flush=True);return
 import torch
 from transformers import CLIPModel,CLIPProcessor
 assert torch.cuda.is_available()
 assert Path(a.model_path).name=='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268' and sha256(Path(a.model_path)/'pytorch_model.bin')=='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
 if (root/'metadata.json').exists():
  m=json.loads((root/'metadata.json').read_text());assert m['status']=='complete' and not m['raw_only'] and m['extraction_source']=='verified_saved_TXT'
  for rel,h in m['feature_artifact_sha256'].items():assert sha256(root/rel)==h
  print('EXTRACT_ALREADY_COMPLETE',flush=True);return
 meta=json.loads((root/'metadata.raw.json').read_text());assert sha256(part_path)==meta['partition_sha256'] and sha256(sim_path)==meta['simulation_sha256']
 allrows=[];splitrows={}
 for field in ['raw_train_manifest','raw_attack_manifest']:
  path=Path(meta[field]);assert sha256(path)==meta[field+'_sha256']
  rows=[json.loads(s) for s in path.read_text().splitlines()]
  for row in rows:assert sha256(Path(row['poisoned_text_path']))==row['poisoned_text_sha256'] and sha256(Path(row['source_image']))==row['source_image_sha256']
  splitrows[field]=rows;allrows.extend(rows)
 torch.set_num_threads(4);model=CLIPModel.from_pretrained(a.model_path,local_files_only=True).eval().cuda();model.requires_grad_(False)
 processor=CLIPProcessor.from_pretrained(a.model_path,local_files_only=True);features={};truncated=0
 with torch.inference_mode():
  for start in range(0,len(allrows),128):
   batch=allrows[start:start+128];texts=[Path(row['poisoned_text_path']).read_text(encoding='utf-8') for row in batch]
   full=processor.tokenizer(texts,truncation=False)['input_ids'];truncated+=sum(len(v)>77 for v in full)
   inputs=processor.tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=77).to('cuda')
   values=torch.nn.functional.normalize(model.get_text_features(**inputs),dim=-1).cpu().numpy().astype(np.float32)
   for row,arr in zip(batch,values):assert arr.shape==(512,) and np.isfinite(arr).all();features[row['poisoned_text_path']]=arr[None,:].copy()
   print(f'EXTRACT_PROGRESS q={a.q} done={min(start+128,len(allrows))}/{len(allrows)}',flush=True)
 artifacts=[];train=splitrows['raw_train_manifest'];test=splitrows['raw_attack_manifest']
 for cid in meta['malicious_clients']:
  img=copy.deepcopy(load_pickle(imgdir/f'{cid}.pkl'));text=copy.deepcopy(load_pickle(textdir/f'{cid}.pkl'))
  for row in train:
   if row['client_id']!=cid:continue
   i=row['record_index'];assert str(text[i][0])==str(img[i][0])==row['sample_key']
   text[i][-1]=features[row['poisoned_text_path']];text[i][-2]=a.target;img[i][-2]=a.target
  for rel,values in [(f'poisoned_img/{cid}.pkl',img),(f'poisoned_text/{cid}.pkl',text)]:
   path=root/rel;save_pickle(path,values);artifacts.append(path)
 img=load_pickle(imgdir/'test.pkl');text=load_pickle(textdir/'test.pkl');attack_img=[];attack_text=[]
 for row in test:
  i=row['record_index'];assert str(img[i][0])==str(text[i][0])==row['sample_key']
  value=copy.deepcopy(text[i]);value[-1]=features[row['poisoned_text_path']]
  attack_img.append(copy.deepcopy(img[i]));attack_text.append(value)
 for rel,values in [('attack_img/test.pkl',attack_img),('attack_text/test.pkl',attack_text)]:
  path=root/rel;save_pickle(path,values);artifacts.append(path)
 meta.update(status='complete',raw_only=False,encoders_frozen=True,clip_revision=Path(a.model_path).name,clip_weights_sha256=sha256(Path(a.model_path)/'pytorch_model.bin'),feature_artifact_sha256={str(p.relative_to(root)):sha256(p) for p in artifacts},feature_shape=[1,512],text_max_length=77,augmented_texts_truncated_to_77=truncated,extraction_source='verified_saved_TXT')
 atomic_json(root/'metadata.json',meta);print(json.dumps(dict(stage='extract',q=a.q,status='complete',feature_files=len(artifacts),truncated=truncated)),flush=True)
if __name__=='__main__':main()
