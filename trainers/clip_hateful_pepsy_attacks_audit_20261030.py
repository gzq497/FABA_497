"""Observational native FedAvg and PMCMFL prototype audit; no algorithm changes."""
import datetime,hashlib,json,math
from pathlib import Path
import torch
from trainers.server_trainer import Server
from attacks.clip_hateful_pepsy_queue_20261030.common import SPECS,root,MODEL,WEIGHT,read
from dataloader.dataload_manager import DataloadManager as NativeDataloadManager
from attacks.three_path import load_sparse_features
import pickle

def enabled(a):return a.dataset=='hateful_memes' and a.img_feat=='clip_vit_b32_img' and a.text_feat=='clip_vit_b32_text' and a.backdoor_attack and a.attack_type in SPECS and a.missing_strategy=='pepsy'

def fingerprint(protos):
 h=hashlib.sha256()
 for k in sorted(protos):h.update(k.encode());h.update(protos[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
PEPSY_CONFIG=dict(pool_size=10,top_k=3,contrastive_weight=.2,prompt_reg_weight=.1,temperature=.2,prompt_dropout=0.,num_fedavg_rounds=0)

def config(args):
 v={k:getattr(args,'pepsy_'+k) for k in PEPSY_CONFIG};assert v==PEPSY_CONFIG,v;return v

def modality(key):
 if key.startswith(('base.img_proj.','sim_projectors.0.')):return 'img'
 if key.startswith(('base.text_rnn.','sim_projectors.1.')):return 'text'
 return 'shared'

class AuditedServer(Server):
 def initialize_epoch_updates(self,epoch):
  super().initialize_epoch_updates(epoch);self.model_aggregation_audit=None;self.prototype_aggregation_audit=None
 def average_weights(self):
  assert enabled(self.args) and self._pepsy_enabled() and not self._pmcmfl_enabled() and type(self.global_model).__name__=='PepsyImageTextClassifier'
  super().average_weights()
  assert self.model_aggregation_audit and self.model_aggregation_audit['all_parameters_numerically_verified']
 def average_weights_pepsy(self):
  assert enabled(self.args) and self._pepsy_enabled() and type(self.global_model).__name__=='PepsyImageTextClassifier';config(self.args)
  samples=list(map(float,self.num_samples_list));stats=self.result_dict[self.epoch]['train'];assert samples and len(samples)==len(stats)==len(self.model_updates)
  weights={'shared':samples}
  for mod in ['img','text']:
   flag='pepsy_has_'+mod;ratio='pepsy_present_'+mod
   weights[mod]=[n*max(float(z.get(flag,z.get(ratio,1.))),0.) for n,z in zip(samples,stats)]
   assert all(0<=w<=n for w,n in zip(weights[mod],samples))
  before={k:v.detach().clone() for k,v in self.global_model.state_dict().items()}
  super().average_weights_pepsy()
  current=self.global_model.state_dict();counts={m:0 for m in weights};retained=[];deltas={m:0. for m in ['pool','sim_projectors','global_fuser','specific_embedding']}
  for key,value in current.items():
   if not value.is_floating_point():continue
   group=modality(key);ws=weights[group];total=sum(ws)
   if total<=0:
    assert group!='shared';expected=before[key];retained.append(group)
   else:
    expected=self.model_updates[0][key]*(ws[0]/total)
    for i in range(1,len(ws)):expected=expected+self.model_updates[i][key]*(ws[i]/total)
   assert torch.isfinite(value).all() and torch.allclose(value,expected,atol=1e-7,rtol=1e-6),('Native PEPSY numerical aggregation mismatch',key)
   counts[group]+=1
   for prefix in deltas:
    if key==prefix or key.startswith(prefix+'.'):deltas[prefix]+=float((value-before[key]).float().square().sum().cpu())
  self.model_aggregation_audit=dict(method='average_weights_pepsy',algorithm='PEPSY modality-aware FedAvg',actual_native_call=True,epoch=self.epoch,client_ids=list(map(str,self.client_update_ids)),weights=samples,modality_weights=weights,modality_totals={k:sum(v) for k,v in weights.items()},retained_modalities=sorted(set(retained)),numerically_checked_parameter_tensors=counts,all_parameters_numerically_verified=True,total_samples=sum(samples),num_updates=len(samples),model_parameters_finite=True,pepsy_parameter_delta_l2={k:v**.5 for k,v in deltas.items()},prompt_checklist_reset=bool((self.global_model.trained_prompts_checklist==0).all()))
  assert self.model_aggregation_audit['prompt_checklist_reset']
 def aggregate_pmcmfl_prototypes(self):
  raise AssertionError('PEPSY must not aggregate PMCMFL prototypes')

def validate_pepsy_round(x):
 assert x['missing_strategy']=='pepsy' and x['model_class']=='PepsyImageTextClassifier' and x['pepsy_config']==PEPSY_CONFIG
 assert x['prototype_aggregation'] is None and x['pmcmfl_config'] is None and not x['prototype_imputation'] and not x['prototype_contrast'] and not any(x['pmcmfl_client_statistics'])
 if not x['evaluated']:assert not x['update_clients'] and x['aggregation'] is None;return
 a=x['aggregation'];assert a['method']=='average_weights_pepsy' and a['algorithm']=='PEPSY modality-aware FedAvg' and a['actual_native_call'] and a['all_parameters_numerically_verified'] and a['model_parameters_finite'] and a['prompt_checklist_reset']
 assert a['client_ids']==x['update_clients'] and a['num_updates']==len(x['pepsy_client_statistics'])==len(a['weights']) and a['total_samples']==sum(a['weights'])==x['training_samples']
 assert a['modality_weights']['shared']==a['weights'] and all(n>0 for n in a['numerically_checked_parameter_tensors'].values())
 for mod in ['img','text']:
  expected=[n*max(float(z.get('pepsy_has_'+mod,z.get('pepsy_present_'+mod,1.))),0.) for n,z in zip(a['weights'],x['pepsy_client_statistics'])]
  assert a['modality_weights'][mod]==expected and a['modality_totals'][mod]==sum(expected)
  assert (mod in a['retained_modalities'])==(sum(expected)==0)
 assert x['pepsy_client_statistics'] and all('pepsy_present_img' in z and 'pepsy_present_text' in z for z in x['pepsy_client_statistics'])
 if x['attack_type']=='attack_test2':
  for z in x['attack_client_statistics']:
   if z:assert z.get('attack_test2_pepsy_completion_path')==1.


class DataloadManager(NativeDataloadManager):
 def _overlay(self,cid,mod):
  if not self.is_malicious_client(cid):return None
  p=self._get_attack_root()/('poisoned_'+mod)/(str(cid)+'.pkl')
  if p.exists():
   with p.open('rb') as f:v=pickle.load(f)
   if isinstance(v,dict):return load_sparse_features(self,cid,mod)
  return None
 def load_img_feat(self,client_id,fold_idx=1):
  v=self._overlay(client_id,'img');return v if v is not None else super().load_img_feat(client_id,fold_idx)
 def load_text_feat(self,client_id,fold_idx=1):
  v=self._overlay(client_id,'text');return v if v is not None else super().load_text_feat(client_id,fold_idx)

def validate(args):
 assert enabled(args) and torch.cuda.is_available()
 b=root(args.attack_type,args.missing_modailty_rate,args.data_dir,args.bd_malicious_ratio);m=read(b/'metadata.json')
 assert m['status']=='complete' and not m['raw_only'] and m['encoders_frozen'] and m['clip_weights_sha256']==WEIGHT and m['clip_revision']==MODEL.name and m['feature_shape']==[1,512] and m['text_max_length']==77
 if args.attack_type=='attack_test2':assert not args.attack_test2_raw_prepare and args.attack_test2_use_raw_trigger_features

def metric_values(attack,test,a):
 v=dict(CA=float(test['acc']),clean_f1=float(test['f1']))
 if attack=='titim':v.update(ASR=100*a['ASR_MM'],IA=100*a['ASR_I'],TA='not_applicable_image_only_attack',MBR='not_defined_for_titim')
 elif attack=='toxictextclip':v.update(ASR=100*a['ASR'],TA=100*a['TA'],IA='not_applicable_text_only_attack',MBR='not_defined_for_toxictextclip')
 else:v.update(ASR=100*a['ASR_Joint'],IA=100*a['ASR_I'],TA=100*a['ASR_T'],MBR=a['MBR'])
 return v

def append_round(args,server,skipped,device,fold,epoch,result_dir,evaluated):
 assert enabled(args) and str(device)=='cuda';sp=SPECS[args.attack_type];train=server.result_dict[epoch].get('train',[])
 if evaluated:assert server.model_aggregation_audit and not server._pmcmfl_enabled() and server.prototype_aggregation_audit is None
 else:assert not server.num_samples_list
 def scalar(split):
  v=server.result_dict[epoch].get(split,{})
  if not isinstance(v,dict):return {}
  result={k:float(x) for k,x in v.items() if isinstance(x,(int,float)) and math.isfinite(x)}
  if split=='test' and v.get('conf') is not None:
   import numpy as np
   cache=getattr(server,'_align_test_label_counts',None)
   if cache is None:
    part=read(Path(args.data_dir)/'partition/hateful_memes/partition_alpha50.json')
    cache=np.array([sum(r[2]==label for r in part['test']) for label in range(2)])
    server._align_test_label_counts=cache
   cm=np.asarray(v['conf'],dtype=float)
   assert cm.shape==(2,2) and np.isfinite(cm).all()
   result['confusion_pct']=cm.tolist()
   result['pred_majority_ratio']=float((cache@cm).max()/cache.sum())
   result['pred_majority_source']='native normalized confusion with original test label counts; rounded at 2 decimals'
   result['single_class_collapse']=bool(np.any(np.all(cm==100.,axis=0)))
  return result
 row=dict(pepsy_config=config(args),pepsy_client_statistics=[{k:(v if isinstance(v,list) else float(v)) for k,v in z.items() if k.startswith('pepsy_')} for z in train],architecture='clip',fold=fold,seed=8*fold,round=epoch,q=args.missing_modailty_rate,missing_strategy='pepsy',attack_type=args.attack_type,client_trainer=sp['client'],model_class=type(server.global_model).__name__,device=str(device),feature_dimensions=[512,512],saved_raw_features=True,fusion=args.att_name if args.att else 'no_att',dropout=float(args.dropout_p),selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),training_samples=int(sum(server.num_samples_list)),evaluated=evaluated,aggregation=server.model_aggregation_audit,prototype_aggregation=server.prototype_aggregation_audit,attack_client_statistics=[{k:float(v) for k,v in x.items() if k.startswith(sp['stat_prefix'])} for x in train],pmcmfl_client_statistics=[{k:float(v) for k,v in x.items() if k.startswith('pmcmfl_')} for x in train],pmcmfl_config=None,prototype_imputation=False,prototype_contrast=False,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_fingerprint=fingerprint(server.global_model.state_dict()))
 row.update(attention=bool(args.att),heads=int(args.d_head))
 if args.attack_type.endswith('_3p'):row['three_path']=server.result_dict[epoch]['three_path']
 if evaluated:
  row.update(dev=scalar('dev'),test=scalar('test'),attack=scalar('attack_test'),attack_test_mm=scalar('attack_test_mm'),attack_test_img_only=scalar('attack_test_img_only'),attack_test_text_only=scalar('attack_test_text_only'),best_epoch=int(server.best_epoch));row['metrics_pct']=metric_values(args.attack_type,row['test'],row['attack'])
  if server.best_epoch==epoch:torch.save(server.global_model.state_dict(),server.result_path/'model.pt')
 validate_pepsy_round(row)
 p=Path(result_dir)/f'clip_pepsy_rounds_fold{fold}.jsonl'
 with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 progress={k:v for k,v in row.items() if k not in ['dev','test','attack','attack_test_mm','attack_test_img_only','attack_test_text_only','attack_client_statistics','pmcmfl_client_statistics']};p=Path(args.data_dir)/'training_progress.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(progress,indent=2));t.replace(p)

def finalize_fold(args,result,server):result.update(best_epoch=int(server.best_epoch),**metric_values(args.attack_type,server.best_test_dict,server.best_attack_result_dict['attack_test']))
