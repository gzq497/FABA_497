"""Observational native FedAvg and PMCMFL prototype audit; no algorithm changes."""
import datetime,hashlib,json,math
from pathlib import Path
import torch
from trainers.server_trainer import Server
from attacks.align_pmcmfl_queue_20261028.common import SPECS,root,MODEL,WEIGHT,read
from dataloader.dataload_manager import DataloadManager as NativeDataloadManager
from attacks.three_path import load_sparse_features
import pickle

def enabled(a):return a.dataset=='crisis-mmd' and a.img_feat=='align_base_img' and a.text_feat=='align_base_text' and a.backdoor_attack and a.attack_type in SPECS and a.missing_strategy=='pmcmfl'

def fingerprint(protos):
 h=hashlib.sha256()
 for k in sorted(protos):h.update(k.encode());h.update(protos[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def config(a):return {k:getattr(a,k) for k in ['pmcmfl_prototype_as_missing_modal','pmcmfl_prototype_as_rep_target','pmcmfl_regularization_start_epoch','pmcmfl_temperature','pmcmfl_img_loss_weight','pmcmfl_text_loss_weight','pmcmfl_fusion_loss_weight','pmcmfl_prototype_init_std']}
class AuditedServer(Server):
 def initialize_epoch_updates(self,epoch):
  super().initialize_epoch_updates(epoch);self.model_aggregation_audit=None;self.prototype_aggregation_audit=None
 def average_weights(self):
  assert enabled(self.args) and type(self.global_model).__name__=='ImageTextClassifier' and not self._pepsy_enabled()
  weights=list(map(float,self.num_samples_list));assert len(weights)>0 and sum(weights)>0
  key=next(k for k,v in self.global_model.state_dict().items() if v.is_floating_point());before=self.global_model.state_dict()[key].detach().clone()
  super().average_weights()
  self.model_aggregation_audit=dict(method='average_weights',algorithm='FedAvg',actual_native_call=True,epoch=self.epoch,client_ids=list(map(str,self.client_update_ids)),weights=weights,total_samples=sum(weights),num_updates=len(weights),model_parameters_finite=all(bool(torch.isfinite(v).all()) for v in self.global_model.state_dict().values()),observed_parameter=key,parameter_delta_l2=float((self.global_model.state_dict()[key]-before).norm().cpu()))
 def aggregate_pmcmfl_prototypes(self):
  assert enabled(self.args) and self._pmcmfl_enabled();before=self.get_pmcmfl_prototypes();expected={k:v.clone() for k,v in before.items()};incoming={};accepted=set(self.last_aggregation_accepted_indices)
  for mod in ['img','text','fusion']:
   sums=torch.zeros_like(before[mod]);counts=torch.zeros_like(before['counts_'+mod])
   for i,u in enumerate(self.prototype_updates):
    if i in accepted and u is not None:sums+=u[mod].cpu()*u['counts_'+mod].cpu().unsqueeze(1);counts+=u['counts_'+mod].cpu()
   valid=counts>0;expected[mod][valid]=sums[valid]/counts[valid].unsqueeze(1);expected['counts_'+mod][valid]=counts[valid];incoming[mod]=counts.tolist()
  super().aggregate_pmcmfl_prototypes();actual=self.get_pmcmfl_prototypes()
  assert all(torch.isfinite(v).all() and torch.allclose(v,expected[k],atol=1e-5,rtol=1e-5) for k,v in actual.items())
  self.prototype_aggregation_audit=dict(method='aggregate_pmcmfl_prototypes',actual_native_call=True,numerical_count_weighted_verified=True,client_ids=list(map(str,self.client_update_ids)),accepted_indices=sorted(accepted),client_prototype_counts=[None if u is None else {m:u['counts_'+m].tolist() for m in ['img','text','fusion']} for u in self.prototype_updates],incoming_counts=incoming,global_counts={m:actual['counts_'+m].tolist() for m in ['img','text','fusion']},global_prototypes_finite=True,prototype_delta_l2=float(sum(float((actual[k]-before[k]).square().sum()) for k in ['img','text','fusion'])**.5),prototype_fingerprint=fingerprint(actual))


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
 assert m['status']=='complete' and not m['raw_only'] and m['encoders_frozen'] and m['align_weights_sha256']==WEIGHT and m['align_revision']==MODEL.name and m['feature_shape']==[1,640] and m['text_max_length']==64
 if args.attack_type=='attack_test2':assert not args.attack_test2_raw_prepare and args.attack_test2_use_raw_trigger_features

def metric_values(attack,test,a):
 v=dict(CA=float(test['acc']),clean_f1=float(test['f1']))
 if attack=='titim':v.update(ASR=100*a['ASR_MM'],IA=100*a['ASR_I'],TA='not_applicable_image_only_attack',MBR='not_defined_for_titim')
 elif attack=='toxictextclip':v.update(ASR=100*a['ASR'],TA=100*a['TA'],IA='not_applicable_text_only_attack',MBR='not_defined_for_toxictextclip')
 else:v.update(ASR=100*a['ASR_Joint'],IA=100*a['ASR_I'],TA=100*a['ASR_T'],MBR=a['MBR'])
 return v

def append_round(args,server,skipped,device,fold,epoch,result_dir,evaluated):
 assert enabled(args) and str(device)=='cuda';sp=SPECS[args.attack_type];train=server.result_dict[epoch].get('train',[])
 if evaluated:assert server.model_aggregation_audit and server.prototype_aggregation_audit
 else:assert not server.num_samples_list
 def scalar(split):
  v=server.result_dict[epoch].get(split,{});return {k:float(x) for k,x in v.items() if isinstance(x,(int,float)) and math.isfinite(x)} if isinstance(v,dict) else {}
 row=dict(architecture='align',fold=fold,seed=8*fold,round=epoch,q=args.missing_modailty_rate,missing_strategy='pmcmfl',attack_type=args.attack_type,client_trainer=sp['client'],model_class=type(server.global_model).__name__,device=str(device),feature_dimensions=[640,640],saved_raw_features=True,fusion=args.att_name if args.att else 'no_att',dropout=float(args.dropout_p),selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),training_samples=int(sum(server.num_samples_list)),evaluated=evaluated,aggregation=server.model_aggregation_audit,prototype_aggregation=server.prototype_aggregation_audit,attack_client_statistics=[{k:float(v) for k,v in x.items() if k.startswith(sp['stat_prefix'])} for x in train],pmcmfl_client_statistics=[{k:float(v) for k,v in x.items() if k.startswith('pmcmfl_')} for x in train],pmcmfl_config=config(args),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_fingerprint=fingerprint(server.global_model.state_dict()))
 row.update(attention=bool(args.att),heads=int(args.d_head))
 if args.attack_type.endswith('_3p'):row['three_path']=server.result_dict[epoch]['three_path']
 if evaluated:
  row.update(dev=scalar('dev'),test=scalar('test'),attack=scalar('attack_test'),attack_test_mm=scalar('attack_test_mm'),attack_test_img_only=scalar('attack_test_img_only'),attack_test_text_only=scalar('attack_test_text_only'),best_epoch=int(server.best_epoch));row['metrics_pct']=metric_values(args.attack_type,row['test'],row['attack'])
  if server.best_epoch==epoch:torch.save(server.global_model.state_dict(),server.result_path/'model.pt');torch.save(server.get_pmcmfl_prototypes(),server.result_path/'pmcmfl_prototypes.pt')
 p=Path(result_dir)/f'align_pmcmfl_rounds_fold{fold}.jsonl'
 with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 progress={k:v for k,v in row.items() if k not in ['dev','test','attack','attack_test_mm','attack_test_img_only','attack_test_text_only','attack_client_statistics','pmcmfl_client_statistics']};p=Path(args.data_dir)/'training_progress.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(progress,indent=2));t.replace(p)

def finalize_fold(args,result,server):result.update(best_epoch=int(server.best_epoch),**metric_values(args.attack_type,server.best_test_dict,server.best_attack_result_dict['attack_test']))
