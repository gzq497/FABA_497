"""Observational native FedAvg and PMCMFL prototype audit; no algorithm changes."""
import datetime,hashlib,json,math
from pathlib import Path
import torch
from trainers.server_trainer import Server

def enabled(a):return a.dataset=='crisis-mmd' and a.img_feat=='clip_vit_b32_img' and a.text_feat=='clip_vit_b32_text' and a.backdoor_attack and a.attack_type=='titim' and a.missing_strategy=='pmcmfl'
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

def append_round(args,server,skipped,device,fold,epoch,result_dir,evaluated):
 assert enabled(args) and str(device)=='cuda' and type(server.global_model).__name__=='ImageTextClassifier'
 if evaluated:assert server.model_aggregation_audit and server.prototype_aggregation_audit
 else:assert not server.num_samples_list
 def scalar(split):return {k:float(v) for k,v in server.result_dict[epoch][split].items() if isinstance(v,(int,float)) and math.isfinite(v)}
 train=server.result_dict[epoch].get('train',[])
 row=dict(fold=fold,seed=8*fold,round=epoch,q=args.missing_modailty_rate,missing_strategy='pmcmfl',attack_type='titim',client_trainer='ClientTITIM',model_class=type(server.global_model).__name__,device=str(device),feature_dimensions=[512,512],saved_raw_features=True,fusion=args.att_name if args.att else 'no_att',dropout=float(args.dropout_p),selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),training_samples=int(sum(server.num_samples_list)),evaluated=evaluated,aggregation=server.model_aggregation_audit,prototype_aggregation=server.prototype_aggregation_audit,titim_client_statistics=[{k:float(v) for k,v in x.items() if k.startswith('titim_')} for x in train if any(k.startswith('titim_') for k in x)],pmcmfl_client_statistics=[{k:float(v) for k,v in x.items() if k.startswith('pmcmfl_')} for x in train],pmcmfl_config=config(args),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),TA='not_applicable_image_only_attack',MBR='not_defined_for_titim')
 if evaluated:
  row.update(dev=scalar('dev'),test=scalar('test'),attack=scalar('attack_test'),attack_test_mm=scalar('attack_test_mm'),attack_test_img_only=scalar('attack_test_img_only'),best_epoch=int(server.best_epoch));row['metrics_pct']=dict(CA=row['test']['acc'],ASR=100*row['attack']['ASR_MM'],IA=100*row['attack']['ASR_I'],clean_MacroF1=row['test']['f1'])
  if server.best_epoch==epoch:
   torch.save(server.global_model.state_dict(),server.result_path/'model.pt');torch.save(server.get_pmcmfl_prototypes(),server.result_path/'pmcmfl_prototypes.pt')
 with (Path(result_dir)/f'pmcmfl_titim_rounds_fold{fold}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 progress={k:v for k,v in row.items() if k not in ['dev','test','attack','attack_test_mm','attack_test_img_only','titim_client_statistics','pmcmfl_client_statistics']};p=Path(args.data_dir)/'training_progress.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(progress,indent=2));t.replace(p)
 print('[clip-pmcmfl-titim round]',json.dumps(progress),flush=True)
def finalize_fold(args,result,server):
 assert enabled(args);a=server.best_attack_result_dict['attack_test'];result.update(best_epoch=int(server.best_epoch),CA=float(server.best_test_dict['acc']),ASR=100*float(a['ASR_MM']),IA=100*float(a['ASR_I']),TA='not_applicable_image_only_attack',MBR='not_defined_for_titim')
