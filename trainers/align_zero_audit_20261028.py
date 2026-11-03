"""Observation only: native clean FedAvg, actual zero-fill inputs, selected checkpoints."""
import datetime,hashlib,json
from pathlib import Path
import torch

def fingerprint(state):
 h=hashlib.sha256()
 for k in sorted(state):h.update(k.encode());h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def install():
 from trainers.server_trainer import Server
 from trainers.fed_avg_trainer import ClientFedAvg
 average=Server.average_weights;update=ClientFedAvg.update_weights
 def checked_average(self,*a,**kw):
  assert self.args.missing_strategy=='zero' and not self.args.backdoor_attack and not self.args.eval_attack
  expected={k:sum(w/float(sum(self.num_samples_list))*u[k].detach().double() for u,w in zip(self.model_updates,self.num_samples_list)) for k in self.model_updates[0]}
  result=average(self,*a,**kw)
  for k,v in self.global_model.state_dict().items():assert torch.allclose(v.double(),expected[k],rtol=1e-5,atol=1e-6),k
  self._align_clean_aggregation=dict(actual_native_call=True,numerical_sample_weighted_verified=True,algorithm='FedAvg',client_ids=list(map(str,self.client_update_ids)),weights=list(map(float,self.num_samples_list)),total_samples=sum(self.num_samples_list))
  return result
 def checked_update(self,*a,**kw):
  counts=dict(paired=0,image_missing=0,text_missing=0,fully_missing=0)
  def check(module,inputs):
   x,y,li,lt=inputs[:4];assert x.shape[-1]==y.shape[-1]==640
   mi=li==0;mt=lt==0
   if mi.any():assert (x[mi]==0).all()
   if mt.any():assert (y[mt]==0).all()
   counts['paired']+=int((~mi&~mt).sum());counts['image_missing']+=int(mi.sum());counts['text_missing']+=int(mt.sum());counts['fully_missing']+=int((mi&mt).sum())
  h=self.model.register_forward_pre_hook(check)
  try:result=update(self,*a,**kw)
  finally:h.remove()
  self.result.update({'align_zero_'+k:v for k,v in counts.items()});return result
 Server.average_weights=checked_average;ClientFedAvg.update_weights=checked_update

def validate_cache(args):
 assert args.img_feat=='align_base_img' and args.text_feat=='align_base_text' and args.missing_strategy=='zero'
 assert not args.backdoor_attack and not args.eval_attack and not args.att
 from features.feature_processing.dual_encoder_features import verify
 verify(args.data_dir,args.dataset,args.alpha,'align')

def append_round(args,server,skipped,device,fold,epoch,result_dir):
 metrics=server.result_dict.get(epoch,{})
 row=dict(architecture='align',dataset=args.dataset,feature_dimensions=[640,640],client_trainer='ClientFedAvg',fold=fold,seed=8*fold,round=epoch,q=args.missing_modailty_rate,device=str(device),missing_strategy='zero',backdoor_attack=False,eval_attack=False,fusion='no_att',dropout=args.dropout_p,selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),training_samples=int(sum(server.num_samples_list)),empty_update=not bool(server.num_samples_list),evaluated=isinstance(metrics.get('dev'),dict) and bool(metrics['dev']),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
 row['aggregation']=getattr(server,'_align_clean_aggregation',None) if not row['empty_update'] else None
 row['zero_fill_checks']={k:sum(v.get('align_zero_'+k,0) for v in metrics.get('train',[])) for k in ['paired','image_missing','text_missing','fully_missing']}
 for split in ['dev','test']:
  if isinstance(metrics.get(split),dict) and metrics[split]:row[split]={k:float(v) for k,v in metrics[split].items() if k in ['loss','f1','acc','uar','sample','pred_majority_ratio']}
 state=server.global_model.state_dict();row['parameters_finite']=all(torch.isfinite(v).all().item() for v in state.values());assert row['parameters_finite']
 row['model_fingerprint']=fingerprint(state)
 if row['evaluated']:row['best_epoch']=int(server.best_epoch)
 verify_recovery_prefix(args,row)
 if row['evaluated'] and server.best_epoch==epoch:torch.save({k:v.detach().cpu() for k,v in state.items()},server.result_path/'model.pt')
 path=Path(result_dir)/f'align_rounds_fold{fold}.jsonl'
 with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 path=Path(args.data_dir)/'align_training_progress.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(path)


def verify_recovery_prefix(args,row):
 p=Path('/hy-tmp/fed/logs/crisismmd_align_zero_20261028/recovery_plan.json')
 if not p.exists() or row['q']!=.9 or args.num_epochs!=200:return
 plan=json.loads(p.read_text());d=Path(plan['repair_dir']);receipt=d/'prefix_verified.json'
 assert str(Path(args.data_dir).resolve())==str(Path(plan['output_dir']).resolve())
 if row['fold']!=1:
  assert receipt.exists() and json.loads(receipt.read_text())['status']=='verified';return
 prefix=Path(plan['original_prefix']);assert hashlib.sha256(prefix.read_bytes()).hexdigest()==plan['prefix_sha256']
 if row['round']<plan['prefix_rounds']:
  expected=[json.loads(x) for x in prefix.read_text().splitlines()][row['round']]
  a={k:v for k,v in row.items() if k!='timestamp'};b={k:v for k,v in expected.items() if k!='timestamp'}
  assert a==b,'Deterministic prefix differs: stop recovery without changing original results'
  state=json.loads(receipt.read_text()) if receipt.exists() else dict(verified_rounds=[])
  state['verified_rounds'].append(row['round']);state.update(status='verified' if state['verified_rounds']==list(range(plan['prefix_rounds'])) else 'checking',original_prefix_sha256=plan['prefix_sha256'],model_fingerprint=row['model_fingerprint'],method='Exact row and model fingerprint equality on deterministic isolated replay; same seed and unchanged native training; originals retained')
  receipt.write_text(json.dumps(state,indent=2));print('RECOVERY_PREFIX_MATCH',row['round'],row['model_fingerprint'],flush=True)
 else:assert receipt.exists() and json.loads(receipt.read_text())['status']=='verified'
