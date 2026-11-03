"""Observational audit only; native TITIM loss, PEPSY model and aggregation unchanged."""
import datetime,json,math
from pathlib import Path
import torch
from trainers.server_trainer import Server

def enabled(args):
 return args.dataset=='crisis-mmd' and args.img_feat=='clip_vit_b32_img' and args.text_feat=='clip_vit_b32_text' and args.backdoor_attack and args.attack_type=='titim' and args.missing_strategy=='pepsy'
class AuditedServer(Server):
 def initialize_epoch_updates(self,epoch):
  super().initialize_epoch_updates(epoch);self.pepsy_aggregation_audit=None
 def average_weights_pepsy(self):
  assert enabled(self.args) and type(self.global_model).__name__=='PepsyImageTextClassifier'
  weights={m:[float(self._pepsy_client_weight(i,None if m=='shared' else m)) for i in range(len(self.model_updates))] for m in ('img','text','shared')}
  before=self.global_model.pool.prompt.detach().clone()
  super().average_weights_pepsy()
  self.pepsy_aggregation_audit=dict(method='average_weights_pepsy',actual_native_call=True,epoch=self.epoch,client_ids=list(map(str,self.client_update_ids)),weights=weights,totals={k:sum(v) for k,v in weights.items()},prompt_parameter_delta_l2=float((self.global_model.pool.prompt.detach()-before).norm().cpu()),prompt_parameters_finite=bool(torch.isfinite(self.global_model.pool.prompt).all()),num_updates=len(self.model_updates))

def append_round(args,server,skipped,device,fold,epoch,result_dir,evaluated):
 assert enabled(args) and str(device).startswith('cuda') and type(server.global_model).__name__=='PepsyImageTextClassifier'
 aggregation=server.pepsy_aggregation_audit
 if evaluated:assert aggregation and aggregation['actual_native_call'] and aggregation['num_updates']>0
 else:assert not server.num_samples_list
 def scalar(split):
  return {k:float(v) for k,v in server.result_dict[epoch][split].items() if isinstance(v,(int,float)) and math.isfinite(v)}
 train=server.result_dict[epoch].get('train',[])
 row=dict(fold=fold,seed=8*fold,round=epoch,q=args.missing_modailty_rate,missing_strategy='pepsy',attack_type='titim',client_trainer='ClientTITIM',model_class=type(server.global_model).__name__,device=str(device),feature_dimensions=[512,512],saved_raw_features=True,fusion=args.att_name if args.att else 'no_att',dropout=float(args.dropout_p),selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),training_samples=int(sum(server.num_samples_list)),evaluated=evaluated,aggregation=aggregation,titim_client_statistics=[{k:float(v) for k,v in x.items() if k.startswith('titim_')} for x in train if any(k.startswith('titim_') for k in x)],pepsy_client_statistics=[{k:(v if isinstance(v,list) else float(v)) for k,v in x.items() if k.startswith('pepsy_')} for x in train],timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),TA='not_applicable_image_only_attack',MBR='not_defined_for_titim')
 if evaluated:
  row.update(dev=scalar('dev'),test=scalar('test'),attack=scalar('attack_test'),attack_test_mm=scalar('attack_test_mm'),attack_test_img_only=scalar('attack_test_img_only'),best_epoch=int(server.best_epoch))
  row['metrics_pct']=dict(CA=row['test']['acc'],ASR=100*row['attack']['ASR_MM'],IA=100*row['attack']['ASR_I'],clean_MacroF1=row['test']['f1'])
  if server.best_epoch==epoch:torch.save(server.global_model.state_dict(),server.result_path/'model.pt')
 with (Path(result_dir)/f'pepsy_titim_rounds_fold{fold}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 progress={k:v for k,v in row.items() if k not in ('dev','test','attack','attack_test_mm','attack_test_img_only','titim_client_statistics','pepsy_client_statistics')}
 path=Path(args.data_dir)/'training_progress.json';temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(progress,indent=2));temp.replace(path)
 print('[clip-pepsy-titim round]',json.dumps(progress),flush=True)

def finalize_fold(args,result,server):
 assert enabled(args);attack=server.best_attack_result_dict['attack_test']
 result.update(best_epoch=int(server.best_epoch),CA=float(server.best_test_dict['acc']),ASR=100*float(attack['ASR_MM']),IA=100*float(attack['ASR_I']),TA='not_applicable_image_only_attack',MBR='not_defined_for_titim')
