"""Observational audit only; native BadCM-3path loss, PEPSY model and aggregation unchanged."""
import datetime,json,math
from pathlib import Path
import torch
from trainers.server_trainer import Server

def enabled(args):
 return args.dataset=='crisis-mmd' and args.img_feat=='clip_vit_b32_img' and args.text_feat=='clip_vit_b32_text' and args.backdoor_attack and args.attack_type=='badcm_3p' and args.missing_strategy=='pepsy'
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
    assert enabled(args) and str(device).startswith("cuda")
    row=dict(fold=fold,seed=8*fold,round=epoch,q=args.missing_modailty_rate,missing_strategy=args.missing_strategy,
             attack_type='badcm_3p',client_trainer='ClientBadCM3P',device=str(device),
             selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],
             skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),
             training_samples=sum(server.num_samples_list),evaluated=evaluated,
             timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    row.update(model_class=type(server.global_model).__name__,feature_dimensions=[512,512],saved_raw_features=True,aggregation=server.pepsy_aggregation_audit)
    row['pepsy_client_statistics']=[{k:(v if isinstance(v,list) else float(v)) for k,v in x.items() if k.startswith('pepsy_')} for x in server.result_dict[epoch].get('train',[])]
    if evaluated:assert row['aggregation'] and row['aggregation']['actual_native_call']
    else:assert not server.num_samples_list
    row['three_path']=server.result_dict[epoch].get('three_path',{})
    row.update(fusion=args.att_name,attention=bool(args.att),heads=int(args.d_head),dropout=float(args.dropout_p))
    row['badcm_client_statistics']=[{k:float(v) for k,v in r.items() if k.startswith('three_path_')} for r in server.result_dict[epoch].get('train',[]) if any(k.startswith('three_path_') for k in r)]
    if evaluated:
        def scalar(split):
            return {k:float(v) for k,v in server.result_dict[epoch][split].items() if isinstance(v,(int,float)) and math.isfinite(v)}
        row.update(dev=scalar('dev'),test=scalar('test'),attack=scalar('attack_test'),attack_test_mm=scalar('attack_test_mm'),attack_test_img_only=scalar('attack_test_img_only'),attack_test_text_only=scalar('attack_test_text_only'),best_epoch=int(server.best_epoch))
        row['metrics_pct']=dict(CA=row['test']['acc'],ASR=100*row['attack']['ASR_Joint'],IA=100*row['attack']['ASR_I'],TA=100*row['attack']['ASR_T'],clean_MacroF1=row['test']['f1'],MBR=row['attack']['MBR'])
        if server.best_epoch==epoch:torch.save(server.global_model.state_dict(),server.result_path/'model.pt')
    with (Path(result_dir)/f'badcm_3p_rounds_fold{fold}.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')
    progress={k:v for k,v in row.items() if k not in ['dev','test','attack','attack_test_mm','attack_test_img_only','attack_test_text_only','badcm_client_statistics','pepsy_client_statistics']}
    path=Path(args.data_dir)/'training_progress.json';temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(progress,indent=2));temp.replace(path)

def finalize_fold(args,result,server):
    assert enabled(args)
    attack=server.best_attack_result_dict['attack_test']
    result.update(best_epoch=int(server.best_epoch),CA=float(server.best_test_dict['acc']),ASR=100*float(attack['ASR_Joint']),IA=100*float(attack['ASR_I']),TA=100*float(attack['ASR_T']))
