"""Round and best-checkpoint audit for the isolated frozen-CLIP BadCM entry."""
import datetime
import json
import math
from pathlib import Path
import torch

def enabled(args):
    return args.dataset=='crisis-mmd' and args.img_feat=='clip_vit_b32_img' and args.text_feat=='clip_vit_b32_text' and args.backdoor_attack and args.attack_type=='badcm'

def append_round(args,server,skipped,device,fold,epoch,result_dir,evaluated):
    assert enabled(args)
    row=dict(fold=fold,seed=8*fold,round=epoch,q=args.missing_modailty_rate,missing_strategy=args.missing_strategy,
             attack_type='badcm',client_trainer='ClientBadCM',device=str(device),
             selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],
             skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),
             training_samples=sum(server.num_samples_list),evaluated=evaluated,
             timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    row.update(fusion=args.att_name,attention=bool(args.att),heads=int(args.d_head),dropout=float(args.dropout_p))
    row['badcm_client_statistics']=[{k:float(v) for k,v in r.items() if k.startswith('badcm_')} for r in server.result_dict[epoch].get('train',[]) if any(k.startswith('badcm_') for k in r)]
    if evaluated:
        def scalar(split):
            return {k:float(v) for k,v in server.result_dict[epoch][split].items() if isinstance(v,(int,float)) and math.isfinite(v)}
        row.update(dev=scalar('dev'),test=scalar('test'),attack=scalar('attack_test'),attack_test_mm=scalar('attack_test_mm'),attack_test_img_only=scalar('attack_test_img_only'),attack_test_text_only=scalar('attack_test_text_only'),best_epoch=int(server.best_epoch))
        row['metrics_pct']=dict(CA=row['test']['acc'],ASR=100*row['attack']['ASR_Joint'],IA=100*row['attack']['ASR_I'],TA=100*row['attack']['ASR_T'])
        if server.best_epoch==epoch:torch.save(server.global_model.state_dict(),server.result_path/'model.pt')
    with (Path(result_dir)/f'badcm_rounds_fold{fold}.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')
    progress={k:v for k,v in row.items() if k not in ['dev','test','attack','attack_test_mm','attack_test_img_only','attack_test_text_only','badcm_client_statistics']}
    path=Path(args.data_dir)/'training_progress.json';temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(progress,indent=2));temp.replace(path)

def finalize_fold(args,result,server):
    assert enabled(args)
    attack=server.best_attack_result_dict['attack_test']
    result.update(best_epoch=int(server.best_epoch),CA=float(server.best_test_dict['acc']),ASR=100*float(attack['ASR_Joint']),IA=100*float(attack['ASR_I']),TA=100*float(attack['ASR_T']))
