"""Audit frozen-CLIP original Bad-MFL without changing its local loss or aggregation."""
import datetime,json,math
from pathlib import Path
import torch

def enabled(args):
    return (args.img_feat=='clip_vit_b32_img' and args.text_feat=='clip_vit_b32_text'
            and args.backdoor_attack and args.attack_type=='bad_mfl')

def append_round(args,server,skipped,device,fold,epoch,result_dir,evaluated):
    if not enabled(args):return
    row=dict(fold=fold,round=epoch,q=args.missing_modailty_rate,missing_strategy=args.missing_strategy,
             attack_type='bad_mfl',client_trainer='ClientBadMFL',device=str(device),
             selected_clients=[str(server.client_ids[i]) for i in server.clients_list[epoch]],
             skipped_clients=list(map(str,skipped)),update_clients=list(map(str,server.client_update_ids)),
             training_samples=sum(server.num_samples_list),evaluated=evaluated,
             timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    train=server.result_dict[epoch].get('train',[])
    row['bad_mfl_client_statistics']=[{k:float(v) for k,v in x.items() if k.startswith('bad_mfl_')} for x in train if any(k.startswith('bad_mfl_') for k in x)]
    if evaluated:
        def scalar(split):
            return {k:float(v) for k,v in server.result_dict[epoch][split].items() if isinstance(v,(int,float)) and math.isfinite(v)}
        row.update(dev=scalar('dev'),test=scalar('test'),attack=scalar('attack_test'),best_epoch=int(server.best_epoch))
        for split in ['attack_test_mm','attack_test_img_only','attack_test_text_only']:
            row[split]=scalar(split)
        a=row['attack'];row['metrics_pct']=dict(CA=row['test']['acc'],ASR=100*a['ASR_Joint'],IA=100*a['ASR_I'],TA=100*a['ASR_T'])
        if server.best_epoch==epoch:torch.save(server.global_model.state_dict(),server.result_path/'model.pt')
    with (Path(result_dir)/f'bad_mfl_rounds_fold{fold}.jsonl').open('a') as h:h.write(json.dumps(row,allow_nan=False)+'\n')
    target=Path(args.data_dir)/'training_progress.json';temp=target.with_suffix('.json.tmp')
    temp.write_text(json.dumps({k:v for k,v in row.items() if k not in ['dev','test','attack','attack_test_mm','attack_test_img_only','attack_test_text_only','bad_mfl_client_statistics']},indent=2));temp.replace(target)

def finalize_fold(args,result,server):
    if not enabled(args):return
    result['best_epoch']=int(server.best_epoch)
    a=server.best_attack_result_dict['attack_test']
    result.update(CA=float(server.best_test_dict['acc']),ASR=100*float(a['ASR_Joint']),IA=100*float(a['ASR_I']),TA=100*float(a['ASR_T']))
