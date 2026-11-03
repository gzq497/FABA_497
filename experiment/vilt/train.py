"""No-Attack FedAvg over cached frozen ViLT three-path representations.

Run extraction first; this command never extracts, downloads or launches a queue.
"""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.vilt_models import CachedViLTClassifier, MODEL_ID, REVISION, MISSING_POLICY
from dataloader.vilt_data import (SUPPORTED_DATASETS, validate_cache, CachedViLTDataset,
                                  client_profiles, sha256)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();truth=[];pred=[];top5_correct=0;loss_sum=0.
    for features,labels in loader:
        features,labels=features.to(device),labels.to(device)
        logits=model(features)
        assert torch.isfinite(logits).all()
        loss_sum+=float(nn.functional.cross_entropy(logits,labels,reduction='sum'))
        top5_correct+=int((logits.topk(min(5,logits.shape[-1]),dim=-1).indices==labels[:,None]).any(dim=1).sum())
        truth.extend(labels.cpu().tolist());pred.extend(logits.argmax(dim=-1).cpu().tolist())
    assert truth
    counts=np.bincount(pred)
    return dict(acc=100*accuracy_score(truth,pred),f1=100*f1_score(truth,pred,average='macro',zero_division=0),
                uar=100*recall_score(truth,pred,average='macro',zero_division=0),
                top5_acc=100*top5_correct/len(truth),loss=loss_sum/len(truth),sample=len(truth),
                pred_majority_class=int(counts.argmax()),pred_majority_ratio=float(counts.max()/len(truth)))


def average_updates(updates,sample_counts):
    total=sum(sample_counts)
    assert total>0
    averaged={key:torch.zeros_like(value) for key,value in updates[0].items()}
    for state,count in zip(updates,sample_counts):
        for key in averaged:
            averaged[key].add_(state[key],alpha=count/total)
    assert all(torch.isfinite(v).all() for v in averaged.values())
    return averaged


def run(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    partition,cache,manifest=validate_cache(args.cache_root,args.data_root,args.dataset,args.alpha)
    profiles,simulation_file=client_profiles(args.data_root,args.dataset,partition,args.q,args.alpha)
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'config.json').exists(),'Output already contains an experiment; refusing overwrite'
    config=dict(vars(args),model_id=MODEL_ID,revision=REVISION,missing_policy=MISSING_POLICY,
                frozen_backbone=True,trainable='classifier only',backdoor_attack=False,
                feature_manifest_sha256=sha256(cache/'manifest.json'),simulation_sha256=sha256(simulation_file),
                selection='maximum paired-clean dev Macro-F1; same-round test',
                sampling='all training clients; fully missing skipped without replacement',
                seed_runs=[8*i for i in range(1,args.num_seeds+1)])
    atomic_json(output/'config.json',config)
    clients=sorted(profiles)
    sample_count=int(args.sample_rate*len(clients))
    if not 1<=sample_count<=len(clients):raise ValueError('Invalid sampled client count')
    loaders={}
    for client in clients:
        if profiles[client] is None:continue
        ds=CachedViLTDataset(cache/f'{client}.npz',profiles[client])
        loaders[client]=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    dev=DataLoader(CachedViLTDataset(cache/'dev.npz','paired'),batch_size=args.batch_size,shuffle=False)
    test=DataLoader(CachedViLTDataset(cache/'test.npz','paired'),batch_size=args.batch_size,shuffle=False)
    results={}
    for fold in range(1,args.num_seeds+1):
        seed=8*fold;seed_all(seed)
        model=CachedViLTClassifier(SUPPORTED_DATASETS[args.dataset],args.hid_size,args.dropout).to(args.device)
        fold_dir=output/f'fold{fold}';fold_dir.mkdir()
        best=None
        for round_index in range(args.rounds):
            selected=[clients[int(i)] for i in np.random.RandomState(round_index).choice(len(clients),sample_count,replace=False)]
            updates=[];sample_counts=[];active=[];started=time.monotonic()
            for client in selected:
                if client not in loaders:continue
                local=copy.deepcopy(model).train()
                optimizer=torch.optim.SGD(local.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=1e-5)
                for _ in range(args.local_epochs):
                    for features,labels in loaders[client]:
                        optimizer.zero_grad(set_to_none=True)
                        logits=local(features.to(args.device))
                        loss=nn.functional.cross_entropy(logits,labels.to(args.device));assert torch.isfinite(loss)
                        loss.backward();nn.utils.clip_grad_norm_(local.parameters(),10.0);optimizer.step()
                updates.append({k:v.detach().clone() for k,v in local.state_dict().items()})
                sample_counts.append(len(loaders[client].dataset));active.append(client)
                del local
            record=dict(fold=fold,seed=seed,round=round_index,q=args.q,device=args.device,
                        selected_clients=selected,update_clients=active,
                        skipped_clients=[c for c in selected if c not in loaders],
                        num_updates=len(updates),training_samples=sum(sample_counts),
                        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),evaluated=False)
            if updates:
                model.load_state_dict(average_updates(updates,sample_counts))
                record.update(evaluated=True,dev=evaluate(model,dev,args.device),test=evaluate(model,test,args.device))
                if best is None or record['dev']['f1']>best['dev']['f1']:
                    best=copy.deepcopy(record)
                    torch.save(model.state_dict(),fold_dir/'model.pt')
            record['elapsed_seconds']=time.monotonic()-started
            with (fold_dir/'rounds.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
            atomic_json(output/'progress.json',{k:v for k,v in record.items() if k not in ['dev','test']})
            print(json.dumps({k:v for k,v in record.items() if k not in ['selected_clients','update_clients','skipped_clients']}),flush=True)
        if best is None:raise RuntimeError('No valid training updates in this seed run')
        results[f'fold{fold}']=dict(best_epoch=best['round'],seed=seed,**best['test'])
        atomic_json(output/'result.json',results)
    metrics=['acc','f1','uar','top5_acc']
    folds=list(results.values())
    results['average']={m:math.fsum(float(f[m]) for f in folds)/len(folds) for m in metrics}
    results['std']={m:statistics.stdev(float(f[m]) for f in folds) if len(folds)>1 else 0. for m in metrics}
    results['reporting']=dict(unit='percent',std_ddof=1,seed_runs=len(folds),cross_validation=False)
    atomic_json(output/'result.json',results)
    print('VILT_CLEAN_TRAINING_COMPLETE',str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=sorted(SUPPORTED_DATASETS),required=True)
    p.add_argument('--data-root',default='/hy-tmp/fed/output')
    p.add_argument('--cache-root',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--q',type=float,choices=[0.1,0.3,0.5,0.7,0.9],required=True)
    p.add_argument('--alpha',type=float,default=5.0)
    p.add_argument('--rounds',type=int,default=200)
    p.add_argument('--num-seeds',type=int,default=5,choices=range(1,6))
    p.add_argument('--sample-rate',type=float,default=0.1)
    p.add_argument('--local-epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=0.05)
    p.add_argument('--hid-size',type=int,default=128)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if min(a.rounds,a.local_epochs,a.batch_size,a.hid_size)<1:p.error('Training sizes must be positive')
    run(a)
