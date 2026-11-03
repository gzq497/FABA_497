"""Audit normal-task CLIP runs without changing client selection or aggregation."""
import datetime
import json
import math
import statistics
from pathlib import Path
import torch


def enabled(args):
    return str(args.img_feat) == 'clip_vit_b32_img' and str(args.text_feat) == 'clip_vit_b32_text' and not args.backdoor_attack


def add_standard_deviations(result):
    folds = [v for k,v in result.items() if k.startswith('fold')]
    result['std'] = {}
    for metric in set.intersection(*(set(f) for f in folds)):
        values = [f[metric] for f in folds]
        if all(isinstance(v, (int,float)) and math.isfinite(v) for v in values):
            result['std'][metric] = statistics.stdev(values) if len(values) > 1 else 0.0


def append_round(args, server, skipped, device, fold, epoch, result_dir, evaluated):
    if not enabled(args):
        return
    selected = [str(server.client_ids[i]) for i in server.clients_list[epoch]]
    row = dict(fold=fold, round=epoch, q=args.missing_modailty_rate,
               missing_strategy=args.missing_strategy, attack=False, device=str(device),
               selected_clients=selected, skipped_clients=list(map(str, skipped)),
               update_clients=list(map(str, server.client_update_ids)),
               training_samples=sum(server.num_samples_list), evaluated=evaluated,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if evaluated:
        metrics = lambda split: {k: (int(v) if k == 'sample' else float(v))
                                 for k,v in server.result_dict[epoch][split].items()
                                 if k in ['acc', 'f1', 'uar', 'top5_acc', 'loss', 'sample', 'pred_majority_class', 'pred_majority_ratio']}
        row.update(dev=metrics('dev'), test=metrics('test'),
                   best_epoch=int(server.best_epoch))
        # Save the initial best checkpoint too, including runs with no later improvement.
        if server.best_epoch == epoch:
            torch.save(server.global_model.state_dict(), server.result_path / 'model.pt')
    path = Path(result_dir) / f'clean_rounds_fold{fold}.jsonl'
    with path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
    progress = Path(args.data_dir) / 'training_progress.json'
    tmp = progress.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({k:v for k,v in row.items() if k not in ['dev','test']}, indent=2))
    tmp.replace(progress)
