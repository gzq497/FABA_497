"""Shared poison-only, sample-balanced three-path local training."""
from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from trainers.evaluation import EvalMetric


THREE_PATH_ATTACKS = ('badcm_3p', 'bad_mfl_3p')
_ALIASES = {
    'badcm-3path': 'badcm_3p', 'badcm_3path': 'badcm_3p', 'badcm-3p': 'badcm_3p',
    'bad-mfl-3path': 'bad_mfl_3p', 'bad_mfl_3path': 'bad_mfl_3p', 'bad-mfl-3p': 'bad_mfl_3p',
}


def normalize_attack_type(name):
    name = str(name).lower()
    return _ALIASES.get(name, name)


def configure_three_path(args):
    args.attack_type = normalize_attack_type(args.attack_type)
    if args.attack_type not in THREE_PATH_ATTACKS:
        return
    if str(getattr(args, 'missing_strategy', 'zero')).lower() not in ('zero', 'pepsy', 'pmcmfl'):
        raise ValueError('Three-path attacks require missing_strategy=zero, pepsy or pmcmfl.')
    if getattr(args, 'modality', 'multimodal') != 'multimodal' or args.dataset == 'ptb-xl':
        raise ValueError('Three-path attacks require multimodal single-label classification.')
    if getattr(args, 'fed_alg', 'fed_avg') not in ('fed_avg', 'fed_opt', 'fed_prox'):
        raise ValueError('Three-path attacks support the original FedAvg/FedOpt/FedProx optimizers.')


def _row_losses(criterion, log_probs, labels):
    per_row = copy.copy(criterion)
    per_row.reduction = 'none'
    losses = per_row(log_probs, labels)
    if losses.ndim != 1 or len(losses) != len(labels):
        raise ValueError('Three-path training requires one classification loss per sample.')
    return losses


def three_path_loss(model, criterion, x_i, x_t, len_i, len_t, labels, poison_mask, poison_weight,
                    pepsy_args=None, pmcmfl_helper=None):
    """Rows must have at least one modality; extra paths apply only to dual poisons.

    The main forward pass is reused for clean and single-modality poison rows.
    Two extra forwards cover only dual-modality poison rows. Each poison row
    receives equal total weight, regardless of its number of legal paths.
    """
    present_i, present_t = len_i > 0, len_t > 0
    if not bool((present_i | present_t).all()):
        raise ValueError('Filter all-missing rows before computing three-path loss.')
    x_i = x_i.masked_fill(~present_i.view(-1, *([1] * (x_i.ndim - 1))), 0)
    x_t = x_t.masked_fill(~present_t.view(-1, *([1] * (x_t.ndim - 1))), 0)
    # PEPSY reconstructs missing representations inside every legal forward.
    # Its batch-level auxiliary objective is charged once on the original batch,
    # as in ClientBadMFL, rather than multiplied by the number of poison paths.
    aux = None
    if pepsy_args is not None and pmcmfl_helper is not None:
        raise ValueError('Choose one missing-modality strategy per batch.')
    pmcmfl_total = None
    pmcmfl_stats = {}
    if pmcmfl_helper is not None:
        pmcmfl_total, log_probs, main_loss, pmcmfl_stats = pmcmfl_helper._compute_loss(
            x_i.float(), x_t.float(), len_i, len_t, labels)
    elif pepsy_args is not None:
        logits, _, aux = model(x_i.float(), x_t.float(), len_i, len_t, return_pepsy=True)
    else:
        logits, _ = model(x_i.float(), x_t.float(), len_i, len_t)
    if pmcmfl_helper is None:
        log_probs = torch.log_softmax(logits, dim=1)
        main_loss = criterion(log_probs, labels)
    dual = poison_mask & present_i & present_t
    stats = {
        'joint_paths': int(dual.sum()),
        'image_paths': int((poison_mask & present_i).sum()),
        'text_paths': int((poison_mask & present_t).sum()),
        'poison_samples': int(poison_mask.sum()),
    }
    auxiliary_loss = main_loss.new_zeros(())
    if pmcmfl_total is not None:
        # The original batch contributes prototype regularization exactly once.
        # Additional poison paths extend classification supervision only.
        auxiliary_loss = pmcmfl_total - main_loss
        stats.update(pmcmfl_stats)
    if aux is not None:
        auxiliary_loss = (float(getattr(pepsy_args, 'pepsy_contrastive_weight', 0.2)) * aux['contrastive_loss']
                          + float(getattr(pepsy_args, 'pepsy_prompt_reg_weight', 0.1)) * aux['prompt_loss'])
        for name, value in dict(loss_main=main_loss, loss_contrastive=aux['contrastive_loss'],
                                loss_prompt=aux['prompt_loss'], present_img=aux['present_img'],
                                present_text=aux['present_text'], alpha_old=aux['alpha_old']).items():
            stats['pepsy_' + name] = float(value.detach())
    if not bool(poison_mask.any()):
        return (pmcmfl_total if pmcmfl_total is not None else main_loss + auxiliary_loss), log_probs, main_loss, stats

    poison_rows = _row_losses(criterion, log_probs[poison_mask], labels[poison_mask])
    if bool(dual.any()):
        if pmcmfl_helper is not None:
            log_i = _pmcmfl_path_log_probs(pmcmfl_helper, x_i[dual], torch.zeros_like(x_t[dual]),
                                           len_i[dual], torch.zeros_like(len_t[dual]), labels[dual])
            log_t = _pmcmfl_path_log_probs(pmcmfl_helper, torch.zeros_like(x_i[dual]), x_t[dual],
                                           torch.zeros_like(len_i[dual]), len_t[dual], labels[dual])
        else:
            logits_i, _ = model(x_i[dual].float(), torch.zeros_like(x_t[dual]).float(),
                                len_i[dual], torch.zeros_like(len_t[dual]))
            logits_t, _ = model(torch.zeros_like(x_i[dual]).float(), x_t[dual].float(),
                                torch.zeros_like(len_i[dual]), len_t[dual])
            log_i, log_t = torch.log_softmax(logits_i, dim=1), torch.log_softmax(logits_t, dim=1)
        loss_i = _row_losses(criterion, log_i, labels[dual])
        loss_t = _row_losses(criterion, log_t, labels[dual])
        poison_rows = poison_rows.clone()
        poison_rows[dual[poison_mask]] = (poison_rows[dual[poison_mask]] + loss_i + loss_t) / 3.0
    poison_loss = poison_rows.mean()
    if bool((~poison_mask).any()):
        clean_loss = criterion(log_probs[~poison_mask], labels[~poison_mask])
        loss = poison_weight * poison_loss + (1.0 - poison_weight) * clean_loss
    else:
        clean_loss = poison_loss.new_zeros(())
        loss = poison_loss
    stats.update(poison_loss=float(poison_loss.detach()), clean_loss=float(clean_loss.detach()),
                 weighted_loss=float(loss.detach()))
    return loss + auxiliary_loss, log_probs, main_loss, stats


def _pmcmfl_path_log_probs(helper, x_i, x_t, len_i, len_t, labels):
    """Use PMCMFL's native prototype completion on each legal poison path."""
    z_i = helper.pathways.encoder_I(x_i.float(), len_i)
    z_t = helper.pathways.encoder_T(x_t.float(), len_t)
    z_i, z_t, _, _ = helper._fill_missing_with_prototypes(
        z_i, z_t, len_i, len_t, labels, helper._prototypes_to_device())
    logits, _ = helper.pathways.logits_from_embeddings(z_i, z_t)
    return helper._classification_log_probs(logits)


class ThreePathTrainingMixin:
    def update_weights(self):
        # Benign clients retain their exact original attack-class training path.
        if not self.is_malicious:
            result = super().update_weights()
            # The parent's evaluator counts repeated local-epoch visits. Keep
            # aggregation weights comparable to malicious clients' original rows.
            self.result['sample'] = int(self.result['sample']) // max(1, int(self.args.local_epochs))
            return result
        configure_three_path(self.args)
        self.model.train()
        self.eval = EvalMetric(False)
        optimizer = self._build_optimizer()
        stat_lists, counts = {}, Counter()
        pepsy_enabled = self._pepsy_enabled()
        pmcmfl_helper = self._pmcmfl_helper if self._pmcmfl_enabled() else None
        if pmcmfl_helper is not None:
            pmcmfl_helper.client_modality_stats = pmcmfl_helper._empty_modality_stats()
        observed_modalities = set()
        original_samples = 0
        for local_epoch in range(int(self.args.local_epochs)):
            for batch in self.dataloader:
                if len(batch) != 6:
                    raise ValueError('Three-path malicious training requires sample keys from the dataloader.')
                x_i, x_t, len_i, len_t, labels, keys = batch
                x_i, x_t, len_i, len_t, labels = [v.to(self.device) for v in (x_i, x_t, len_i, len_t, labels)]
                valid = (len_i > 0) | (len_t > 0)
                if pmcmfl_helper is not None:
                    pmcmfl_helper._observe_modalities(len_i > 0, len_t > 0)
                counts['skipped_rows'] += int((~valid).sum())
                if not bool(valid.any()):
                    continue
                x_i, x_t, len_i, len_t, labels = [v[valid] for v in (x_i, x_t, len_i, len_t, labels)]
                if pepsy_enabled:
                    if bool((len_i > 0).any()):
                        observed_modalities.add('img')
                    if bool((len_t > 0).any()):
                        observed_modalities.add('text')
                keys = self._filter_keys(keys, valid)
                poison = torch.tensor([str(k) in self.poisoned_keys for k in keys], device=self.device, dtype=torch.bool)
                if bool(poison.any()) and not bool((labels[poison] == int(self.args.bd_target_label)).all()):
                    raise ValueError('Poisoned-key labels do not match bd_target_label.')
                self.model.zero_grad()
                optimizer.zero_grad()
                loss, log_probs, loss_main, stats = three_path_loss(
                    self.model, self.criterion, x_i, x_t, len_i, len_t, labels, poison, self.poison_weight,
                    pepsy_args=self.args if pepsy_enabled else None, pmcmfl_helper=pmcmfl_helper)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                self.eval.append_classification_results(labels, log_probs, loss_main)
                if local_epoch == 0:
                    original_samples += len(labels)
                for name, value in stats.items():
                    if name.endswith('_paths') or name == 'poison_samples':
                        counts[name] += value
                    else:
                        self._append_stat(stat_lists, name if name.startswith(('pepsy_', 'pmcmfl_')) else 'three_path_' + name, value)
        self._finalize_result(stat_lists)
        self.result['sample'] = original_samples
        for key in ('joint_paths', 'image_paths', 'text_paths', 'poison_samples', 'skipped_rows'):
            self.result['three_path_' + key] = int(counts[key])
        self.result['three_path_poison_weight'] = self.poison_weight
        if pepsy_enabled:
            self.result['pepsy_modalities'] = sorted(observed_modalities)
            self.result['pepsy_has_img'] = float('img' in observed_modalities)
            self.result['pepsy_has_text'] = float('text' in observed_modalities)
        if pmcmfl_helper is not None:
            # Recompute on the original dataloader, never on expanded paths.
            self.prototype_update = pmcmfl_helper.compute_local_prototypes()
            self.result.update(pmcmfl_helper._client_modality_result_stats())
        logging.info('[%s] client=%s device=%s epoch=%s samples=%d J/I/T=%d/%d/%d',
                     self.args.attack_type, self.client_id, self.device,
                     getattr(self.args, 'current_global_epoch', None), original_samples,
                     counts['joint_paths'], counts['image_paths'], counts['text_paths'])


def load_sparse_features(manager, client_id, modality):
    """Overlay poison-only artifacts on clean features in memory; never copy clean artifacts."""
    import pickle
    clean_root = manager.img_feat_path if modality == 'img' else manager.text_feat_path
    clean_path = clean_root / ('alpha' + str(manager.args.alpha).replace('.', '')) / (str(client_id) + '.pkl')
    overlay_path = manager._get_attack_root() / ('poisoned_' + modality) / (str(client_id) + '.pkl')
    with clean_path.open('rb') as handle:
        rows = pickle.load(handle)
    with overlay_path.open('rb') as handle:
        overlay = pickle.load(handle)
    known = {str(row[0]) for row in rows}
    if not set(overlay).issubset(known):
        raise ValueError('Poison overlay contains keys outside its clean client.')
    for idx, row in enumerate(rows):
        change = overlay.get(str(row[0]))
        if change is not None:
            row = list(row)
            row[-2], row[-1] = change['label'], change['feature']
            rows[idx] = row
    return rows


def record_three_path_round(args, server, skip_ids, device, fold, epoch, output_dir):
    if args.attack_type not in THREE_PATH_ATTACKS:
        return
    train = server.result_dict[epoch]['train']
    record = dict(attack_type=args.attack_type, missing_strategy=getattr(args, 'missing_strategy', 'zero'),
                  fold=int(fold), round=int(epoch), device=str(device),
                  cuda_device=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  valid_clients=len(server.num_samples_list), skipped_clients=list(skip_ids),
                  original_samples=int(sum(server.num_samples_list)))
    for key in ('joint_paths', 'image_paths', 'text_paths'):
        record[key] = int(sum(r.get('three_path_' + key, 0) for r in train))
    server.result_dict[epoch]['three_path'] = record
    path = Path(output_dir) / f'three_path_fold{fold}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w' if epoch == 0 else 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
    logging.info('[three-path round] %s', json.dumps(record))


def add_fold_standard_deviations(results):
    folds = [value for key, value in results.items() if key.startswith('fold')]
    results['std'] = {}
    for metric in results.get('average', {}):
        values = [fold[metric] for fold in folds if isinstance(fold.get(metric), (int, float, np.number))]
        if values:
            results['std'][metric] = float(np.std(values, ddof=1)) if len(values) > 1 else None
