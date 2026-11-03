"""FABA/attack_test2: save all raw inputs first, then encode their saved bytes."""
import argparse
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AlignModel as CLIPModel, AlignProcessor as CLIPProcessor

from attacks.attack_test2 import prepare_raw_trigger as original
from attacks.bad_mfl.prepare_clip_food101 import (
    atomic_json, digest_file, load_pickle, write_pickle, save_raw, read_raw_records,
)

ROOT = Path('/hy-tmp/fed')
SNAPSHOT = ROOT / '.model_cache/align-base/e96a37facc7b1f59090ece82293226b817afd6ba'


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=['raw', 'extract', 'verify'], required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--raw_data_dir', default=str(ROOT / 'data'))
    p.add_argument('--dataset', default='hateful_memes', choices=['hateful_memes'])
    p.add_argument('--alpha', type=float, default=5.0)
    p.add_argument('--missing_modailty_rate', type=float, required=True)
    p.add_argument('--bd_malicious_ratio', type=float, default=0.1)
    p.add_argument('--bd_poison_ratio', type=float, default=0.1)
    p.add_argument('--bd_target_label', type=int, default=0)
    p.add_argument('--bd_seed', type=int, default=2024)
    p.add_argument('--bd_trigger_size', type=int, default=24)
    p.add_argument('--model_path', default=str(SNAPSHOT))
    p.add_argument('--attack_test1_text_charset', default='abcdefghijklmnopqrstuvwxyz0123456789')
    p.add_argument('--attack_test1_text_trigger_budget', type=int, default=5)
    p.add_argument('--attack_test1_text_beam_width', type=int, default=5)
    original.add_raw_prepare_args(p)
    p.add_argument('--smoke_limit', type=int, default=0, help='Only for isolated integration fixtures.')
    p.set_defaults(missing_modality=True, missing_strategy='pmcmfl', attack_type='attack_test2',
                   img_feature_type='align_base_img', text_feature_type='align_base_text',
                   attack_test2_raw_text_max_length=64)
    return p.parse_args()


def signature(a):
    import hashlib
    values = {k: v for k, v in vars(a).items() if k != 'stage'}
    for key, path in [('partition', Path(a.output_dir)/f'partition/{a.dataset}/partition_alpha50.json'),
                      ('simulation', Path(a.output_dir)/f'simulation_feature/{a.dataset}/mm{original._tag(a.missing_modailty_rate)}_alpha50.json')]:
        values[key + '_sha256'] = digest_file(path)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class AlignRawVLTrojanImageTrigger(original.RawVLTrojanImageTrigger):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.mode=='blend':
            gen=torch.Generator(device='cpu');gen.manual_seed(int(kwargs.get('seed',2024)))
            self.delta=torch.nn.Parameter((torch.randn((3,289,289),generator=gen)*.05).to(self.device))

class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip = model

    def forward(self, pixels):
        return F.normalize(self.clip.get_image_features(pixel_values=pixels), dim=-1)


class ClipTextTrigger(original.RawTextBeamTrigger):
    @torch.no_grad()
    def _embed(self, texts):
        inputs = self.fm.tokenizer(list(texts), return_tensors='pt', padding=True,
                                   truncation=True, max_length=64).to('cuda')
        return F.normalize(self.fm.model.get_text_features(**inputs), dim=-1)


def load_clip(a):
    assert torch.cuda.is_available(), 'GPU is required'
    torch.set_num_threads(4)
    model = CLIPModel.from_pretrained(a.model_path, local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(a.model_path, local_files_only=True)
    processor.feature_extractor=processor.image_processor
    return model, processor


def canonical_image(processor, image):
    # Save the exact RGB image at the encoder's own resize/center-crop boundary.
    value = processor.feature_extractor(images=image.convert('RGB'), do_normalize=False, do_rescale=False,
                                         return_tensors='np')['pixel_values'][0]
    value = np.rint(value.transpose(1, 2, 0)).clip(0, 255).astype(np.uint8)
    assert value.shape == (289, 289, 3)
    return Image.fromarray(value)


def generate_raw(a):
    root = original.build_attack_root(a)
    raw_marker = root/'metadata.raw.json'
    sig = signature(a)
    if raw_marker.exists():
        metadata = json.loads(raw_marker.read_text())
        assert metadata['signature'] == sig
        print('RAW_ALREADY_COMPLETE', root, flush=True)
        return
    assert not (root/'metadata.json').exists(), 'Refusing existing completion marker'
    root.mkdir(parents=True, exist_ok=True)
    partition, lookup = original.load_partition(a)
    sim, sim_path = original.load_missing_simulation(a)
    sim_lookup = original.build_sim_lookup(sim)
    _, malicious = original.choose_malicious_clients(original.clean_img_dir(a), a.bd_malicious_ratio, a.bd_seed)
    support, images, texts = original.collect_visible_support(
        a, lookup, malicious, sim_lookup, a.attack_test2_raw_support_size,
        a.attack_test2_raw_text_support_size)
    training_keys = {str(row[0]) for cid, rows in partition.items() if cid not in ['dev', 'test'] for row in rows}
    holdout_keys = {str(row[0]) for cid in ['dev', 'test'] for row in partition[cid]}
    assert set(support['image_keys']+support['text_keys']) <= training_keys - holdout_keys
    model, processor = load_clip(a)
    transform = lambda image: processor(images=image, return_tensors='pt')['pixel_values'][0]
    trigger = AlignRawVLTrojanImageTrigger(
        ImageEncoder(model), transform, torch.device('cuda'), mode=a.attack_test2_raw_image_mode,
        patch_size=a.attack_test2_raw_patch_size, blend_alpha=a.attack_test2_raw_blend_alpha,
        eps=a.attack_test2_raw_eps, lr=a.attack_test2_raw_lr,
        source_weight=a.attack_test2_raw_source_weight, cluster_weight=a.attack_test2_raw_cluster_weight,
        opposite_weight=a.attack_test2_raw_opposite_weight, tv_weight=a.attack_test2_raw_tv_weight,
        seed=a.bd_seed)
    trigger.mean = torch.tensor(processor.feature_extractor.image_mean, device='cuda').view(3,1,1)
    trigger.std = torch.tensor(processor.feature_extractor.image_std, device='cuda').view(3,1,1)
    print('OPTIMIZE_IMAGE_TRIGGER', len(images), 'epochs', a.attack_test2_raw_epochs, flush=True)
    loss = trigger.optimize(images, a.attack_test2_raw_epochs, a.attack_test2_raw_batch_size) if images else {'skipped': True}
    trigger.save(root/'raw_triggers')
    text_trigger = ClipTextTrigger(SimpleNamespace(model=model, tokenizer=processor.tokenizer, device='cuda'),
        a.attack_test1_text_charset, a.attack_test1_text_trigger_budget, a.attack_test1_text_beam_width, 64)
    print('SEARCH_TEXT_TRIGGER', len(texts), flush=True)
    text_string, text_score = text_trigger.search(texts) if texts else ('', 0.0)
    atomic_json(root/'raw_triggers/text_trigger.json', dict(text=text_string, score=text_score))
    metadata = dict(attack_type='attack_test2', method='FABA', dataset=a.dataset, signature=sig,
        artifact_format='sparse_poison_overlay_v1', status='raw_complete', raw_only=True,
        q_dependent_prepare=True, missing_rate=a.missing_modailty_rate,
        missing_strategy='pmcmfl', alpha=a.alpha, target_label=a.bd_target_label,
        malicious_ratio=a.bd_malicious_ratio, poison_ratio=a.bd_poison_ratio, seed=a.bd_seed,
        malicious_clients=malicious, poisoned_keys={}, poisoned_modes={}, attack_test_keys=[],
        attack_test_modes={}, feature_extractors={'image':a.img_feature_type,'text':a.text_feature_type},
        align_revision=Path(a.model_path).name, align_weights_sha256=digest_file(Path(a.model_path)/'pytorch_model.bin'),
        support_policy='malicious_train_q_visible_only', support_keys=support,
        simulation_file=sim_path, partition_file=str(Path(a.output_dir)/f'partition/{a.dataset}/partition_alpha50.json'),
        raw_preprocessing='ALIGN official resize346/center-crop289 RGB, then original blend/patch; PNG extraction skips repeated geometry and applies official rescale2/255-offset1',
        raw_trigger=dict(image_optimization_loss=loss, text_trigger=text_string, text_trigger_score=text_score),
        attack_eval_ratio=a.attack_test2_raw_attack_test_ratio,
        pipeline_order=['all_five_q_saved_raw_png_txt','all_five_q_extract_saved_bytes','verify','smoke','train'])
    rng = random.Random(a.bd_seed)
    records = {'train':[], 'attack_test':[]}
    def save(raw, cid, mode, split):
        import shutil
        assert shutil.disk_usage(root).free>2*1024**3
        key = str(raw[0]); has_i=original.image_visible(mode); has_t=original.text_visible(mode)
        path = original.resolve_img_path(raw[1], a.raw_data_dir, a.dataset)
        image = None
        if has_i:
            with Image.open(path) as opened:
                image = trigger.apply_to_pil(canonical_image(processor, opened))
        text = text_trigger.append(str(raw[-1])) if has_t else None
        rec = save_raw(root, split, cid, key, image, text, raw[2], a.bd_target_label, has_i, has_t,
                       {'mode':mode,'method':'FABA','image_trigger':has_i,'text_trigger':has_t})
        rec['original_img_path'] = path
        records[split].append(rec)
        if split == 'train':
            metadata['poisoned_keys'][cid].append(key)
            metadata['poisoned_modes'][cid][key] = {'modality_mode':mode,'image_trigger_applied':has_i,'text_trigger_applied':has_t}
        else:
            metadata['attack_test_keys'].append(key)
        n = len(records[split])
        if n % 100 == 0:
            import shutil
            assert shutil.disk_usage(a.output_dir).free > 2*1024**3, 'Data disk below 2 GiB reserve'
            print('RAW_PROGRESS', split, n, 'q', a.missing_modailty_rate, flush=True)
    for cid in malicious:
        metadata['poisoned_keys'][cid]=[]; metadata['poisoned_modes'][cid]={}
        candidates=[]
        for row in original.load_feature_dict(original.clean_img_dir(a)/(cid+'.pkl')):
            key = str(row[0])
            mode = original.modality_mode_from_sim(original.get_sim_vector(sim_lookup,cid,key))
            if int(row[-2]) != a.bd_target_label and mode != 'both_missing': candidates.append((key,mode))
        n = round(len(candidates)*a.bd_poison_ratio)
        if a.bd_poison_ratio>0 and candidates: n=max(1,n)
        selected=rng.sample(candidates,min(n,len(candidates)))
        if a.smoke_limit: selected=selected[:a.smoke_limit]
        for key,mode in selected: save(lookup[key],cid,mode,'train')
    test_rows=[row for row in partition['test'] if int(row[2])!=a.bd_target_label]
    count=math.ceil(len(test_rows)*a.attack_test2_raw_attack_test_ratio)
    chosen=test_rows if count>=len(test_rows) else rng.sample(test_rows,count)
    if a.smoke_limit: chosen=chosen[:a.smoke_limit]
    for row in chosen: save(row,None,'image_text','attack_test')
    for split, rows in records.items():
        path=root/'raw_samples'/(split+'_manifest.jsonl'); path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        metadata['raw_'+split+'_manifest']=str(path)
        metadata['raw_'+split+'_manifest_sha256']=digest_file(path)
        metadata['raw_'+split+'_count']=len(rows)
    metadata['attack_eval_count']=len(records['attack_test'])
    atomic_json(raw_marker,metadata)
    print('RAW_COMPLETE', a.missing_modailty_rate, {k:len(v) for k,v in records.items()}, flush=True)


def verify(a):
    root=original.build_attack_root(a)
    metadata=json.loads((root/'metadata.json').read_text())
    assert metadata['signature']==signature(a) and metadata['status']=='complete' and not metadata['raw_only']
    for split in ['train','attack_test']: read_raw_records(root,metadata,split)
    for name, expected in metadata['feature_artifact_sha256'].items():
        assert digest_file(root/name)==expected
        rows=load_pickle(root/name)
        values=[v['feature'] for v in rows.values()] if isinstance(rows,dict) else [r[-1] for r in rows]
        for x in values:
            if x is not None:
                assert x.shape==(1,640) and x.dtype==np.float32 and np.isfinite(x).all()
                assert np.isclose(np.linalg.norm(x),1,atol=1e-5)
    print('VERIFIED',a.missing_modailty_rate,flush=True)


def extract(a):
    root=original.build_attack_root(a)
    if (root/'metadata.json').exists(): return verify(a)
    metadata=json.loads((root/'metadata.raw.json').read_text())
    assert metadata['signature']==signature(a)
    train,test=[read_raw_records(root,metadata,s) for s in ['train','attack_test']]
    model,processor=load_clip(a)
    # All feature values come from checksum-verified files, never in-memory trigger outputs.
    @torch.no_grad()
    def encode(rows):
        result_i={};result_t={}
        for modality,field,batch in [('img','poisoned_img_path',4),('text','poisoned_text_path',128)]:
            selected=[(idx,r) for idx,r in enumerate(rows) if r[field]]
            result=result_i if modality=='img' else result_t
            for start in range(0,len(selected),batch):
                chunk=selected[start:start+batch]
                if modality=='img':
                    images=[]
                    for _,r in chunk:
                        with Image.open(r[field]) as im: images.append(im.convert('RGB'))
                    pixels=processor.feature_extractor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt')['pixel_values'].cuda()
                    features=model.get_image_features(pixel_values=pixels)
                else:
                    texts=[Path(r[field]).read_text() for _,r in chunk]
                    inputs=processor.tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=64).to('cuda')
                    features=model.get_text_features(**inputs)
                values=F.normalize(features,dim=-1).float().cpu().numpy()
                assert values.shape==(len(chunk),640) and np.isfinite(values).all()
                for (idx,_),value in zip(chunk,values): result[idx]=value[None,:].copy()
                if start//batch%10==0: print('EXTRACT_PROGRESS',modality,start+len(chunk),'/',len(selected),flush=True)
        return result_i,result_t
    images,texts=encode(train)
    for cid in metadata['malicious_clients']:
        for modality,values in [('img',images),('text',texts)]:
            overlay={r['sample_key']:dict(feature=values.get(idx),label=r['target_label'])
                     for idx,r in enumerate(train) if r['client_id']==cid}
            write_pickle(root/('poisoned_'+modality)/(cid+'.pkl'),overlay)
    images,texts=encode(test)
    for modality,values in [('img',images),('text',texts)]:
        rows=[[r['sample_key'],r['original_img_path'],r['original_label'],values[idx]] for idx,r in enumerate(test)]
        write_pickle(root/('attack_'+modality)/'test.pkl',rows)
    metadata.update(status='complete',raw_only=False,extraction_source='verified_saved_png_txt')
    metadata['feature_artifact_sha256']={str(p.relative_to(root)):digest_file(p)
        for sub in ['poisoned_img','poisoned_text','attack_img','attack_text'] for p in (root/sub).glob('*.pkl')}
    atomic_json(root/'metadata.json',metadata)
    verify(a)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    a=arguments()
    assert a.alpha==5.0
    random.seed(a.bd_seed);np.random.seed(a.bd_seed);torch.manual_seed(a.bd_seed);torch.cuda.manual_seed_all(a.bd_seed)
    {'raw':generate_raw,'extract':extract,'verify':verify}[a.stage](a)
