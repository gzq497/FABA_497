"""Separate saved-pixel generation and frozen CLIP extraction for TITIM."""
import argparse
import copy
import json
import logging
from pathlib import Path
import pickle
import random

from PIL import Image

from attacks.titim import UPSTREAM_COMMIT
from attacks.titim.trigger import BadNetSquareConfig, apply_badnet_square
from attacks.titim.prepare_features import save_raw, poison_count, image_present
from features.feature_processing.clip_features import sha256, atomic_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)

def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.pkl.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)

def dump_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--stage', choices=['raw','extract'], required=True)
    parser.add_argument('--q', type=float, required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--malicious_ratio', type=float, default=0.1)
    parser.add_argument('--poison_ratio', type=float, default=0.1)
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2024)
    args=parser.parse_args()
    out=Path(args.output_dir)
    tag=str(args.q).replace('.','')
    root=out/f'attacks/titim/crisis-mmd/alpha50/mm{tag}/mr{str(args.malicious_ratio).replace(".","")}_pr{str(args.poison_ratio).replace(".","")}_tl{args.target}_seed{args.seed}'
    part_path=out/'partition/crisis-mmd/partition_alpha50.json'
    sim_path=out/f'simulation_feature/crisis-mmd/mm{tag}_alpha50.json'
    part=json.loads(part_path.read_text());sim=json.loads(sim_path.read_text())
    raw=root/'raw_samples'
    if args.stage=='raw':
        if (root/'metadata.raw.json').exists():
            print('RAW_ALREADY_COMPLETE',args.q,flush=True);return
        root.mkdir(parents=True,exist_ok=True);raw.mkdir(exist_ok=True)
        clients=sorted(k for k in part if k not in ['dev','test'])
        eligible=[cid for cid in clients if any(image_present(sim[cid],i) for i in range(len(part[cid])))]
        count=int(round(len(clients)*args.malicious_ratio))
        if args.malicious_ratio>0 and count==0:count=1
        count=min(count,len(eligible))
        malicious=sorted(random.Random(args.seed).sample(eligible,count))
        assert malicious
        rng=random.Random(args.seed)
        meta=dict(attack_type='titim', attack_name='TITIM-BadNet-Square-Mix',source_commit=UPSTREAM_COMMIT,
                  dataset='crisis-mmd',missing_rate=args.q,alpha=5.0,target_label=args.target,
                  malicious_ratio=args.malicious_ratio,poison_ratio=args.poison_ratio,seed=args.seed,
                  image_only=True,text_modified=False,supported_attack_metrics=['CA','ASR_MM','ASR_I'],
                  malicious_clients=malicious,client_count=len(clients),image_capable_client_count=len(eligible),
                  client_selection='original TITIM: sample from q image-capable clients; cap by eligible count',
                  q_dependent_prepare=True,train_intensities=[0.3,1.0],eval_intensity=1.0,block_ratio=0.125,margin_ratio=0.03125,
                  intensity_assignment='uniform_random_over_configured_train_intensities',
                  poisoned_keys={},poisoned_modes={},attack_test_keys=[],attack_test_modes={},
                  raw_train_manifest=str(raw/'train_manifest.jsonl'),raw_attack_manifest=str(raw/'attack_test_manifest.jsonl'),
                  partition_sha256=sha256(part_path),simulation_sha256=sha256(sim_path),status='raw_complete',raw_only=True,
                  raw_sample_root=str(raw),pipeline_order=['all five raw saved PNG/TXT','all clean CLIP features','all five poisoned CLIP extraction','verify hashes','smoke','train'])
        train_rows=[];test_rows=[]
        def generate(record,split,client_id,intensity,index):
            import shutil
            assert shutil.disk_usage(out).free>2*1024**3, 'Data disk below 2 GiB reserve'
            key,source,label,text=record
            source=Path(source)
            with Image.open(source) as image:
                poisoned=apply_badnet_square(image,BadNetSquareConfig(intensity,0.125,0.03125))
            row=save_raw(raw,split,client_id,key,source,text,poisoned,label,args.target,intensity)
            row.update(record_index=index,source_img_sha256=sha256(source),original_img_sha256=sha256(Path(row['original_img_path'])),poisoned_img_sha256=sha256(Path(row['poisoned_img_path'])))
            return row
        for cid in malicious:
            available=[i for i,r in enumerate(part[cid]) if r[2]!=args.target and image_present(sim[cid],i)]
            indices=rng.sample(available,poison_count(len(available),args.poison_ratio))
            meta['poisoned_keys'][cid]=[];meta['poisoned_modes'][cid]={}
            for index in indices:
                row=generate(part[cid][index],'train',cid,rng.choice([0.3,1.0]),index)
                train_rows.append(row);key=str(row['sample_key'])
                meta['poisoned_keys'][cid].append(key)
                meta['poisoned_modes'][cid][key]=dict(mode='raw_image_only',intensity=row['train_or_eval_intensity'],raw_img_path=row['poisoned_img_path'],text_modified=False)
        eligible_test=[i for i,r in enumerate(part['test']) if r[2]!=args.target]
        indices=rng.sample(eligible_test,min(poison_count(len(part['test']),args.poison_ratio),len(eligible_test)))
        assert indices
        for index in indices:
            row=generate(part['test'][index],'attack_test',None,1.0,index);test_rows.append(row)
            key=str(row['sample_key']);meta['attack_test_keys'].append(key)
            meta['attack_test_modes'][key]=dict(mode='raw_image_only',intensity=1.0,raw_img_path=row['poisoned_img_path'],text_modified=False)
        dump_lines(Path(meta['raw_train_manifest']),train_rows);dump_lines(Path(meta['raw_attack_manifest']),test_rows)
        meta.update(raw_train_count=len(train_rows),raw_attack_test_count=len(test_rows),attack_eval_count=len(test_rows),
                    raw_train_manifest_sha256=sha256(Path(meta['raw_train_manifest'])),raw_attack_manifest_sha256=sha256(Path(meta['raw_attack_manifest'])))
        atomic_json(root/'metadata.raw.json',meta)
        print(json.dumps(dict(stage='raw',q=args.q,status='complete',train=len(train_rows),attack_test=len(test_rows),malicious_clients=len(malicious))),flush=True)
        return
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    if (root/'metadata.json').exists():
        m=json.loads((root/'metadata.json').read_text());assert m['status']=='complete' and not m['raw_only'];return
    meta=json.loads((root/'metadata.raw.json').read_text())
    assert meta['partition_sha256']==sha256(part_path) and meta['simulation_sha256']==sha256(sim_path)
    assert meta['raw_train_manifest_sha256']==sha256(Path(meta['raw_train_manifest']))
    assert meta['raw_attack_manifest_sha256']==sha256(Path(meta['raw_attack_manifest']))
    model=CLIPModel.from_pretrained(args.model_path,local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(args.model_path,local_files_only=True)
    assert not any(p.requires_grad for p in model.parameters())
    img_dir=out/'feature/img/clip_vit_b32_img/crisis-mmd/alpha50'
    text_dir=out/'feature/text/clip_vit_b32_text/crisis-mmd/alpha50'
    manifest=[]
    for split in ['train','attack']:
        field='raw_train_manifest' if split=='train' else 'raw_attack_manifest'
        rows=[json.loads(s) for s in Path(meta[field]).read_text().splitlines()]
        for row in rows:
            for path_field,hash_field in [('original_img_path','original_img_sha256'),('poisoned_img_path','poisoned_img_sha256'),('clean_text_path','clean_text_sha256')]:
                assert sha256(Path(row[path_field]))==row[hash_field]
        manifest.extend(rows)
    features={}
    with torch.inference_mode():
        for start in range(0,len(manifest),64):
            batch=manifest[start:start+64]
            images=[]
            for row in batch:
                with Image.open(row['poisoned_img_path']) as image:images.append(image.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt').to('cuda')
            values=model.get_image_features(pixel_values=inputs['pixel_values'])
            values=torch.nn.functional.normalize(values,dim=-1).cpu().numpy().astype(np.float32)
            for row,array in zip(batch,values):
                assert array.shape==(512,) and np.isfinite(array).all()
                features[row['poisoned_img_path']]=array[None,:].copy()
            print(f'EXTRACT_PROGRESS q={args.q} done={min(start+64,len(manifest))}/{len(manifest)}',flush=True)
    train=[json.loads(s) for s in Path(meta['raw_train_manifest']).read_text().splitlines()]
    test=[json.loads(s) for s in Path(meta['raw_attack_manifest']).read_text().splitlines()]
    artifacts=[]
    for cid in meta['malicious_clients']:
        clean=load_pickle(img_dir/f'{cid}.pkl');poisoned=copy.deepcopy(clean)
        for row in train:
            if row['client_id']!=cid:continue
            index=row['record_index'];assert str(clean[index][0])==row['sample_key']
            poisoned[index][-1]=features[row['poisoned_img_path']];poisoned[index][-2]=args.target
        path=root/f'poisoned_img/{cid}.pkl';save_pickle(path,poisoned);artifacts.append(path)
    clean_img=load_pickle(img_dir/'test.pkl');clean_text=load_pickle(text_dir/'test.pkl')
    attack_img=[];attack_text=[]
    for row in test:
        i=row['record_index'];assert str(clean_img[i][0])==str(clean_text[i][0])==row['sample_key']
        image=copy.deepcopy(clean_img[i]);image[-1]=features[row['poisoned_img_path']]
        attack_img.append(image);attack_text.append(copy.deepcopy(clean_text[i]))
    for rel,records in [('attack_img/test.pkl',attack_img),('attack_text/test.pkl',attack_text)]:
        path=root/rel;save_pickle(path,records);artifacts.append(path)
    meta.update(status='complete',raw_only=False,extraction_source='verified_saved_png_txt',
                feature_extractors=dict(image='clip_vit_b32_img',text='clip_vit_b32_text'),
                encoders_frozen=True,clip_revision=Path(args.model_path).name,clip_weights_sha256=sha256(Path(args.model_path)/'pytorch_model.bin'),
                feature_artifact_sha256={str(p.relative_to(root)):sha256(p) for p in artifacts})
    atomic_json(root/'metadata.json',meta)
    print(json.dumps(dict(stage='extract',q=args.q,status='complete',feature_files=len(artifacts))),flush=True)

if __name__=='__main__':main()
