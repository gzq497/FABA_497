"""Extract independently encoded CLIP/ALIGN PKLs in the existing FL row schema."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model.dual_encoder import FrozenDualEncoder
from constants.vision_language import PROFILES


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def extract(data_root, dataset, alpha, modality, backbone, batch_size=4):
    if modality not in ('img', 'text') or batch_size < 1:
        raise ValueError('Invalid modality/batch size.')
    root = Path(data_root)
    tag = 'alpha' + str(float(alpha)).replace('.', '')
    part = root / 'partition' / dataset / ('partition_' + tag + '.json')
    partition = json.loads(part.read_text(encoding='utf-8'))
    if not {'dev', 'test'}.issubset(partition):
        raise ValueError('Reuse the existing partition, including dev and test.')
    name = backbone.spec[modality]
    out = root / 'feature' / modality / name / dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Output is not empty; inspect rather than silently reusing features: ' + str(out))
    source = Path(backbone.source)
    weights = {p.name: sha256(p) for p in sorted(source.glob('*'))
               if p.is_file() and (p.suffix in ('.bin', '.safetensors', '.json') or p.name == 'vocab.txt')}
    meta = dict(status='extracting', architecture=backbone.architecture,
                model_id=backbone.spec['model_id'], revision=backbone.spec['revision'],
                checkpoint_files_sha256=weights, feature_name=name, feature_shape=[1, backbone.spec['dim']],
                dtype='float32', frozen=True, l2_normalized=True, independently_encoded=True,
                representation='projected pooled embedding', text_max_length=backbone.max_length,
                partition_sha256=sha256(part), dataset=dataset, alpha=alpha,
                processor=type(backbone.processor).__name__, modality=modality,
                torch_version=torch.__version__, files={}, truncated_texts=0)
    atomic_json(out / 'encoder_metadata.json', meta)
    for client, rows in partition.items():
        result = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            if any(len(row) != 4 or not isinstance(row[2], int) for row in batch):
                raise ValueError('Expected [key, image_path, label, text] partition rows.')
            if modality == 'img':
                images = []
                for row in batch:
                    p = Path(row[1])
                    if not p.is_absolute():
                        p = root / p
                    with Image.open(p) as image:
                        images.append(image.convert('RGB'))
                values = backbone.encode_images(images)
            else:
                texts = [str(row[3]) for row in batch]
                tokens = backbone.processor.tokenizer(texts, truncation=False, padding=False)['input_ids']
                meta['truncated_texts'] += sum(len(t) > backbone.max_length for t in tokens)
                values = backbone.encode_texts(texts)
            values = values.cpu().numpy().astype(np.float32, copy=False)
            for row, value in zip(batch, values):
                result.append([row[0], row[1], row[2], value[None].copy()])
        p = out / (str(client) + '.pkl')
        tmp = p.with_suffix('.pkl.tmp')
        with tmp.open('wb') as f:
            pickle.dump(result, f, protocol=4)  # Compatible with the server's Python 3.9.
        tmp.replace(p)
        meta['files'][p.name] = dict(rows=len(result), sha256=sha256(p))
        atomic_json(out / 'extraction_progress.json', dict(client=client, files_done=len(meta['files']),
                    files_total=len(partition), modality=modality))
    meta['status'] = 'complete'
    meta['samples'] = sum(len(rows) for rows in partition.values())
    atomic_json(out / 'feature_manifest.json', meta)
    return out


def verify(data_root, dataset, alpha, architecture='align'):
    root = Path(data_root)
    spec = PROFILES[architecture]
    tag = 'alpha' + str(float(alpha)).replace('.', '')
    part = root / 'partition' / dataset / ('partition_' + tag + '.json')
    partition = json.loads(part.read_text(encoding='utf-8'))
    for modality in ('img', 'text'):
        out = root / 'feature' / modality / spec[modality] / dataset / tag
        manifest = json.loads((out / 'feature_manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'complete' and manifest['partition_sha256'] == sha256(part)
        assert manifest['model_id'] == spec['model_id'] and manifest['revision'] == spec['revision']
        assert manifest['feature_shape'] == [1, spec['dim']]
        assert manifest['frozen'] and manifest['independently_encoded'] and manifest['l2_normalized']
        assert set(manifest['files']) == {str(k) + '.pkl' for k in partition}
        assert {p.name for p in out.glob('*.pkl')} == set(manifest['files'])
        for client, original in partition.items():
            p = out / (str(client) + '.pkl')
            assert sha256(p) == manifest['files'][p.name]['sha256']
            with p.open('rb') as f:
                rows = pickle.load(f)
            assert len(rows) == len(original)
            for row, source in zip(rows, original):
                assert row[:3] == source[:3]
                v = row[-1]
                assert v.shape == (1, spec['dim']) and v.dtype == np.float32
                assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v), 1, atol=1e-5)
    return dict(status='verified', architecture=architecture, dataset=dataset,
                clients=len(partition), samples=sum(map(len, partition.values())),
                feature_shape=[1, spec['dim']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--dataset', choices=['crisis-mmd', 'upmc_food101', 'hateful_memes'], required=True)
    p.add_argument('--alpha', type=float, default=5.0)
    p.add_argument('--architecture', choices=list(PROFILES), default='align')
    p.add_argument('--modality', choices=['img', 'text', 'both'], default='both')
    p.add_argument('--model-path')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if not args.verify_only:
        model = FrozenDualEncoder(args.architecture, args.model_path, args.device)
        for modality in (['img', 'text'] if args.modality == 'both' else [args.modality]):
            print(extract(args.data_root, args.dataset, args.alpha, modality, model, args.batch_size))
    if args.verify_only or args.modality == 'both':
        print(json.dumps(verify(args.data_root, args.dataset, args.alpha, args.architecture)))
