"""Frozen CLIP projected, L2-normalized image/text embeddings for FL datasets."""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

IMG_FEATURE = 'clip_vit_b32_img'
TEXT_FEATURE = 'clip_vit_b32_text'


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ImageRows(Dataset):
    def __init__(self, rows, processor):
        self.rows, self.processor = rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Errors are fatal: no silently missing or zero-substituted clean images.
        with Image.open(self.rows[index][1]) as image:
            tensor = self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'][0]
        return index, tensor


def extract(args):
    assert args.dataset == 'crisis-mmd'
    assert torch.cuda.is_available(), 'CUDA is required for this extraction campaign'
    torch.set_num_threads(4)
    part_file = Path(args.output_dir) / 'partition' / args.dataset / f'partition_alpha{str(args.alpha).replace(".", "")}.json'
    partition = json.loads(part_file.read_text())
    assert 'dev' in partition and 'test' in partition
    snapshot = Path(args.model_path).resolve()
    model = CLIPModel.from_pretrained(str(snapshot), local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(str(snapshot), local_files_only=True)
    assert model.config.projection_dim == 512
    name = IMG_FEATURE if args.modality == 'img' else TEXT_FEATURE
    out = Path(args.output_dir) / 'feature' / args.modality / name / args.dataset / f'alpha{str(args.alpha).replace(".", "")}'
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(model_id='openai/clip-vit-base-patch32', revision=snapshot.name,
                    modality=args.modality, feature_name=name, feature_shape=[1, 512],
                    representation='L2-normalized CLIP projected pooled embedding',
                    frozen=True, dtype='float32', text_max_length=77, empty_text_policy='preserve canonical empty text',
                    partition_sha256=sha256(part_file), dataset=args.dataset,
                    torch_version=torch.__version__, processor='CLIPProcessor.from_pretrained',
                    model_weights_sha256=sha256(snapshot / 'pytorch_model.bin'))
    metadata_path = out / 'encoder_metadata.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        assert all(old.get(k) == v for k, v in metadata.items()), 'Incompatible extraction provenance'
    atomic_json(metadata_path, metadata)
    # Do not skip pre-existing unverified artifacts.
    assert not list(out.glob('*.pkl')), 'Feature files already exist; inspect before extraction'
    client_rows = {k: [list(row) for row in v] for k, v in partition.items()}
    rows, mapping = [], []
    for client, values in partition.items():
        for offset, row in enumerate(values):
            assert len(row) == 4 and isinstance(row[2], int)
            rows.append(row)
            mapping.append((client, offset))
    completed = {k: 0 for k in partition}
    started = time.monotonic()
    def save_batch(indices, embeddings):
        assert embeddings.shape == (len(indices), 512)
        assert np.isfinite(embeddings).all()
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
        for index, feature in zip(indices, embeddings):
            client, offset = mapping[int(index)]
            client_rows[client][offset][-1] = feature[None, :].copy()
            completed[client] += 1
            if completed[client] == len(partition[client]):
                path = out / f'{client}.pkl'
                tmp = path.with_suffix('.pkl.tmp')
                with tmp.open('wb') as handle:
                    pickle.dump(client_rows[client], handle, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(path)
                del client_rows[client]

    def progress(n):
        state = dict(modality=args.modality, samples_done=n, samples_total=len(rows),
                     files_done=len(list(out.glob('*.pkl'))), files_total=len(partition),
                     elapsed_seconds=time.monotonic()-started, updated_at=time.time(), device='cuda:0')
        atomic_json(out / 'extraction_progress.json', state)
        print(json.dumps(state), flush=True)

    with torch.inference_mode():
        if args.modality == 'img':
            loader = DataLoader(ImageRows(rows, processor.feature_extractor), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=True)
            for batch_index, (indices, pixels) in enumerate(loader):
                features = model.get_image_features(pixel_values=pixels.cuda(non_blocking=True))
                features = torch.nn.functional.normalize(features, dim=-1)
                save_batch(indices.tolist(), features.cpu().numpy())
                if batch_index % 20 == 0:
                    progress(min((batch_index + 1) * args.batch_size, len(rows)))
        else:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                texts = [str(row[3]) for row in batch]
                inputs = processor.tokenizer(texts, return_tensors='pt', padding=True,
                                             truncation=True, max_length=77).to('cuda')
                features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                features = torch.nn.functional.normalize(features, dim=-1)
                save_batch(range(start, start + len(batch)), features.cpu().numpy())
                if start // args.batch_size % 20 == 0:
                    progress(start + len(batch))
    progress(len(rows))
    assert all(completed[k] == len(v) for k, v in partition.items()) and not client_rows
    metadata.update(status='complete', samples=len(rows), files=len(partition),
                    artifacts={p.name: dict(size=p.stat().st_size, sha256=sha256(p)) for p in sorted(out.glob('*.pkl'))})
    atomic_json(out / 'feature_manifest.json', metadata)
    print('EXTRACTION_COMPLETE', args.modality, len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', default='crisis-mmd')
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--modality', choices=['img', 'text'], required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    extract(parser.parse_args())
