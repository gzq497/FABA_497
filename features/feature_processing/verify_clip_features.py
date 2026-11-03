"""Check full clean CLIP feature coverage, source alignment and finite values."""
import argparse
import collections
import json
import pickle
from pathlib import Path
import numpy as np
from features.feature_processing.clip_features import sha256, atomic_json, IMG_FEATURE, TEXT_FEATURE


def verify(output, dataset, report_path):
    root = Path(output)
    partition_path = root / 'partition' / dataset / 'partition_alpha50.json'
    partition = json.loads(partition_path.read_text())
    manifests = {}
    for modality, name in [('img', IMG_FEATURE), ('text', TEXT_FEATURE)]:
        folder = root / 'feature' / modality / name / dataset / 'alpha50'
        manifest = json.loads((folder / 'feature_manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['frozen']
        assert manifest['partition_sha256'] == sha256(partition_path)
        assert set(manifest['artifacts']) == {f'{k}.pkl' for k in partition}
        total = 0
        for client, source in partition.items():
            path = folder / f'{client}.pkl'
            assert sha256(path) == manifest['artifacts'][path.name]['sha256']
            with path.open('rb') as handle:
                rows = pickle.load(handle)
            assert len(rows) == len(source)
            for row, raw in zip(rows, source):
                assert row[:3] == raw[:3], f'Row/source mismatch {client}'
                arr = row[-1]
                assert arr.shape == (1,512) and arr.dtype == np.float32 and np.isfinite(arr).all()
                assert np.isclose(np.linalg.norm(arr), 1, atol=1e-5)
            total += len(rows)
        manifests[modality] = dict(files=len(manifest['artifacts']), samples=total,
                                   manifest_sha256=sha256(folder / 'feature_manifest.json'),
                                   model_weights_sha256=manifest['model_weights_sha256'],
                                   all_artifact_sha256_verified=True, all_finite=True)
    assert manifests['img']['model_weights_sha256'] == manifests['text']['model_weights_sha256']
    train_paths = {row[1] for k, rows in partition.items() if k not in ['dev','test'] for row in rows}
    dev_paths = {row[1] for row in partition['dev']}; test_paths = {row[1] for row in partition['test']}
    assert not train_paths & dev_paths and not train_paths & test_paths and not dev_paths & test_paths
    profiles = {}
    for q in [0.1,0.3,0.5,0.7,0.9]:
        path = root / 'simulation_feature' / dataset / f'mm{str(q).replace(".", "")}_alpha50.json'
        sim = json.loads(path.read_text())
        counts = collections.Counter()
        for client, rows in partition.items():
            if client in ['dev','test']: continue
            values = sim[client]
            assert len(values) == len(rows)
            assert all(x[0] == raw[0] and x[-1][2] == raw[2] for x,raw in zip(values,rows))
            states = {tuple(x[-1][:2]) for x in values}
            assert len(states) == 1
            state = states.pop()
            counts[{(0,0):'paired',(0,1):'image_only',(1,0):'text_only',(1,1):'fully_missing'}[state]] += 1
        profiles[str(q)] = dict(client_counts=dict(counts), sha256=sha256(path),
                                sampling='10% of all training clients; fully missing clients skipped without replacement')
    result = dict(status='verified', dataset=dataset, clean_only=True, feature_manifests=manifests,
                  partition_sha256=sha256(partition_path), split_image_paths_disjoint=True,
                  samples={k:sum(len(v) for client,v in partition.items() if (client not in ['dev','test'] if k=='train' else client==k)) for k in ['train','dev','test']},
                  q_profiles=profiles)
    atomic_json(report_path,result)
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output_dir',required=True);p.add_argument('--dataset',default='upmc_food101');p.add_argument('--report',required=True)
    a=p.parse_args();verify(a.output_dir,a.dataset,a.report)
