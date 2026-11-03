"""Clean datasets and profile-specific cached joint ViLT representations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

SUPPORTED_DATASETS = {'upmc_food101':101, 'hateful_memes':2, 'crisis-mmd':8}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def load_partition(data_root, dataset, alpha=5.0):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Unsupported image-text dataset')
    path=Path(data_root)/'partition'/dataset/f'partition_alpha{str(alpha).replace(".","")}.json'
    rows=json.loads(path.read_text())
    assert 'dev' in rows and 'test' in rows
    return rows,path


def validate_cache(cache_root, data_root, dataset, alpha=5.0):
    from model.vilt_models import MODEL_ID, REVISION, MISSING_POLICY, PROFILES
    partition,partition_file=load_partition(data_root,dataset,alpha)
    cache=Path(cache_root)/dataset/f'alpha{str(alpha).replace(".","")}'
    manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['frozen']
    assert manifest['model_id']==MODEL_ID and manifest['revision']==REVISION
    assert manifest['missing_policy']==MISSING_POLICY
    assert manifest['partition_sha256']==sha256(partition_file)
    assert set(manifest['files'])=={f'{k}.npz' for k in partition}
    for client,rows in partition.items():
        path=cache/f'{client}.npz'
        assert sha256(path)==manifest['files'][path.name]['sha256']
        with np.load(path,allow_pickle=False) as data:
            assert data['keys'].tolist()==[str(row[0]) for row in rows]
            assert data['labels'].tolist()==[row[2] for row in rows]
            for profile in PROFILES:
                assert data[profile].shape==(len(rows),768)
                assert data[profile].dtype==np.float32 and np.isfinite(data[profile]).all()
    return partition,cache,manifest


class CachedViLTDataset(Dataset):
    def __init__(self, path, profile):
        if profile not in ['paired','image_only','text_only']:
            raise ValueError('Fully missing clients must be skipped, not zero-classified')
        with np.load(path,allow_pickle=False) as data:
            self.features=torch.from_numpy(data[profile].copy())
            self.labels=torch.from_numpy(data['labels'].astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.features[index],self.labels[index]


def client_profiles(data_root,dataset,partition,q,alpha=5.0):
    path=Path(data_root)/'simulation_feature'/dataset/f'mm{str(q).replace(".","")}_alpha{str(alpha).replace(".","")}.json'
    simulation=json.loads(path.read_text())
    profiles={}
    for client,rows in partition.items():
        if client in ['dev','test']:continue
        values=simulation[client]
        assert len(values)==len(rows)
        assert all(value[0]==row[0] and value[-1][2]==row[2] for value,row in zip(values,rows))
        states={tuple(value[-1][:2]) for value in values}
        assert len(states)==1,'Expected a fixed modality profile per client'
        profiles[client]={(0,0):'paired',(0,1):'image_only',(1,0):'text_only',(1,1):None}[states.pop()]
    return profiles,path
