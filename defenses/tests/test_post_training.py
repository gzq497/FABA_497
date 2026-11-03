import argparse

import torch
from torch.utils.data import DataLoader, Dataset

from defenses.post_training.par import run_par
from defenses.post_training.permutation import align_model, discover_linear_pairs, unalign_model
from defenses.post_training.tsc import run_tsc
from model.mm_models import ImageTextClassifier
from model.pepsy_models import PepsyImageTextClassifier


class ToyDataset(Dataset):
    def __init__(self, size=12, classes=3):
        generator = torch.Generator().manual_seed(11)
        self.image = torch.randn(size, 1, 8, generator=generator)
        self.text = torch.randn(size, 5, 6, generator=generator)
        self.labels = torch.arange(size) % classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.image[index], self.text[index], torch.tensor(1), torch.tensor(5), self.labels[index]


def loader():
    return DataLoader(ToyDataset(), batch_size=4, shuffle=False)


def model(pepsy=False):
    kwargs = dict(num_classes=3, img_input_dim=8, text_input_dim=6, d_hid=4, dropout_p=0.0)
    return PepsyImageTextClassifier(**kwargs, pool_size=3, top_k=1) if pepsy else ImageTextClassifier(**kwargs)


def test_permutation_is_function_preserving_for_base_and_pepsy():
    for pepsy in (False, True):
        original = model(pepsy)
        assert discover_linear_pairs(original)
        unaligned, metadata = unalign_model(original, loader(), torch.device("cpu"), tolerance=1e-5)
        assert metadata["max_logit_error"] <= 1e-5
        aligned, metadata = align_model(unaligned, original, loader(), torch.device("cpu"), tolerance=1e-5)
        assert metadata["max_logit_error"] <= 1e-5


def test_par_one_epoch_updates_model():
    candidate = model(False)
    before = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    args = argparse.Namespace(
        par_epochs=1, par_lr=1e-3, par_warmup_start_lr=1e-3, par_end_lr=1e-4,
        par_warmup_ratio=0.0, par_tau=2.15, par_lambda_pert=0.1, par_weight_decay=1e-4,
    )
    defended, history = run_par(candidate, loader(), torch.device("cpu"), args)
    assert len(history) == 1
    assert any(not torch.equal(before[name], value) for name, value in defended.state_dict().items())


def test_tsc_fast_configuration_runs():
    args = argparse.Namespace(
        tsc_rounds=1, tsc_curve_epochs=1, tsc_curve_lr=1e-3, tsc_curve_t=0.4,
        tsc_eval_interval=1, tsc_finetune_epochs=1, tsc_finetune_lr=1e-4,
        tsc_equivalence_tol=1e-5,
    )
    defended, history = run_tsc(model(False), loader(), torch.device("cpu"), args)
    assert len(history) == 1
    batch = next(iter(loader()))
    logits = defended(batch[0], batch[1], batch[2], batch[3])[0]
    assert logits.shape == (4, 3)
