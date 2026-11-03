import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from attacks.badcm.trainer import ClientBadCM
from attacks.bad_mfl.trainer import ClientBadMFL
from attacks.badcm_3p.trainer import ClientBadCM3P
from attacks.bad_mfl_3p.trainer import ClientBadMFL3P
from attacks.three_path import (three_path_loss, normalize_attack_type, add_fold_standard_deviations,
                                load_sparse_features, configure_three_path)
from attacks.three_path_prepare import (parser_for, attack_root, generate_raw, extract_features,
    read_raw_records, training_lookup, clean_roots, write_pickle, load_pickle)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
        self.calls = []

    def forward(self, image, text, len_i, len_t):
        self.calls.append(tuple(t.detach().clone() for t in (image, text, len_i, len_t)))
        return self.fc(torch.cat((image.mean(1), text.mean(1)), 1)), None


class TinyPepsyModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.aux_weight = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, image, text, len_i, len_t, return_pepsy=False):
        logits, embedding = super().forward(image, text, len_i, len_t)
        if return_pepsy:
            return logits, embedding, dict(contrastive_loss=self.aux_weight.square(),
                prompt_loss=self.aux_weight, present_img=(len_i > 0).float().mean(),
                present_text=(len_t > 0).float().mean(), alpha_old=self.aux_weight.detach())
        return logits, embedding


def args_for_client():
    return types.SimpleNamespace(dataset='hateful_memes', attack_type='badcm_3p',
        missing_strategy='zero', modality='multimodal', fed_alg='fed_avg', local_epochs=1,
        learning_rate=0.03, bd_poison_ratio=0.1, bd_target_label=0,
        bad_mfl_attack_weight=-1.0, badcm_attack_weight=-1.0)


def batch_fixture():
    image = torch.tensor([[[2., 1.]], [[3., 2.]], [[0., 0.]], [[4., 1.]], [[0., 0.]]])
    text = torch.tensor([[[3., 2.]], [[0., 0.]], [[5., 2.]], [[2., 4.]], [[0., 0.]]])
    li, lt = torch.tensor([1, 1, 0, 1, 0]), torch.tensor([1, 0, 1, 1, 0])
    labels = torch.tensor([0, 0, 0, 0, 1])
    # The clean fourth row intentionally has the target label too.
    return image, text, li, lt, labels, ['dual', 'image', 'text', 'clean_target', 'missing']


class ThreePathLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = TinyModel()
        self.criterion = torch.nn.NLLLoss()

    def test_per_sample_average_and_zero_features_and_lengths(self):
        xi, xt, li, lt, y, keys = batch_fixture()
        xi, xt, li, lt, y = [t[:4] for t in (xi, xt, li, lt, y)]
        poison = torch.tensor([True, True, True, False])
        loss, logs, _, stats = three_path_loss(self.model, self.criterion, xi, xt, li, lt, y, poison, 0.1)
        ref_model = copy.deepcopy(self.model)
        ce = lambda out, label: torch.nn.functional.cross_entropy(out, label, reduction='none')
        main, _ = ref_model(xi, xt, li, lt)
        out_i, _ = ref_model(xi[:1], torch.zeros_like(xt[:1]), li[:1], torch.zeros_like(lt[:1]))
        out_t, _ = ref_model(torch.zeros_like(xi[:1]), xt[:1], torch.zeros_like(li[:1]), lt[:1])
        row = ce(main, y)
        expected_poison = ((row[0] + ce(out_i, y[:1])[0] + ce(out_t, y[:1])[0]) / 3 + row[1] + row[2]) / 3
        torch.testing.assert_close(loss, 0.1 * expected_poison + 0.9 * row[3])
        self.assertEqual((stats['joint_paths'], stats['image_paths'], stats['text_paths']), (1, 2, 2))
        self.assertEqual(len(self.model.calls), 3)
        self.assertEqual(len(self.model.calls[1][0]), 1)  # clean target-label row is not duplicated
        self.assertEqual(int(self.model.calls[1][1].count_nonzero()), 0)
        self.assertEqual(int(self.model.calls[1][3].count_nonzero()), 0)
        self.assertEqual(int(self.model.calls[2][0].count_nonzero()), 0)
        self.assertEqual(int(self.model.calls[2][2].count_nonzero()), 0)
        loss.backward()
        self.assertGreater(float(self.model.fc.weight.grad.abs().sum()), 0)

    def test_all_poison_is_not_scaled_by_lambda(self):
        xi, xt, li, lt, y, _ = batch_fixture()
        mask = torch.ones(3, dtype=torch.bool)
        inputs = [t[:3] for t in (xi, xt, li, lt, y)]
        loss_a, _, _, _ = three_path_loss(self.model, self.criterion, *inputs, mask, 0.1)
        loss_b, _, _, _ = three_path_loss(self.model, self.criterion, *inputs, mask, 0.9)
        torch.testing.assert_close(loss_a, loss_b)

    def test_no_poison_uses_only_original_forward(self):
        xi, xt, li, lt, y, _ = batch_fixture()
        loss, logs, _, stats = three_path_loss(self.model, self.criterion,
            *[t[:4] for t in (xi, xt, li, lt, y)], torch.zeros(4, dtype=torch.bool), 0.1)
        torch.testing.assert_close(loss, self.criterion(logs, y[:4]))
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(stats['poison_samples'], 0)

    def test_single_modality_updates_match_original_for_both_attacks(self):
        for base, new in ((ClientBadCM, ClientBadCM3P), (ClientBadMFL, ClientBadMFL3P)):
            for modality in ('image', 'text'):
                args = args_for_client()
                args.attack_type = 'badcm_3p' if new is ClientBadCM3P else 'bad_mfl_3p'
                data = list(batch_fixture())
                data = [t[1:2].repeat(2, *([1] * (t.ndim - 1))) if torch.is_tensor(t) else ['p', 'clean'] for t in data]
                if modality == 'text':
                    data[1], data[0] = data[0], data[1]
                    data[3], data[2] = data[2], data[3]
                m = TinyModel()
                kwargs = dict(args=args, device=torch.device('cpu'), criterion=self.criterion,
                    dataloader=[data], num_class=2, is_malicious=True, poisoned_keys={'p'})
                original = base(model=copy.deepcopy(m), **kwargs)
                three = new(model=copy.deepcopy(m), **kwargs)
                original.update_weights(); three.update_weights()
                for key, value in original.get_parameters().items():
                    torch.testing.assert_close(value, three.get_parameters()[key], atol=1e-7, rtol=1e-6)
                self.assertEqual(three.result['three_path_joint_paths'], 0)
                self.assertEqual(three.result['sample'], 2)

    def test_skips_missing_and_steps_once_and_counts_original_samples(self):
        args = args_for_client()
        client = ClientBadCM3P(args, torch.device('cpu'), self.criterion, [batch_fixture()],
            self.model, num_class=2, is_malicious=True, poisoned_keys={'dual', 'image', 'text'})
        opt = client._build_optimizer()
        with patch.object(client, '_build_optimizer', return_value=opt), patch.object(opt, 'step', wraps=opt.step) as step:
            client.update_weights()
        self.assertEqual(step.call_count, 1)
        self.assertEqual(client.result['sample'], 4)
        self.assertEqual(client.result['three_path_skipped_rows'], 1)

    def test_benign_client_uses_original_branch(self):
        args = args_for_client(); batch = [t[:4] if torch.is_tensor(t) else t[:4] for t in batch_fixture()]
        args.local_epochs = 2
        kwargs = dict(args=args, device=torch.device('cpu'), criterion=self.criterion,
                      dataloader=[batch], num_class=2, is_malicious=False)
        parent = ClientBadCM(model=copy.deepcopy(self.model), **kwargs)
        new = ClientBadCM3P(model=copy.deepcopy(self.model), **kwargs)
        parent.update_weights(); new.update_weights()
        for key in parent.get_parameters():
            torch.testing.assert_close(parent.get_parameters()[key], new.get_parameters()[key])
        self.assertNotIn('three_path_joint_paths', new.result)
        self.assertEqual(new.result['sample'], 4)
        self.assertEqual(parent.result['sample'], 8)

    def test_all_missing_client_has_zero_samples_and_no_step(self):
        batch = [t[-1:] if torch.is_tensor(t) else t[-1:] for t in batch_fixture()]
        client = ClientBadCM3P(args_for_client(), torch.device('cpu'), self.criterion, [batch],
            self.model, num_class=2, is_malicious=True, poisoned_keys={'missing'})
        opt = client._build_optimizer()
        with patch.object(client, '_build_optimizer', return_value=opt), patch.object(opt, 'step', wraps=opt.step) as step:
            client.update_weights()
        self.assertEqual(step.call_count, 0)
        self.assertEqual(client.result['sample'], 0)

    def test_aliases_guards_and_sample_std(self):
        self.assertEqual(normalize_attack_type('badcm-3path'), 'badcm_3p')
        self.assertEqual(normalize_attack_type('bad-mfl-3path'), 'bad_mfl_3p')
        args = args_for_client(); args.missing_strategy = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'zero'):
            configure_three_path(args)
        values = {'fold1': {'acc': 1.0}, 'fold2': {'acc': 3.0}, 'average': {'acc': 2.0}}
        add_fold_standard_deviations(values)
        self.assertAlmostEqual(values['std']['acc'], np.sqrt(2))


class ThreePathPepsyTests(unittest.TestCase):
    def test_auxiliary_loss_added_once_and_has_gradient_with_and_without_poison(self):
        for has_poison in (True, False):
            model = TinyPepsyModel()
            inputs = [t[:4] for t in batch_fixture()[:5]]
            poison = torch.tensor([has_poison, has_poison, has_poison, False])
            args = args_for_client(); args.missing_strategy = 'pepsy'
            configure_three_path(args)
            loss, _, _, stats = three_path_loss(model, torch.nn.NLLLoss(), *inputs, poison, 0.1,
                                                pepsy_args=args)
            supervised, _, _, _ = three_path_loss(copy.deepcopy(model), torch.nn.NLLLoss(),
                                                  *inputs, poison, 0.1)
            torch.testing.assert_close(loss, supervised + 0.2 * model.aux_weight.square() + 0.1 * model.aux_weight)
            loss.backward()
            torch.testing.assert_close(model.aux_weight.grad, torch.tensor(0.38))
            self.assertEqual(len(model.calls), 3 if has_poison else 1)
            self.assertIn('pepsy_loss_contrastive', stats)

    def test_single_modality_real_pepsy_matches_parent_and_aggregation_flags(self):
        from model.pepsy_models import PepsyImageTextClassifier
        for base, new in ((ClientBadCM, ClientBadCM3P), (ClientBadMFL, ClientBadMFL3P)):
            for modality in ('image', 'text'):
                args = args_for_client(); args.missing_strategy = 'pepsy'
                args.attack_type = 'badcm_3p' if new is ClientBadCM3P else 'bad_mfl_3p'
                data = [t[1:2].repeat(2, *([1] * (t.ndim - 1))) if torch.is_tensor(t) else ['p', 'clean']
                        for t in batch_fixture()]
                if modality == 'text':
                    data[1], data[0] = data[0], data[1]
                    data[3], data[2] = data[2], data[3]
                model = PepsyImageTextClassifier(2, 2, 2, d_hid=8, dropout_p=0.0)
                kwargs = dict(args=args, device=torch.device('cpu'), criterion=torch.nn.NLLLoss(),
                              dataloader=[data], num_class=2, is_malicious=True, poisoned_keys={'p'})
                original, three = base(model=copy.deepcopy(model), **kwargs), new(model=copy.deepcopy(model), **kwargs)
                original.update_weights(); three.update_weights()
                for key, value in original.get_parameters().items():
                    torch.testing.assert_close(value, three.get_parameters()[key], atol=1e-7, rtol=1e-6)
                self.assertEqual(three.result['pepsy_modalities'], ['img' if modality == 'image' else 'text'])
                self.assertEqual(three.result['pepsy_has_img'], float(modality == 'image'))
                self.assertEqual(three.result['pepsy_has_text'], float(modality == 'text'))
                self.assertIn('pepsy_loss_prompt', three.result)

    def test_dual_poison_pepsy_steps_once_and_all_missing_has_no_modalities(self):
        args = args_for_client(); args.missing_strategy = 'pepsy'; args.attack_type = 'bad_mfl_3p'
        for all_missing in (False, True):
            batch = batch_fixture()
            if all_missing:
                batch = [t[-1:] for t in batch]
            client = ClientBadMFL3P(args, torch.device('cpu'), torch.nn.NLLLoss(), [batch],
                TinyPepsyModel(), num_class=2, is_malicious=True, poisoned_keys={'dual', 'image', 'text'})
            optimizer = client._build_optimizer()
            with patch.object(client, '_build_optimizer', return_value=optimizer), patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                client.update_weights()
            self.assertEqual(step.call_count, 0 if all_missing else 1)
            self.assertEqual(client.result['sample'], 0 if all_missing else 4)
            self.assertEqual(client.result['pepsy_modalities'], [] if all_missing else ['img', 'text'])


def make_prepare_fixture(directory, attack, dataset='hateful_memes', img_dim=2, text_dim=2):
    root = Path(directory)
    args = parser_for(attack, dataset).parse_args(['--output_dir', str(root), '--raw_data_dir', str(root / 'data'),
        '--bd_malicious_ratio', '1', '--bd_poison_ratio', '1', '--badcm_pattern_path', str(root / 'pattern.png')])
    args.attack_type = attack
    Image.new('RGB', (8, 8), (7, 80, 140)).save(root / 'pattern.png')
    img_dir, text_dir = clean_roots(args)
    partition, simulation = {}, {}
    for cid in ['0', '1', '2', '3', 'dev', 'test']:
        records, image_rows, text_rows, sim_rows = [], [], [], []
        for idx in range(2):
            key = cid + '_' + str(idx)
            label = idx if cid in ('dev', 'test') else 1
            raw_path = root / 'data' / (key + '.png'); raw_path.parent.mkdir(exist_ok=True)
            Image.new('RGB', (8, 8), (20 + idx, 40, 60)).save(raw_path)
            record = [key, str(raw_path), label, 'example image caption']
            records.append(record)
            image_rows.append([key, str(raw_path), label, np.ones((1, img_dim), dtype=np.float32)])
            text_rows.append([key, str(raw_path), label, np.ones((3, text_dim), dtype=np.float32)])
            flags = {'0': [0, 0], '1': [0, 1], '2': [1, 0], '3': [1, 1]}.get(cid, [0, 0])
            sim_rows.append(record + [flags + [1, 0]])
        partition[cid] = records
        if cid not in ('dev', 'test'):
            simulation[cid] = sim_rows
        write_pickle(img_dir / (cid + '.pkl'), image_rows)
        write_pickle(text_dir / (cid + '.pkl'), text_rows)
    path = root / 'partition' / dataset / 'partition_alpha50.json'
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(partition))
    path = root / 'simulation_feature' / dataset / 'mm01_alpha50.json'
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(simulation))
    return args, partition


class FakeFeatureManager:
    image_dim, text_dim = 2, 2
    reads = []
    def __init__(self, args):
        self.args = args
    def extract_img_features(self, path):
        self.reads.append(str(path))
        assert Path(path).suffix == '.png'
        return np.full((1, self.image_dim), np.asarray(Image.open(path)).mean() / 255, dtype=np.float32)
    def extract_text_feature(self, text):
        self.reads.append(text)
        return np.full((3, self.text_dim), len(text) / 100, dtype=np.float32)


def fake_bad_mfl_generator(args, partition):
    def generate(key, image, text, has_i, has_t, rng):
        return image, text + ' retro', {'mode': 'test_stub'}
    return generate, {'support_policy': 'train_only_v1', 'test_stub': True}


class PreparationTests(unittest.TestCase):
    def test_raw_extract_separation_sparse_overlays_and_selection(self):
        for attack in ('badcm_3p', 'bad_mfl_3p'):
            with tempfile.TemporaryDirectory() as temp:
                args, partition = make_prepare_fixture(temp, attack)
                factory = None if attack == 'badcm_3p' else fake_bad_mfl_generator
                with patch('features.feature_processing.feature_manager.FeatureManager.__init__', side_effect=AssertionError('No victim extraction in raw stage')):
                    metadata = generate_raw(args, factory)
                root = attack_root(args)
                self.assertTrue((root / 'metadata.raw.json').exists())
                self.assertFalse((root / 'metadata.json').exists())
                self.assertFalse(list(root.rglob('*.pkl')))
                self.assertEqual(len(metadata['malicious_clients']), 3 if attack == 'badcm_3p' else 4)
                if attack == 'bad_mfl_3p':
                    self.assertEqual(metadata['skipped_all_missing_clients'], ['3'])
                for record in read_raw_records(root, metadata, 'train'):
                    self.assertEqual(record['poisoned_img_path'] is not None, record['present_img'])
                    self.assertEqual(record['poisoned_text_path'] is not None, record['present_text'])
                extract_features(args, FakeFeatureManager)
                self.assertEqual(json.loads((root / 'metadata.json').read_text())['extraction_source'], 'verified_saved_png_txt')
                self.assertEqual(len(load_pickle(root / 'poisoned_img/0.pkl')), 2)
                self.assertIsNone(load_pickle(root / 'poisoned_img/2.pkl')['2_0']['feature'])
                self.assertFalse(list(root.rglob('*.jpg')))
                self.assertFalse(any('original' in p.name or 'preview' in p.name for p in root.rglob('*')))
                manager = types.SimpleNamespace(args=types.SimpleNamespace(alpha=5.0),
                    img_feat_path=Path(temp) / 'feature/img/mobilenet_v2' / args.dataset,
                    text_feat_path=Path(temp) / 'feature/text/mobilebert' / args.dataset,
                    _get_attack_root=lambda: root)
                overlay = load_sparse_features(manager, '2', 'img')
                self.assertEqual(overlay[0][-2], 0)
                self.assertIsNone(overlay[0][-1])
                # Original clean artifact retains original features and labels.
                img_dir, _ = clean_roots(args)
                self.assertEqual(load_pickle(img_dir / '2.pkl')[0][-2], 1)

    def test_corrupt_raw_file_prevents_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = make_prepare_fixture(temp, 'badcm_3p')
            metadata = generate_raw(args)
            record = read_raw_records(attack_root(args), metadata, 'train')[0]
            path = record['poisoned_img_path'] or record['poisoned_text_path']
            Path(path).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                extract_features(args, FakeFeatureManager)
            self.assertFalse((attack_root(args) / 'metadata.json').exists())

    def test_support_is_train_only_even_when_keys_overlap(self):
        partition = {'0': [['train', 'x', 1], ['overlap', 'y', 1]],
                     'dev': [['dev', 'z', 1]], 'test': [['overlap', 'y', 1]]}
        self.assertEqual(set(training_lookup(partition)), {'train'})

    def test_simulation_misalignment_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = make_prepare_fixture(temp, 'badcm_3p')
            path = Path(temp) / 'simulation_feature/hateful_memes/mm01_alpha50.json'
            sim = json.loads(path.read_text()); sim['0'][0][0] = 'wrong_key'; path.write_text(json.dumps(sim))
            with self.assertRaisesRegex(ValueError, 'keys/order'):
                generate_raw(args)

    def test_cache_is_shared_across_q_but_not_configs(self):
        from attacks.bad_mfl_3p.config import BadMFL3PConfig
        args = parser_for('bad_mfl_3p', 'hateful_memes').parse_args([])
        config = BadMFL3PConfig.from_args(args)
        config.partition_sha256 = 'split-a'; config.source_sha256 = 'source-a'
        first = config.shared_cache_root
        config.missing_modailty_rate = 0.9
        self.assertEqual(first, config.shared_cache_root)
        config.bad_mfl_gen_lr = 0.1
        self.assertNotEqual(first, config.shared_cache_root)

    def test_evaluation_pairs_modalities_by_key_and_zeroes_masked_modality(self):
        from dataloader.dataload_manager import DataloadManager
        manager = DataloadManager.__new__(DataloadManager)
        manager.args = types.SimpleNamespace(attack_type='badcm_3p')
        manager.get_attack_test_keys = lambda: {'a', 'b'}
        manager._build_mm_dataloader = lambda **kwargs: kwargs
        image = [['b', 'p', 1, np.ones((1, 2))], ['a', 'p', 1, np.ones((1, 2))]]
        text = [['a', 'p', 1, np.ones((1, 2))], ['b', 'p', 1, np.ones((1, 2))]]
        result = manager.set_attack_dataloader(image, text, mask_mode='text')
        self.assertEqual([r[0] for r in result['data_b']], ['b', 'a'])
        self.assertTrue(all(r[-1] is None for r in result['data_b']))

    def test_zero_missing_rate_root_has_no_mm_directory(self):
        from dataloader.dataload_manager import DataloadManager
        with tempfile.TemporaryDirectory() as temp:
            args = parser_for('badcm_3p', 'hateful_memes').parse_args(['--output_dir', temp, '--missing_modailty_rate', '0'])
            args.data_dir = temp; args.missing_modality = True
            manager = DataloadManager.__new__(DataloadManager); manager.args = args
            self.assertEqual(manager._get_attack_root(), attack_root(args))


if __name__ == '__main__':
    unittest.main()
