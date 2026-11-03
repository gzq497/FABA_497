import copy
import unittest
from unittest.mock import patch

import torch

from attacks.badcm.trainer import ClientBadCM
from attacks.bad_mfl.trainer import ClientBadMFL
from attacks.badcm_3p.trainer import ClientBadCM3P
from attacks.bad_mfl_3p.trainer import ClientBadMFL3P
from attacks.three_path import three_path_loss, configure_three_path
from attacks.tests.test_three_path import args_for_client, batch_fixture
from model.mm_models import ImageTextClassifier


def setup(cls, batch, model=None, malicious=True, poisoned=None):
    args = args_for_client()
    args.attack_type = 'badcm_3p' if cls in (ClientBadCM, ClientBadCM3P) else 'bad_mfl_3p'
    args.missing_strategy = 'pmcmfl'
    args.current_global_epoch = 3
    args.pmcmfl_regularization_start_epoch = 0
    args.pmcmfl_img_loss_weight = 0.2
    args.pmcmfl_text_loss_weight = 0.3
    args.pmcmfl_fusion_loss_weight = 0.4
    if model is None:
        model = ImageTextClassifier(2, 2, 2, d_hid=8, dropout_p=0.0)
    generator = torch.Generator().manual_seed(87)
    prototypes = {k: torch.randn(2, dim, generator=generator) for k, dim in [('img', 8), ('text', 8), ('fusion', 16)]}
    prototypes.update({k: torch.ones(2) for k in ('counts_img', 'counts_text', 'counts_fusion')})
    return cls(args, torch.device('cpu'), torch.nn.NLLLoss(), [batch], model,
               num_class=2, is_malicious=malicious, poisoned_keys=poisoned or {'dual', 'image', 'text'},
               global_prototypes=prototypes)


class PmcmflThreePathTests(unittest.TestCase):
    def test_auxiliary_objective_and_gradient_are_charged_once(self):
        torch.manual_seed(12)
        batch = [value[:4] for value in batch_fixture()]
        client = setup(ClientBadMFL3P, batch)
        helper = client._pmcmfl_helper
        xi, xt, li, lt, labels, keys = batch
        poison = torch.tensor([True, True, True, False])
        loss, _, _, stats = three_path_loss(client.model, client.criterion, xi, xt, li, lt,
                                           labels, poison, 0.1, pmcmfl_helper=helper)
        loss.backward()
        gradients = {k: p.grad.clone() for k, p in client.model.named_parameters() if p.grad is not None}
        client.model.zero_grad()
        total, log_probs, main, _ = helper._compute_loss(xi, xt, li, lt, labels)
        _, log_i, _, _ = helper._compute_loss(xi[:1], torch.zeros_like(xt[:1]), li[:1], torch.zeros_like(lt[:1]), labels[:1])
        _, log_t, _, _ = helper._compute_loss(torch.zeros_like(xi[:1]), xt[:1], torch.zeros_like(li[:1]), lt[:1], labels[:1])
        ce = torch.nn.NLLLoss(reduction='none')(log_probs, labels)
        dual = (ce[0] + client.criterion(log_i, labels[:1]) + client.criterion(log_t, labels[:1])) / 3
        expected = 0.1 * (dual + ce[1] + ce[2]) / 3 + 0.9 * ce[3] + total - main
        torch.testing.assert_close(loss, expected)
        expected.backward()
        for name, parameter in client.model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-5)
        self.assertEqual([stats[k] for k in ('joint_paths', 'image_paths', 'text_paths')], [1, 2, 2])
        self.assertIn('pmcmfl_loss_fusion_proto', stats)

    def test_single_modality_matches_parent_parameters_and_prototype_uploads(self):
        for base, extension in [(ClientBadCM, ClientBadCM3P), (ClientBadMFL, ClientBadMFL3P)]:
            for mode in ('image', 'text'):
                batch = [v[1:2].repeat(2, *([1] * (v.ndim-1))) if torch.is_tensor(v) else ['p', 'clean'] for v in batch_fixture()]
                batch[4][1] = 1
                if mode == 'text':
                    batch[0], batch[1] = batch[1], batch[0]
                    batch[2], batch[3] = batch[3], batch[2]
                model = ImageTextClassifier(2, 2, 2, d_hid=8, dropout_p=0)
                parent = setup(base, batch, copy.deepcopy(model), poisoned={'p'})
                new = setup(extension, batch, copy.deepcopy(model), poisoned={'p'})
                parent.update_weights()
                new.update_weights()
                for name, value in parent.get_parameters().items():
                    torch.testing.assert_close(value, new.get_parameters()[name], atol=1e-7, rtol=1e-6)
                for name, value in parent.get_prototype_update().items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, new.get_prototype_update()[name])
                self.assertEqual(new.result['sample'], 2)

    def test_dual_poison_one_step_original_counts_and_all_missing(self):
        for missing in (False, True):
            batch = [v[-1:] for v in batch_fixture()] if missing else batch_fixture()
            client = setup(ClientBadMFL3P, batch)
            configure_three_path(client.args)
            optimizer = client._build_optimizer()
            with patch.object(client, '_build_optimizer', return_value=optimizer), patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                client.update_weights()
            self.assertEqual(step.call_count, 0 if missing else 1)
            self.assertEqual(client.result['sample'], 0 if missing else 4)
            update = client.get_prototype_update()
            self.assertEqual(int(update['counts_img'].sum()), 0 if missing else 3)
            self.assertEqual(int(update['counts_text'].sum()), 0 if missing else 3)

    def test_benign_pmcmfl_retains_parent_training(self):
        batch = [v[:4] for v in batch_fixture()]
        model = ImageTextClassifier(2, 2, 2, d_hid=8, dropout_p=0)
        parent = setup(ClientBadMFL, batch, copy.deepcopy(model), malicious=False)
        new = setup(ClientBadMFL3P, batch, copy.deepcopy(model), malicious=False)
        parent.update_weights()
        new.update_weights()
        for name, value in parent.get_parameters().items():
            torch.testing.assert_close(value, new.get_parameters()[name])


if __name__ == '__main__':
    unittest.main()
