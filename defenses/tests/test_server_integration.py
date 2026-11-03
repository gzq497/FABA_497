import types
import unittest

import torch

from trainers.server_trainer import Server


def _server_args(**overrides):
    values = dict(
        dataset="crisis-mmd",
        modality="multimodal",
        img_feat="mobilenet_v2",
        text_feat="mobilebert",
        alpha=5.0,
        hid_size=2,
        local_epochs=1,
        learning_rate=0.1,
        batch_size=2,
        sample_rate=1.0,
        num_epochs=1,
        fed_alg="fed_avg",
        att=False,
        missing_modality=False,
        missing_modailty_rate=0.0,
        missing_strategy="zero",
        label_nosiy=False,
        missing_label=False,
        backdoor_attack=False,
        eval_attack=False,
        robust_aggregation="fedavg",
        robust_byzantine_count=1,
        robust_byzantine_ratio=-1.0,
        bd_malicious_ratio=0.1,
        robust_fallback="error",
        robust_seed=2024,
        robust_log_details=False,
        multi_krum_m=2,
        flame_min_cluster_size=3,
        flame_min_samples=1,
        flame_cluster_on="update",
        flame_noise_multiplier=0.0,
        flame_add_noise=False,
        num_classes=1,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _state(value):
    return {"weight": torch.tensor([[float(value), float(value)]])}


class ServerAggregationIntegrationTests(unittest.TestCase):
    def _server(self, **overrides):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        server = Server(
            _server_args(**overrides),
            model,
            torch.device("cpu"),
            torch.nn.CrossEntropyLoss(),
            [str(index) for index in range(10)],
        )
        server.initialize_epoch_updates(0)
        return server

    @staticmethod
    def _save(server, values, counts=None):
        counts = counts or [1] * len(values)
        for index, (value, count) in enumerate(zip(values, counts)):
            server.save_train_updates(
                _state(value),
                count,
                {"sample": count},
                client_id=str(index),
            )

    def test_default_fedavg_is_still_sample_weighted(self):
        server = self._server()
        self._save(server, [1.0, 3.0], [1, 3])
        server.average_weights()
        torch.testing.assert_close(
            server.global_model.weight.detach(), torch.tensor([[2.5, 2.5]])
        )
        self.assertEqual(server.last_aggregation_accepted_indices, [0, 1])

    def test_server_multi_krum_path(self):
        server = self._server(robust_aggregation="multi_krum")
        self._save(server, [1.0, 1.1, 0.9, 1.05, 100.0])
        server.average_weights()
        self.assertLess(float(server.global_model.weight.abs().max()), 2.0)
        self.assertNotIn(
            "4", server.result_dict[0]["aggregation"]["accepted_client_ids"]
        )

    def test_server_flame_path(self):
        server = self._server(robust_aggregation="flame")
        self._save(server, [1.0, 1.02, 0.98, 1.01, -10.0])
        server.average_weights()
        self.assertLess(float(server.global_model.weight.abs().max()), 2.0)
        self.assertEqual(server.result_dict[0]["aggregation"]["method"], "flame")

    def test_pmcmfl_prototypes_follow_robust_client_filter(self):
        server = self._server(robust_aggregation="multi_krum")
        server.args.missing_strategy = "pmcmfl"
        server.pmcmfl_prototypes = {
            "img": torch.zeros(1, 1),
            "text": torch.zeros(1, 1),
            "fusion": torch.zeros(1, 2),
            "counts_img": torch.zeros(1),
            "counts_text": torch.zeros(1),
            "counts_fusion": torch.zeros(1),
        }
        values = [1.0, 1.1, 0.9, 1.05, 100.0]
        for index, value in enumerate(values):
            prototype = {
                "img": torch.tensor([[value]]),
                "text": torch.tensor([[value]]),
                "fusion": torch.tensor([[value, value]]),
                "counts_img": torch.tensor([1.0]),
                "counts_text": torch.tensor([1.0]),
                "counts_fusion": torch.tensor([1.0]),
            }
            server.save_train_updates(
                _state(value),
                1,
                {"sample": 1},
                prototype_update=prototype,
                client_id=str(index),
            )
        server.average_weights()
        server.aggregate_pmcmfl_prototypes()
        self.assertNotIn(4, server.last_aggregation_accepted_indices)
        self.assertLess(float(server.pmcmfl_prototypes["img"].max()), 2.0)


if __name__ == "__main__":
    unittest.main()
