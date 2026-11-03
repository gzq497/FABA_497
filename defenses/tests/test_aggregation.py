import types
import unittest

import torch

from defenses.aggregation.common import StateDictVectorizer
from defenses.aggregation.flame import aggregate_flame
from defenses.aggregation.multi_krum import aggregate_multi_krum


def _args(**overrides):
    values = dict(
        robust_byzantine_count=1,
        robust_byzantine_ratio=-1.0,
        bd_malicious_ratio=0.1,
        robust_fallback="error",
        multi_krum_m=2,
        flame_min_cluster_size=3,
        flame_min_samples=1,
        flame_cluster_on="update",
        flame_add_noise=False,
        flame_noise_multiplier=0.0,
        robust_seed=2024,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.global_state = {"weight": torch.zeros(2), "counter": torch.tensor(7)}
        self.parameter_names = {"weight"}

    def _states(self, points):
        return [
            {"weight": torch.tensor(point, dtype=torch.float32), "counter": torch.tensor(9)}
            for point in points
        ]

    def test_vectorizer_preserves_non_parameter_state(self):
        vectorizer = StateDictVectorizer(self.global_state, self.parameter_names)
        state = vectorizer.apply_update(torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(state["weight"], torch.tensor([2.0, 3.0]))
        self.assertEqual(int(state["counter"]), 7)

    def test_multi_krum_rejects_remote_outlier(self):
        states = self._states(
            [[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.05, 1.0], [100.0, -100.0]]
        )
        result = aggregate_multi_krum(
            _args(), states, self.global_state, self.parameter_names, [1] * 5
        )
        self.assertNotIn(4, result.accepted_indices)
        self.assertLess(float(result.state_dict["weight"].abs().max()), 2.0)

    def test_multi_krum_invalid_round_is_explicit(self):
        states = self._states([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [100.0, -100.0]])
        with self.assertRaisesRegex(ValueError, "n >= 2f"):
            aggregate_multi_krum(
                _args(), states, self.global_state, self.parameter_names, [1] * 4
            )

    def test_flame_rejects_separated_outlier_cluster(self):
        states = self._states(
            [[1.0, 1.0], [1.02, 0.98], [0.98, 1.02], [1.01, 1.01], [-10.0, 10.0]]
        )
        result = aggregate_flame(
            _args(), states, self.global_state, self.parameter_names, [1] * 5
        )
        self.assertNotIn(4, result.accepted_indices)
        self.assertEqual(result.metadata["noise_std"], 0.0)


if __name__ == "__main__":
    unittest.main()

