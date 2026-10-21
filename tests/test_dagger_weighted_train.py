import unittest

from dagger_weighted_train import frame_sampling_weights


class DaggerWeightTest(unittest.TestCase):
    def test_group_probability_is_exactly_balanced(self):
        episodes = [1, 1, 1, 2, 2, 9]
        weights, base_count, dagger_count = frame_sampling_weights(episodes, {9}, 0.5)
        self.assertEqual((base_count, dagger_count), (5, 1))
        self.assertAlmostEqual(sum(w for w, ep in zip(weights, episodes) if ep == 9), 0.5)
        self.assertAlmostEqual(sum(w for w, ep in zip(weights, episodes) if ep != 9), 0.5)

    def test_custom_probability(self):
        episodes = [1, 2, 8, 9]
        weights, _, _ = frame_sampling_weights(episodes, {8, 9}, 0.7)
        self.assertAlmostEqual(sum(weights[2:]), 0.7)
        self.assertAlmostEqual(sum(weights[:2]), 0.3)

    def test_requires_both_groups(self):
        with self.assertRaisesRegex(ValueError, "Both"):
            frame_sampling_weights([8, 9], {8, 9}, 0.5)


if __name__ == "__main__":
    unittest.main()
