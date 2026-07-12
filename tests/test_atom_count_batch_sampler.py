from __future__ import annotations

import unittest

import numpy as np

from rfm.data.sampling import AtomCountBatchSampler


class AtomCountBatchSamplerTest(unittest.TestCase):
    def test_single_rank_covers_every_sample_without_padding_duplicates(self) -> None:
        counts = np.asarray([3, 8, 2, 7, 4, 6, 5, 9, 1, 10], dtype=np.int32)
        sampler = AtomCountBatchSampler(counts, 4, seed=11)
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(counts))))
        self.assertEqual(sorted(len(batch) for batch in batches), [2, 4, 4])
        for batch in batches:
            values = counts[batch]
            self.assertLessEqual(int(values.max() - values.min()), len(batch) - 1)

    def test_epoch_shuffle_is_deterministic_and_changes_batch_order(self) -> None:
        counts = np.repeat(np.arange(1, 9), 4)
        first = AtomCountBatchSampler(counts, 4, seed=5)
        second = AtomCountBatchSampler(counts, 4, seed=5)
        self.assertEqual(list(first), list(second))
        first.set_epoch(1)
        self.assertNotEqual(list(first), list(second))

    def test_ddp_ranks_have_equal_steps_and_disjoint_samples_without_padding(self) -> None:
        counts = np.repeat(np.arange(1, 7), 4)
        rank0 = AtomCountBatchSampler(counts, 3, seed=7, num_replicas=2, rank=0)
        rank1 = AtomCountBatchSampler(counts, 3, seed=7, num_replicas=2, rank=1)
        batches0 = list(rank0)
        batches1 = list(rank1)
        self.assertEqual(len(batches0), len(batches1))
        indices0 = {index for batch in batches0 for index in batch}
        indices1 = {index for batch in batches1 for index in batch}
        self.assertFalse(indices0 & indices1)
        self.assertEqual(indices0 | indices1, set(range(len(counts))))


if __name__ == "__main__":
    unittest.main()
