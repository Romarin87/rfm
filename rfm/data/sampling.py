"""Batch samplers for padded reaction tensors."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class AtomCountBatchSampler(Sampler[list[int]]):
    """Group similar atom counts while shuffling batch order each epoch.

    MRTO contains dense pair operations and low-rank triangle updates. Grouping
    rows with similar ``n_atoms`` avoids spending most FLOPs on padding. Under
    DDP, one sorted global batch is split across ranks so every rank executes
    the same number of optimizer microsteps with comparable shapes.
    """

    def __init__(
        self,
        atom_counts: Sequence[int] | np.ndarray,
        batch_size: int,
        *,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ):
        counts = np.asarray(atom_counts, dtype=np.int32)
        if counts.ndim != 1 or len(counts) == 0:
            raise ValueError("atom_counts must be a non-empty one-dimensional sequence")
        if np.any(counts <= 0):
            raise ValueError("atom_counts must be positive")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid DDP layout: num_replicas={num_replicas}, rank={rank}")
        self.atom_counts = counts
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        global_batch = self.batch_size * self.num_replicas
        if self.drop_last:
            return len(self.atom_counts) // global_batch
        return math.ceil(len(self.atom_counts) / global_batch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        random_order = torch.randperm(len(self.atom_counts), generator=generator).numpy()
        order = random_order[
            np.argsort(self.atom_counts[random_order], kind="stable")
        ]
        global_batch = self.batch_size * self.num_replicas
        if self.drop_last:
            order = order[: (len(order) // global_batch) * global_batch]
        elif self.num_replicas > 1 and len(order) % global_batch:
            pad = global_batch - len(order) % global_batch
            order = np.concatenate([order, np.repeat(order[-1], pad)])

        batches = [order[start : start + global_batch] for start in range(0, len(order), global_batch)]
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        for batch_index in batch_order:
            global_indices = batches[batch_index]
            local_indices = global_indices[self.rank :: self.num_replicas]
            yield [int(index) for index in local_indices]
