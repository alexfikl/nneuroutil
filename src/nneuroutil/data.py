# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator

import array_api_compat
import numpy as np

from nneuroutil.typing import ArrayND

# {{{ SlicedDataLoader


class SlicedDataLoader:
    """A lightweight data loader for in-memory data."""

    ds: tuple[ArrayND, ...]
    """A tuple of arrays that should be loaded together in slices of *batch_size*."""

    size: int
    """The size of the dataset."""
    batch_size: int
    """The size of each batch."""
    shuffle: bool
    """If *True*, the data is shuffled at every epoch."""
    drop_last: bool
    """If *True*, the last batch will be dropped if the size of the datasets is
    not exactly divisible by the *batch_size*. Otherwise, the last batch will
    simply be smaller.
    """

    def __init__(
        self,
        *ds: ArrayND,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(ds) == 0:
            raise ValueError("no dataset given")

        dataset = ds[0]
        if not all(dataset.shape == d.shape for d in ds[1:]):
            raise ValueError(
                f"datasets must have the same shape: {[tuple(d.shape) for d in ds]}"
            )

        if batch_size <= 0:
            raise ValueError(f"'batch_size' must be positive: {batch_size}")

        if rng is None:
            rng = np.random.default_rng()

        xp = array_api_compat.array_namespace(dataset)

        num_samples = len(dataset)
        if drop_last:
            num_samples = (num_samples // batch_size) * batch_size

        if shuffle:
            indices = xp.asarray(
                rng.permutation(num_samples),
                dtype=xp.int32,
                device=dataset.device,
            )
        else:
            indices = xp.arange(num_samples, dtype=xp.int32, device=dataset.device)

        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = num_samples

        self.indices = indices
        self.device = dataset.device
        self.rng = rng

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[ArrayND, ...]]:
        for start in range(0, self.num_samples, self.batch_size):
            sl = self.indices[start : start + self.batch_size]
            yield tuple(d[sl] for d in self.ds)


# }}}
