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

        num_samples = len(dataset)
        if drop_last:
            num_samples = (num_samples // batch_size) * batch_size

        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = num_samples

        self.device = dataset.device
        self.rng = rng

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[ArrayND, ...]]:
        xp = array_api_compat.array_namespace(*self.ds)

        if self.shuffle:
            indices = xp.asarray(
                self.rng.permutation(self.num_samples),
                dtype=xp.int32,
                device=self.device,
            )
        else:
            indices = xp.arange(self.num_samples, dtype=xp.int32, device=self.device)

        for start in range(0, self.num_samples, self.batch_size):
            sl = indices[start : start + self.batch_size]
            yield tuple(d[sl] for d in self.ds)


# }}}


# {{{ SlidingWindowDataset


class SlidingWindowDataset:
    """A sliding window dataset for tensors of shape ``(nrealizations, maxit, d)``.

    This constructs sliding windows of the shape ``(L, d)`` for each realization.
    """

    xs: tuple[ArrayND, ...]
    """The tensors for which to compute sliding windows."""
    window_size: int
    """The size of each window."""
    nwindows: int
    """The number of windows per realization."""

    def __init__(
        self,
        *xs: ArrayND,
        window_size: int,
        nwindows: int | None = None,
    ) -> None:
        if not xs:
            raise ValueError(f"{type(self).__name__}: no tensors are provided")

        shape = xs[0].shape
        if not len(shape) == 3:
            raise ValueError(f"tensors must be 3d: {shape}")

        if not all(shape == x.shape for x in xs[1:]):
            raise ValueError("tensors must have the same shape")

        self.xs = xs
        self.window_size = window_size
        self.nrealizations, self.maxit, self.dim = shape

        if nwindows is None:
            nwindows = self.maxit - self.window_size + 1

        if nwindows < 1:
            raise ValueError(f"'nwindows' must be >= 1: got {nwindows}")

        self.nwindows = nwindows
        self.total_windows = self.nrealizations * self.nwindows
        self.window_step = (
            (self.maxit - self.window_size) / (nwindows - 1) if nwindows > 1 else 0.0
        )

        if self.window_step > self.window_size - 1:
            raise ValueError(
                "'nwindows' is too large for 'window_size': consecutive windows "
                "do not overlap (min overlap is 1)"
            )

    def __len__(self) -> int:
        return self.total_windows

    def window(self, index: int) -> slice:
        if not -len(self) <= index < len(self):
            raise IndexError(
                f"index {index} is out of bounds for dataset of length {len(self)}"
            )

        i = index % self.nwindows
        n = round(i * self.window_step)

        return slice(n, n + self.window_size)

    def __getitem__(self, index: int) -> tuple[ArrayND, ...]:
        ridx = index // self.nwindows
        window = self.window(index)

        return tuple(x[ridx, window, :] for x in self.xs)


# }}}
