# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import array_api_compat
import numpy as np
from typing_extensions import TypeIs

from nneuroutil.typing import ArrayND

# {{{ SlicedDataLoader


class SlicedDataLoader:
    """A lightweight data loader for in-memory data."""

    ds: tuple[ArrayND[np.inexact[Any]], ...]
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
        *ds: ArrayND[np.inexact[Any]],
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
        xp: Any = None,
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

        if xp is None:
            xp = array_api_compat.array_namespace(*ds)

        self.xp = xp
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = num_samples

        self.device = dataset.device
        self.rng = rng

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[ArrayND[np.inexact[Any]], ...]]:
        if self.shuffle:
            indices = self.xp.asarray(
                self.rng.permutation(self.num_samples),
                dtype=self.xp.int32,
                device=self.device,
            )
        else:
            indices = self.xp.arange(
                self.num_samples, dtype=self.xp.int32, device=self.device
            )

        for start in range(0, self.num_samples, self.batch_size):
            sl = indices[start : start + self.batch_size]
            yield tuple(d[sl] for d in self.ds)


# }}}


# {{{ SlidingWindowDataset


class SlidingWindowDataset:
    """A sliding window dataset for tensors of shape ``(nrealizations, maxit, d)``.

    This constructs sliding windows of the shape ``(L, d)`` for each realization.
    """

    xs: tuple[ArrayND[np.inexact[Any]], ...]
    """The tensors for which to compute sliding windows."""
    window_size: int
    """The size of each window."""
    nwindows: int
    """The number of windows per realization."""

    def __init__(
        self,
        *xs: ArrayND[np.inexact[Any]],
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

    def __getitem__(self, index: int) -> tuple[ArrayND[np.inexact[Any]], ...]:
        ridx = index // self.nwindows
        window = self.window(index)

        return tuple(x[ridx, window, :] for x in self.xs)


# }}}


# {{{ random_split


def is_int_sequence(x: Sequence[Any]) -> TypeIs[Sequence[int]]:
    return all(type(xi) is int for xi in x)


def random_split(
    xs: Sequence[ArrayND[np.inexact[Any]]],
    splits: Sequence[int] | Sequence[float],
    *,
    rng: np.random.Generator | None = None,
    xp: Any = None,
) -> tuple[tuple[ArrayND[np.inexact[Any]], ...], ...]:
    """Randomly split a sequence of arrays with the same leading dimension.

    The split can be given as a sequence of integers that sum up to the size of
    the arrays or as a sequence of percentages in :math:`[0, 1]`. If the split
    is given as percentages, any remaining elements are distributed to all the
    splits.
    """

    # NOTE: this is mainly inspired by torch.utils.data.random_split
    # https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/data/dataset.py#L449
    # but works on generic Sequences and does not use any pytorch datastructures.
    # otherwise the code is pretty equivalent

    if not xs:
        return ()

    if xp is None:
        xp = array_api_compat.array_namespace(*xs)

    if rng is None:
        rng = np.random.default_rng()

    n = len(xs[0])
    if not all(len(x) == n for x in xs):
        raise ValueError("'xs' datasets must have the same leading dimension")

    # NOTE: pytorch checks isclose(sum(splits), 1) and splits <= 1 here.
    if is_int_sequence(splits):
        lengths = splits
    else:
        if not math.isclose(sum(splits), 1):
            raise ValueError(f"'splits' must sum to 1: {splits}")

        lengths = []
        for i, frac in enumerate(splits):
            if frac < 0 or frac > 1:
                raise ValueError(f"split '{i}' is not in [0, 1]: {splits}")

            lengths.append(math.floor(n * frac))

        # NOTE: add 1 to the lengths until the remainder is distributed
        remainder = n - sum(lengths)
        for i in range(remainder):
            lengths[i % len(lengths)] += 1

    if n != sum(lengths):
        raise ValueError(
            f"'splits' do not add up to dataset size: got {sum(lengths)} (expected {n})"
        )

    from itertools import accumulate

    indices = xp.asarray(rng.permutation(n), dtype=xp.int32, device=xs[0].device)
    return tuple(
        tuple(x[indices[offset - length : offset]] for x in xs)
        for offset, length in zip(accumulate(lengths), lengths, strict=True)
    )


# }}}
