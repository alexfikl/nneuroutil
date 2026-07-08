# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from nneuroutil.array_api_extras import array_equal
from nneuroutil.helpers import module_logger

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_sliced_data_loader


def test_sliced_data_loader(xp: Any) -> None:
    from nneuroutil.data import SlicedDataLoader

    rng = np.random.default_rng(seed=42)

    # size % batch_size == 0
    x = xp.reshape(xp.arange(12), (6, 2))
    loader = SlicedDataLoader(x, batch_size=2, shuffle=False, rng=rng)
    assert len(loader) == 3

    batches = list(loader)
    assert len(batches) == 3
    assert array_equal(batches[0][0], x[0:2])
    assert array_equal(batches[1][0], x[2:4])
    assert array_equal(batches[2][0], x[4:6])

    # drop_last = True
    x = xp.arange(7)
    loader = SlicedDataLoader(x, batch_size=2, shuffle=False, drop_last=True)
    assert len(loader) == 3  # 6 // 2

    batches = list(loader)
    assert len(batches) == 3
    assert list(batches[0][0]) == [0, 1]
    assert list(batches[1][0]) == [2, 3]
    assert list(batches[2][0]) == [4, 5]

    # drop_last = False
    x = xp.arange(7)
    loader = SlicedDataLoader(x, batch_size=2, shuffle=False, drop_last=False)
    assert len(loader) == 4

    batches = list(loader)
    assert len(batches) == 4
    assert list(batches[3][0]) == [6]


def test_sliced_data_loader_shuffle(xp: Any) -> None:
    from nneuroutil.data import SlicedDataLoader

    rng = np.random.default_rng(seed=42)

    x = xp.reshape(xp.arange(12), (6, 2))
    loader = SlicedDataLoader(x, batch_size=2, shuffle=True, rng=rng)

    assert len(loader) == 3
    batches = list(loader)
    assert len(batches) == 3

    # all elements appear exactly once across batches
    combined = []
    for (bx,) in batches:
        combined.extend(list(bx.ravel()))
    assert sorted(combined) == list(range(12))


def test_sliced_data_loader_multi_dataset(xp: Any) -> None:
    from nneuroutil.data import SlicedDataLoader

    x = xp.arange(6)
    y = x * 10
    loader = SlicedDataLoader(x, y, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 3
    for i, (bx, by) in enumerate(batches):
        assert array_equal(bx, x[i * 2 : (i + 1) * 2])
        assert array_equal(by, y[i * 2 : (i + 1) * 2])


def test_sliced_data_loader_batch_size(xp: Any) -> None:
    from nneuroutil.data import SlicedDataLoader

    x = xp.arange(5)
    loader = SlicedDataLoader(x, batch_size=1, shuffle=False)

    assert len(loader) == 5
    assert list(loader)[3][0] == xp.asarray([3])

    x = np.arange(5)
    loader = SlicedDataLoader(x, batch_size=10, shuffle=False)

    assert len(loader) == 1
    assert array_equal(next(iter(loader))[0], x)


# }}}


# {{{ test_sliced_data_loader_errors


def test_sliced_data_loader_no_dataset() -> None:
    from nneuroutil.data import SlicedDataLoader

    with pytest.raises(ValueError, match="no dataset given"):
        SlicedDataLoader()

    with pytest.raises(ValueError, match="must have the same shape"):
        SlicedDataLoader(np.arange(6), np.arange(12))

    with pytest.raises(ValueError, match="'batch_size' must be positive"):
        SlicedDataLoader(np.arange(6), batch_size=0)


# }}}


# {{{ test_sliding_window_dataset


def test_sliding_window_dataset(xp: Any) -> None:
    from nneuroutil.data import SlidingWindowDataset

    nruns, maxit, dim = 2, 5, 3
    window_size = 3
    nwindows = maxit - window_size + 1

    x = xp.reshape(
        xp.arange(nruns * maxit * dim, dtype=xp.float64), (nruns, maxit, dim)
    )

    dataset = SlidingWindowDataset(x, window_size=window_size)

    assert len(dataset) == nruns * nwindows

    # first window of first realization starts at 0
    (win,) = dataset[0]
    assert array_equal(win, x[0, 0:window_size, :])

    # first window of second realization
    (win,) = dataset[nwindows]
    assert array_equal(win, x[1, 0:window_size, :])

    # last window ends at maxit
    (win,) = dataset[-1]
    assert array_equal(win, x[-1, maxit - window_size : maxit, :])


# }}}


# {{{ test_sliding_window_dataset_nwindows


def test_sliding_window_dataset_nwindows(xp: Any) -> None:
    from nneuroutil.data import SlidingWindowDataset

    nruns, maxit, dim = 2, 10, 3
    window_size = 3
    nwindows = 5

    x = xp.reshape(
        xp.arange(nruns * maxit * dim, dtype=xp.float64), (nruns, maxit, dim)
    )
    dataset = SlidingWindowDataset(x, window_size=window_size, nwindows=nwindows)

    assert len(dataset) == nruns * nwindows

    for r in range(nruns):
        # first window always starts at 0
        (first,) = dataset[r * nwindows]
        assert array_equal(first, x[r, 0:window_size, :])

        # last window always ends at maxit
        (last,) = dataset[r * nwindows + nwindows - 1]
        assert array_equal(last, x[r, maxit - window_size : maxit, :])

    # spot-check intermediate windows use round() spacing
    step = (maxit - window_size) / (nwindows - 1)
    for r in range(nruns):
        for i in range(nwindows):
            (win,) = dataset[r * nwindows + i]
            start = round(i * step)
            assert array_equal(win, x[r, start : start + window_size, :])


# }}}


# {{{ test_sliding_window_dataset_overlap


def test_sliding_window_dataset_overlap(xp: Any) -> None:
    from nneuroutil.data import SlidingWindowDataset

    nruns, maxit, dim = 2, 10, 3
    x = xp.reshape(
        xp.arange(nruns * maxit * dim, dtype=xp.float64), (nruns, maxit, dim)
    )

    # window_size=3, nwindows=4 -> step = 7/3 > 2 -> consecutive windows
    # do not overlap by at least 1
    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowDataset(x, window_size=3, nwindows=4)

    # borderline: step == window_size - 1 -> overlap exactly 1 (allowed)
    SlidingWindowDataset(x, window_size=4, nwindows=3)


# }}}


# {{{ test_random_split


def test_random_split_partitions(xp: Any) -> None:
    from nneuroutil.data import random_split

    rng = np.random.default_rng(seed=42)
    x = xp.arange(10)

    # integer partitions of different sizes
    splits = random_split([x], [3, 7], rng=rng)
    assert len(splits) == 2
    assert [s[0].shape[0] for s in splits] == [3, 7]
    assert array_equal(xp.sort(xp.concat([s[0] for s in splits])), x)

    splits = random_split([x], [2, 3, 5], rng=rng)
    assert [s[0].shape[0] for s in splits] == [2, 3, 5]
    assert array_equal(xp.sort(xp.concat([s[0] for s in splits])), x)

    # an empty split (size 0) is allowed
    splits = random_split([x], [0, 10], rng=rng)
    assert [s[0].shape[0] for s in splits] == [0, 10]
    assert array_equal(xp.sort(xp.concat([s[0] for s in splits])), x)

    # fractions: the floor remainder is distributed round-robin.
    # n=11, [0.5, 0.5] -> floor([5, 5]) -> remainder 1 -> [6, 5]
    y = xp.arange(11)
    splits = random_split([y], [0.5, 0.5], rng=rng)
    assert [s[0].shape[0] for s in splits] == [6, 5]
    assert array_equal(xp.sort(xp.concat([s[0] for s in splits])), y)


def test_random_split_multi_array(xp: Any) -> None:
    from nneuroutil.data import random_split

    rng = np.random.default_rng(seed=42)
    x = xp.arange(10)
    y = x * 10

    splits = random_split([x, y], [6, 4], rng=rng)
    assert len(splits) == 2
    assert [s[0].shape[0] for s in splits] == [6, 4]

    # parallel arrays stay aligned under the same permutation
    for sx, sy in splits:
        assert sx.shape[0] == sy.shape[0]
        assert array_equal(sy, sx * 10)

    # each array is partitioned disjointly and completely
    for j, original in enumerate((x, y)):
        assert array_equal(xp.sort(xp.concat([s[j] for s in splits])), original)


def test_random_split_errors(xp: Any) -> None:
    from nneuroutil.data import random_split

    x = xp.arange(10)

    # integer lengths do not add up to the dataset size
    with pytest.raises(ValueError, match="do not add up to dataset size"):
        random_split([x], [3, 4])

    # fractions must sum to 1
    with pytest.raises(ValueError, match="must sum to 1"):
        random_split([x], [0.3, 0.3])

    # fractions must lie in [0, 1] (sum is 1, but an element is out of range)
    with pytest.raises(ValueError, match=r"not in \[0, 1\]"):
        random_split([x], [1.5, -0.5])

    # parallel arrays must share the leading dimension
    with pytest.raises(ValueError, match="same leading dimension"):
        random_split([x, xp.arange(12)], [5, 5])


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
