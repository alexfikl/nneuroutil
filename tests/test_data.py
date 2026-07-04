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

    perm = list(loader.indices)
    assert sorted(perm) == [0, 1, 2, 3, 4, 5]
    assert perm != [0, 1, 2, 3, 4, 5]

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
