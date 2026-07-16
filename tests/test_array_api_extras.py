# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from nneuroutil.array_api_extras import array_equal, deinterleave, interleave
from nneuroutil.helpers import module_logger

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_interleave


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_interleave(xp: Any, axis: int) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal((4, 6)))
    y = xp.asarray(rng.standard_normal((4, 6)))

    z = interleave(x, y, axis=axis)

    expected_shape = list(x.shape)
    expected_shape[axis] *= 2
    assert z.shape == tuple(expected_shape)

    idx0 = (slice(None),) * (axis % x.ndim) + (slice(0, None, 2),)
    idx1 = (slice(None),) * (axis % x.ndim) + (slice(1, None, 2),)
    assert array_equal(z[idx0], x)
    assert array_equal(z[idx1], y)

    # roundtrip
    x2, y2 = deinterleave(z, axis=axis)
    assert array_equal(x2, x)
    assert array_equal(y2, y)

    # shape mismatch
    with pytest.raises(ValueError, match="same shape"):
        interleave(x, y[:, :-1])


def test_deinterleave_odd_size(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = xp.asarray(rng.standard_normal((4, 5)))

    with pytest.raises(ValueError, match="must be even"):
        deinterleave(z, axis=-1)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
