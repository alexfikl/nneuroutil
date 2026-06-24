# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from nneuroutil.helpers import module_logger
from nneuroutil.scores import dice_score
from nneuroutil.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_dice_score


def test_dice_score(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 16)))

    # default
    eps = 1.0e-6
    result = dice_score(x, y, eps=eps)
    expected = (2.0 * xp.sum(x * y) + eps) / (xp.sum(x) + xp.sum(y) + eps)
    assert float(xp.abs(result - expected)) < 1.0e-15

    # zeros
    x = xp.zeros((3, 8))
    y = xp.zeros((3, 8))
    assert float(xp.abs(dice_score(x, y) - 1.0)) < 1.0e-15

    # equal
    x = xp.ones((4, 16))
    assert float(xp.abs(dice_score(x, x) - 1.0)) < 1.0e-15

    # mismatch
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 8)))
    with pytest.raises(ValueError, match="shape mismatch"):
        dice_score(x, y)


def test_dice_score_axis(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 16)))

    eps = 1.0e-6
    result = dice_score(x, y, eps=eps, axis=1)
    assert result.shape == (4,)

    expected_0 = (2.0 * xp.sum(x[0] * y[0]) + eps) / (xp.sum(x[0]) + xp.sum(y[0]) + eps)
    assert float(xp.abs(result[0] - expected_0)) < 1.0e-15


def test_dice_score_partial_zeros(xp: Any) -> None:
    # sum(x[i]) == 0 for both rows; second row is all zeros
    x = xp.asarray([[1.0, -1.0, 2.0, -2.0], [0.0, 0.0, 0.0, 0.0]])
    y = xp.asarray([[0.5, -0.5, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]])

    result = dice_score(x, y, axis=1)
    assert float(xp.abs(result[0] - 2.0)) < 1.0e-5
    assert float(xp.abs(result[1] - 1.0)) < 1.0e-15


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
