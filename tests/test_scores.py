# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from nneuroutil.helpers import module_logger
from nneuroutil.scores import (
    dice_score,
    jaccard_index,
    tversky_index,
    volumetric_similarity,
)

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


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
    assert float(xp.abs(dice_score(x, y, eps=eps) - 1.0)) < 1.0e-15

    # equal
    x = xp.ones((4, 16))
    assert float(xp.abs(dice_score(x, x, eps=eps) - 1.0)) < 1.0e-15

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

    result = dice_score(x, y, eps=1.0e-6, axis=1)
    assert float(xp.abs(result[0] - 2.0)) < 1.0e-5
    assert float(xp.abs(result[1] - 1.0)) < 1.0e-15


# }}}


# {{{ test_jaccard_index


def test_jaccard_index(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 16)))

    eps = 1.0e-6
    x_n_y = xp.sum(x * y)
    x_u_y = xp.sum(x) + xp.sum(y) - x_n_y
    result = jaccard_index(x, y, eps=eps)
    expected = (x_n_y + eps) / (x_u_y + eps)
    assert float(xp.abs(result - expected)) < 1.0e-15

    # zeros
    x = xp.zeros((3, 8))
    y = xp.zeros((3, 8))
    assert float(xp.abs(jaccard_index(x, y, eps=eps) - 1.0)) < 1.0e-15

    # equal
    x = xp.ones((4, 16))
    assert float(xp.abs(jaccard_index(x, x, eps=eps) - 1.0)) < 1.0e-15


def test_jaccard_index_axis(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 16)))

    eps = 1.0e-6
    result = jaccard_index(x, y, eps=eps, axis=1)
    assert result.shape == (4,)

    x_n_y = xp.sum(x[0] * y[0])
    x_u_y = xp.sum(x[0]) + xp.sum(y[0]) - x_n_y
    expected_0 = (x_n_y + eps) / (x_u_y + eps)
    assert float(xp.abs(result[0] - expected_0)) < 1.0e-15


def test_jaccard_index_partial_zeros(xp: Any) -> None:
    x = xp.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    y = xp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    eps = 1.0e-6
    result = jaccard_index(x, y, eps=eps, axis=1)
    assert float(xp.abs(result[0] - 0.5)) < 10.0 * eps
    assert float(xp.abs(result[1] - 1.0)) < 1.0e-15


# }}}


# {{{ test_volumetric_similarity


def test_volumetric_similarity(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 16)))

    eps = 1.0e-6
    vx = xp.sum(x)
    vy = xp.sum(y)
    result = volumetric_similarity(x, y, eps=eps)
    expected = 1.0 - (xp.abs(vx - vy) + eps) / (vx + vy + eps)
    assert float(xp.abs(result - expected)) < 1.0e-15

    # zeros: vx = vy = 0 → result = 1 - eps/eps = 0
    x = xp.zeros((3, 8))
    y = xp.zeros((3, 8))
    assert float(xp.abs(volumetric_similarity(x, y, eps=eps))) < 1.0e-15

    # equal ones
    x = xp.ones((4, 16))
    expected = 1.0 - eps / (2.0 * xp.sum(x) + eps)
    assert float(xp.abs(volumetric_similarity(x, x, eps=eps) - expected)) < 1.0e-15

    # mismatch
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 8)))
    with pytest.raises(ValueError, match="shape mismatch"):
        volumetric_similarity(x, y)


def test_volumetric_similarity_axis(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal((4, 16)))
    y = xp.asarray(rng.standard_normal((4, 16)))

    eps = 1.0e-6
    result = volumetric_similarity(x, y, eps=eps, axis=1)
    assert result.shape == (4,)

    vx = xp.sum(x[0])
    vy = xp.sum(y[0])
    expected_0 = 1.0 - (xp.abs(vx - vy) + eps) / (vx + vy + eps)
    assert float(xp.abs(result[0] - expected_0)) < 1.0e-15


def test_volumetric_similarity_partial_zeros(xp: Any) -> None:
    x = xp.asarray([[3.0, 1.0], [0.0, 0.0]])
    y = xp.asarray([[1.0, 1.0], [0.0, 0.0]])

    eps = 1.0e-6
    result = volumetric_similarity(x, y, eps=eps, axis=1)
    assert float(xp.abs(result[0] - 2.0 / 3.0)) < 10.0 * eps
    assert float(xp.abs(result[1])) < 1.0e-15


# }}}


# {{{ test_tversky_index


def test_tversky_index(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.integers(0, 2, (4, 16)))
    y = xp.asarray(rng.integers(0, 2, (4, 16)))

    eps = 1.0e-6
    x_n_y = xp.sum(x * y)
    x_m_y = xp.sum(x * (1 - y))
    y_m_x = xp.sum((1 - x) * y)
    result = tversky_index(x, y, eps=eps)
    expected = (x_n_y + eps) / (x_n_y + 0.5 * x_m_y + 0.5 * y_m_x + eps)
    assert float(xp.abs(result - expected)) < 1.0e-15

    # alpha = beta = 0.5 with eps = 0 matches dice_score
    assert (
        float(xp.abs(tversky_index(x, y, alpha=0.5, beta=0.5) - dice_score(x, y)))
        < 1.0e-15
    )

    # alpha = beta = 1.0 matches jaccard_index
    assert (
        float(xp.abs(tversky_index(x, y, alpha=1.0, beta=1.0) - jaccard_index(x, y)))
        < 1.0e-15
    )

    # zeros
    x = xp.zeros((3, 8))
    y = xp.zeros((3, 8))
    assert float(xp.abs(tversky_index(x, y, eps=eps) - 1.0)) < 1.0e-15

    # equal
    x = xp.ones((4, 16))
    assert float(xp.abs(tversky_index(x, x, eps=eps) - 1.0)) < 1.0e-15

    # mismatch
    x = xp.asarray(rng.integers(0, 2, (4, 16)))
    y = xp.asarray(rng.integers(0, 2, (4, 8)))
    with pytest.raises(ValueError, match="shape mismatch"):
        tversky_index(x, y)


def test_tversky_index_axis(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.integers(0, 2, (4, 16)))
    y = xp.asarray(rng.integers(0, 2, (4, 16)))

    eps = 1.0e-6
    result = tversky_index(x, y, eps=eps, axis=1)
    assert result.shape == (4,)

    x_n_y = xp.sum(x[0] * y[0])
    x_m_y = xp.sum(x[0] * (1 - y[0]))
    y_m_x = xp.sum((1 - x[0]) * y[0])
    expected_0 = (x_n_y + eps) / (x_n_y + 0.5 * x_m_y + 0.5 * y_m_x + eps)
    assert float(xp.abs(result[0] - expected_0)) < 1.0e-15


def test_tversky_index_partial_zeros(xp: Any) -> None:
    x = xp.asarray([[1.0, 0.0], [0.0, 0.0]])
    y = xp.asarray([[1.0, 1.0], [0.0, 0.0]])

    eps = 1.0e-6
    result = tversky_index(x, y, eps=eps, axis=1)
    assert float(xp.abs(result[0] - 2.0 / 3.0)) < 10.0 * eps
    assert float(xp.abs(result[1] - 1.0)) < 1.0e-15


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
