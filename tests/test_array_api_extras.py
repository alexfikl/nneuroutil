# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from nneuroutil.array_api_extras import (
    array_equal,
    deinterleave,
    fill_diagonal,
    histogram,
    interleave,
)
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


# {{{ test_fill_diagonal


@pytest.mark.parametrize("shape", [(4, 4), (3, 5), (5, 3)])
def test_fill_diagonal(xp: Any, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(seed=42)
    a = xp.asarray(rng.standard_normal(shape))

    x = fill_diagonal(a, 1.0, inplace=False)

    # the diagonal is set to 'value' and the rest of the array is unchanged
    expected = xp.where(xp.eye(*shape, dtype=bool), 1.0, a)
    assert array_equal(x, expected)


def test_fill_diagonal_3d(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    a = xp.asarray(rng.standard_normal((2, 3, 3)))

    x = fill_diagonal(a, 2.0, inplace=False)

    # the diagonal of each slice of the trailing two axes is set to 'value'
    expected = xp.where(xp.eye(3, dtype=bool), 2.0, a)
    assert array_equal(x, expected)


def test_fill_diagonal_inplace(xp: Any) -> None:
    if "jax" in xp.__name__:
        pytest.skip("jax arrays are immutable")

    rng = np.random.default_rng(seed=42)
    a = xp.asarray(rng.standard_normal((4, 4)))

    x = fill_diagonal(a, 1.0, inplace=True)

    # the input is modified in place and the same object is returned
    assert x is a
    assert array_equal(xp.diagonal(a), xp.full((4,), 1.0, dtype=a.dtype))


def test_fill_diagonal_not_writable(xp: Any) -> None:
    if "jax" in xp.__name__ or "torch" in xp.__name__:
        pytest.skip("only numpy supports read-only arrays")

    a = xp.asarray(np.eye(3))
    a.flags.writeable = False  # spell: disable

    with pytest.raises(ValueError, match="not writable"):
        fill_diagonal(a, 1.0, inplace=True)


def test_fill_diagonal_validation(xp: Any) -> None:
    with pytest.raises(ValueError, match="at least dimension 2"):
        fill_diagonal(xp.asarray(np.array([1.0, 2.0, 3.0])), 1.0, inplace=False)

    with pytest.raises(NotImplementedError, match="not implemented"):
        fill_diagonal(xp.asarray(np.eye(2)), 1.0, wrap=True, inplace=False)


# }}}


# {{{ test_histogram_fallback


@pytest.mark.parametrize("density", [False, True])
@pytest.mark.parametrize("bins", [8, 32])
def test_histogram_fallback(xp: Any, *, bins: int, density: bool) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal(1000))

    lo, hi = -2.5, 2.5
    counts, edges = histogram(
        x, bins, range=(lo, hi), density=density, fallback=True, xp=xp
    )
    ref_counts, ref_edges = histogram(x, bins, range=(lo, hi), density=density, xp=xp)

    assert counts.shape == (bins,)
    assert edges.shape == (bins + 1,)

    # the fallback always computes float64 edges matching the native ones
    assert array_equal(edges, xp.astype(ref_edges, xp.float64))

    # the reference counts may come back in a different dtype, so compare in float64
    counts = xp.astype(counts, xp.float64)
    ref_counts = xp.astype(ref_counts, xp.float64)

    if density:
        assert np.allclose(counts, ref_counts, rtol=1e-6, atol=1e-12)
    else:
        assert array_equal(counts, ref_counts)
        assert xp.sum(counts) == xp.sum((x >= lo) & (x <= hi))


def test_histogram_fallback_range(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal(1000))

    counts, edges = histogram(x, 12, fallback=True, xp=xp)
    ref_counts, ref_edges = histogram(x, 12, xp=xp)

    # the fallback computes the same edges from [x.min(), x.max()] and must
    # count the maximum element in the last bin, exactly like numpy
    assert array_equal(edges, xp.astype(ref_edges, xp.float64))
    assert array_equal(xp.astype(counts, xp.float64), xp.astype(ref_counts, xp.float64))


def test_histogram_fallback_upper_edge(xp: Any) -> None:
    x = xp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.0])

    counts, edges = histogram(x, 2, range=(0.0, 2.0), fallback=True, xp=xp)

    # the upper edge is inclusive: bin [1, 2] gets {1.0, 1.5, 2.0, 2.0}
    assert array_equal(edges, xp.asarray(np.array([0.0, 1.0, 2.0])))
    assert array_equal(counts, xp.asarray(np.array([2, 4])))


def test_histogram_fallback_nonfinite(xp: Any) -> None:
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="not finite"):
            histogram(xp.asarray(np.array([1.0, value, 2.0])), 4, fallback=True, xp=xp)


@pytest.mark.parametrize("density", [False, True])
def test_histogram_fallback_integer(xp: Any, *, density: bool) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.integers(0, 10, size=1000))

    bins = 8
    lo, hi = 0, 10
    counts, edges = histogram(
        x, bins, range=(lo, hi), density=density, fallback=True, xp=xp
    )
    ref_counts, ref_edges = histogram(x, bins, range=(lo, hi), density=density)

    assert counts.shape == (bins,)
    assert edges.shape == (bins + 1,)

    # integer inputs must produce the same float64 edges as the native path
    assert array_equal(edges, xp.astype(ref_edges, xp.float64))

    counts = xp.astype(counts, xp.float64)
    ref_counts = xp.astype(ref_counts, xp.float64)

    if density:
        assert np.allclose(counts, ref_counts, rtol=1e-6, atol=1e-12)
    else:
        assert array_equal(counts, ref_counts)
        assert xp.sum(counts) == xp.sum((x >= lo) & (x <= hi))


@pytest.mark.parametrize("density", [False, True])
def test_histogram_fallback_constant(xp: Any, *, density: bool) -> None:
    x = xp.asarray([1.0, 1.0, 1.0])

    bins = 4
    counts, edges = histogram(x, bins, density=density, fallback=True, xp=xp)
    ref_counts, ref_edges = histogram(x, bins, density=density, xp=xp)

    # numpy expands the auto-detected range by +/- 0.5 for constant arrays
    expected_edges = xp.asarray(np.array([0.5, 0.75, 1.0, 1.25, 1.5]))
    assert array_equal(edges, expected_edges)
    assert array_equal(edges, xp.astype(ref_edges, xp.float64))

    if density:
        assert counts.shape == (bins,)
        assert np.allclose(counts, ref_counts, rtol=1e-6, atol=1e-12)
        assert xp.sum(counts) * (edges[1] - edges[0]) == pytest.approx(1.0)
    else:
        assert array_equal(counts, xp.asarray(np.array([0, 0, 3, 0])))
        assert array_equal(counts, ref_counts)


def test_histogram_fallback_counts(xp: Any) -> None:
    x = xp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    counts, edges = histogram(x, 2, range=(0.0, 2.5), fallback=True, xp=xp)

    # values on an interior edge go in the right bin and out-of-range ones are
    # dropped: bin [0, 1.25) gets {0.0, 0.5, 1.0} and [1.25, 2.5] gets {1.5, 2.0}
    assert array_equal(edges, xp.asarray(np.array([0.0, 1.25, 2.5])))
    assert array_equal(counts, xp.asarray(np.array([3, 2])))


def test_histogram_fallback_density(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x = xp.asarray(rng.standard_normal(1000))

    lo, hi = -2.5, 2.5
    counts, edges = histogram(x, 12, range=(lo, hi), density=True, fallback=True, xp=xp)

    # a normalized histogram integrates to 1 over its range
    assert xp.sum(counts) * (edges[1] - edges[0]) == pytest.approx(1.0, abs=1e-12)


def test_histogram_fallback_validation(xp: Any) -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        histogram(xp.asarray(np.ones((2, 2))), 4, fallback=True, xp=xp)

    for bins in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            histogram(xp.asarray(np.array([1.0, 2.0])), bins, fallback=True, xp=xp)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
