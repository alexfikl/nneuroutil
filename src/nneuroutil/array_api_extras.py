# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import array_api_compat
import numpy as np

from nneuroutil.typing import Array1D, ArrayND

# {{{ array_equal


def array_equal(
    a: ArrayND[np.number[Any]],
    b: ArrayND[np.number[Any]],
    *,
    equal_nan: bool = False,
    xp: Any = None,
) -> bool:
    """*True* if two arrays have the same shape and elements, *False* otherwise."""

    if equal_nan:
        raise NotImplementedError("'equal_nan=True")

    if xp is None:
        xp = array_api_compat.array_namespace(a, b)

    a, b = xp.asarray(a), xp.asarray(b)
    if a.shape != b.shape:
        return False

    return bool(xp.all(xp.equal(a, b)))


# }}}


# {{{ interleave


def interleave(
    x: ArrayND[np.number[Any]],
    y: ArrayND[np.number[Any]],
    /,
    *,
    axis: int = -1,
    xp: Any = None,
) -> ArrayND[np.number[Any]]:
    """Interleave the elements of *x* and *y* along the given axis *axis*.

    The two inputs must have the same shape. The result has the same shape as the
    inputs, except along *axis*, where the size is doubled and the entries of
    *x* and *y* alternate. For one-dimensional arrays, this is effectively just
    ``(x[0], y[0], x[1], y[1], ...)``.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"interleave: x and y must have the same shape: {x.shape} and {y.shape}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    # normalize negative axis before inserting a new one
    axis %= x.ndim

    # stack the arrays
    stacked = xp.stack([x, y], axis=axis + 1)

    # reshape everything so that it interleaves correctly
    new_shape = (*x.shape[:axis], 2 * x.shape[axis], *x.shape[axis + 1 :])
    return xp.reshape(stacked, new_shape)


# }}}


# {{{ deinterleave


def deinterleave(
    z: ArrayND[Any],
    /,
    *,
    axis: int = -1,
    xp: Any = None,
) -> tuple[ArrayND[Any], ArrayND[Any]]:
    """Split the entries of *z* into two arrays along the given axis *axis*.

    This is the inverse of :func:`interleave`: the size along *axis* must be
    even, and the two outputs correspond to the even- and odd-indexed entries.
    """
    axis %= z.ndim
    n = z.shape[axis]

    if n % 2 != 0:
        raise ValueError(f"deinterleave: size along axis {axis} must be even: {n}")

    if xp is None:
        xp = array_api_compat.array_namespace(z)

    new_shape = (*z.shape[:axis], n // 2, 2, *z.shape[axis + 1 :])
    reshaped = xp.reshape(z, new_shape)

    idx0 = (slice(None),) * axis + (slice(None), 0)
    idx1 = (slice(None),) * axis + (slice(None), 1)

    return reshaped[idx0], reshaped[idx1]


# }}}


# {{{ histogram


def histogram(
    x: Array1D[np.floating[Any]],
    bins: int,
    *,
    range: tuple[float, float] | None = None,  # ruff: ignore[builtin-argument-shadowing]
    density: bool = False,
    fallback: bool = False,
    xp: Any = None,
) -> tuple[Array1D[np.floating[Any]], Array1D[np.floating[Any]]]:
    """Compute the histogram of *x*.

    This implements a function similar to :func:`numpy.histogram` that works
    with the Array API. In most cases, it will simply dispatch to the appropriate
    backend. However, a fallback implementation based on the existing Array APIs
    is available (but generally slower).

    :arg x: input data, which must be a one-dimensional array.
    :arg bins: number of bins into which to separate the given data. Note that
        this can only be an integer, unlike in :func:`numpy.histogram`.
    :arg range: a tuple ``(xmin, xmax)`` giving the lower and upper bounds of the
        bins. If not provided, this will be ``(x.min(), x.max())``.
    :arg density: if *True*, the result will be the probability density function
        at each bin (normalized so that it integrates to 1). Otherwise, it is
        simply the number of values in each bin.
    :arg fallback: if *True*, a fallback implementation will be used instead of
        the existing ones. This is meant mostly for testing.

    :returns: a tuple of ``(counts, edges)`` similar to :func:`numpy.histogram`.
    """
    if x.ndim != 1:
        raise ValueError(f"'x' input must be 1-dimensional: {x.shape}")

    if bins <= 0:
        raise ValueError(f"'bins' must be positive: {bins}")

    if not fallback and array_api_compat.is_numpy_array(x):
        return np.histogram(x, bins=bins, range=range, density=density)
    elif not fallback and array_api_compat.is_jax_array(x):
        import jax.numpy as jnp

        return jnp.histogram(x, bins=bins, range=range, density=density)
    elif not fallback and array_api_compat.is_torch_array(x):
        import torch

        # NOTE: torch.histogram does not work for integers, so we cast it
        if not (x.is_floating_point() or x.is_complex()):
            x = x.to(torch.get_default_dtype())

        return torch.histogram(x, bins, range=range, density=density)

    if xp is None:
        xp = array_api_compat.array_namespace(x)
    else:
        assert array_api_compat.array_namespace(x) is xp

    # implement a fallback for other array libraries
    if range is None:
        xmin, xmax = xp.min(x), xp.max(x)

        if not xp.isfinite(xmin) or not xp.isfinite(xmax):
            raise ValueError(f"autodetected range [{xmin}, {xmax}] is not finite")

        # expand the range for constant arrays, like numpy does
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
    else:
        xmin, xmax = range

    if xmin >= xmax:
        raise ValueError(f"invalid range (xmin >= xmax): {range}")

    # compute where each element in x falls using searchsorted
    dtype = xp.result_type(x.dtype, xp.float64)
    edges = xp.linspace(xmin, xmax, bins + 1, dtype=dtype)
    idx = xp.searchsorted(edges, x, side="right") - 1

    # clamp values at the last edge into the last bin, like numpy does
    idx = xp.minimum(idx, xp.asarray(bins - 1, dtype=idx.dtype))

    # drop any elements outside of the given range
    mask = (x >= xmin) & (x <= xmax)  # ty: ignore[unsupported-operator]
    idx = xp.where(mask, idx, xp.asarray(bins, dtype=idx.dtype))

    # count elements per bin by sorting the bin indices and using searchsorted
    # (O(n log n)) instead of building an O(n * bins) boolean matrix
    sorted_idx = xp.sort(idx)
    boundaries = xp.searchsorted(
        sorted_idx, xp.arange(1, bins + 1, dtype=idx.dtype), side="left"
    )
    pad = xp.concatenate([xp.zeros((1,), dtype=boundaries.dtype), boundaries])
    counts = xp.diff(pad)

    if density:
        counts = counts / (xp.sum(counts) * (edges[1] - edges[0]))  # ruff: ignore[non-augmented-assignment]

    return counts, edges


# }}}
