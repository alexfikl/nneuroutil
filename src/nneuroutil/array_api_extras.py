# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import array_api_compat
import numpy as np

from nneuroutil.typing import Array1D, ArrayND, ScalarTypeT

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

log = logging.getLogger(__name__)


# {{{ to_dtype


def to_namespace_dtype(xp: Any, dtype: DTypeLike) -> Any:
    """Convert the :class:`numpy.dtype` *dtype* into one supported by the
    Array API backend *xp*.
    """
    if dtype is None:
        raise ValueError("'dtype' cannot be None (the default is backend specific)")

    try:
        dtype = np.dtype(dtype)
    except TypeError:
        # NOTE: this assumes that the dtype is backend specific. This can also
        # fail for just an incorrect dtype thing, but oh well..
        return dtype

    return xp.asarray(np.asarray(0, dtype=dtype)).dtype


# }}}


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


# {{{ fill_diagonal


def fill_diagonal(
    x: ArrayND[ScalarTypeT],
    value: Any,
    *,
    wrap: bool = False,
    inplace: bool = True,
    xp: Any = None,
) -> ArrayND[ScalarTypeT]:
    """Fill the main diagonal of the given array of any dimensionality.

    For an array ``x`` with ``x.ndim >= 2``, the diagonal is the list of
    values ``a[i, ..., i]`` with indices ``i`` all identical.

    With ``inplace=True`` (the default) the array is modified in place and the
    (same) array is returned. Otherwise, a copy with the modified diagonal is
    returned and *x* is left untouched.

    :arg wrap: ``wrap=True`` is not implemented.
    """
    if wrap:
        raise NotImplementedError("'wrap=True' is not implemented")

    if x.ndim < 2:
        raise ValueError(f"'x' must have at least dimension 2: {x.ndim}")

    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if inplace:
        if not array_api_compat.is_writeable_array(x):  # spell: disable
            raise ValueError(
                f"'x' is not writable (cannot use 'inplace=True'): {type(x)}"
            )
        # NOTE: is_writable_array seems to restrict the type in fun ways
        x = cast("ArrayND[ScalarTypeT]", x)

        n, m = x.shape[-2:]
        k = min(n, m)
        idx = xp.arange(0, k, device=array_api_compat.device(x))
        x[idx, idx] = value
    else:
        n, m = x.shape[-2:]
        k = min(n, m)
        rows = xp.arange(n, device=array_api_compat.device(x))[:, None]
        cols = xp.arange(m, device=array_api_compat.device(x))[None, :]
        x = xp.where(rows == cols, value, x)

    return x


# }}}


# {{{ view_as_complex


def view_as_complex(
    x: ArrayND[np.floating[Any]],
    *,
    copy: bool = False,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    """Reinterpret interleaved real storage as a complex array.

    The last dimension of *x* must have even size ``2 d`` and contain the real
    and imaginary parts interleaved as ``(Re z[0], Im z[0], ...)``. The result
    has shape ``(..., d)`` and shares memory with *x* when the last axis is
    contiguous.

    :arg copy: if *True*, allow a copy when a zero-copy view is not possible.
        If *False*, raise a :class:`ValueError` instead.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "real floating"):
        raise TypeError(f"'x' must have a real dtype: {x.dtype}")

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError(f"the last dimension of 'x' must be even: {x.shape}")

    if array_api_compat.is_torch_array(x):
        import torch

        # NOTE: this is always a view, unlike x.reshape, which could copy
        x = x.unflatten(-1, (-1, 2))

        try:
            return torch.view_as_complex(x)
        except RuntimeError as exc:
            if not copy:
                raise ValueError(
                    "'x' cannot be viewed as complex without copying (pass 'copy=True')"
                ) from exc

            return torch.view_as_complex(x.contiguous())
    elif array_api_compat.is_numpy_array(x):
        dtype = np.result_type(x.dtype, np.complex64)  # f4 -> c8, f8 -> c16

        try:
            return x.view(dtype)
        except ValueError as exc:
            if not copy:
                raise ValueError(
                    "'x' cannot be viewed as complex without copying (pass 'copy=True')"
                ) from exc

            return np.ascontiguousarray(x).view(dtype)
    elif array_api_compat.is_jax_array(x):
        import jax.numpy as jnp

        out = x.view(jnp.result_type(x.dtype, jnp.complex64))
        return jnp.array(out, copy=True) if copy else out
    else:
        if not copy:
            raise TypeError(f"zero-copy complex views are not supported for {type(x)}")

        d = x.shape[-1] // 2
        z = xp.reshape(x, (*x.shape[:-1], d, 2))

        return z[..., 0] + 1j * z[..., 1]


# }}}


# {{{ view_as_real


def view_as_real(
    z: ArrayND[np.complexfloating[Any]],
    *,
    copy: bool = False,
    xp: Any = None,
) -> ArrayND[np.floating[Any]]:
    """Reinterpret a complex array as interleaved real storage.

    The result has shape ``(..., 2 d)`` and contains the real and imaginary
    parts interleaved as ``(Re z[0], Im z[0], ...)``. It shares memory with *z*
    when the last axis is contiguous.

    :arg copy: if *True*, allow a copy when a zero-copy view is not possible.
        If *False*, raise a :class:`ValueError` instead.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(z)

    if not xp.isdtype(z.dtype, "complex floating"):
        raise TypeError(f"'z' must have a complex dtype: {z.dtype}")

    if z.ndim == 0:
        raise ValueError("'z' must have at least one dimension")

    if array_api_compat.is_torch_array(z):
        import torch

        r = torch.view_as_real(z)
        try:
            return r.view(*z.shape[:-1], -1)
        except RuntimeError as exc:
            if not copy:
                raise ValueError(
                    "'z' cannot be viewed as real without copying (pass 'copy=True')"
                ) from exc

            return r.reshape(*z.shape[:-1], -1)
    elif array_api_compat.is_numpy_array(z):
        dtype = np.finfo(z.dtype).dtype

        try:
            return z.view(dtype)
        except ValueError as exc:
            if not copy:
                raise ValueError(
                    "'z' cannot be viewed as real without copying (pass 'copy=True')"
                ) from exc

            return np.ascontiguousarray(z).view(dtype)
    elif array_api_compat.is_jax_array(z):
        import jax.numpy as jnp

        dtype = np.finfo(z.dtype).dtype
        out = z.view(dtype)

        return jnp.array(out, copy=True) if copy else out
    else:
        if not copy:
            raise TypeError(f"zero-copy real views are not supported for {type(z)}")

        d = z.shape[-1]
        return xp.reshape(
            xp.stack([xp.real(z), xp.imag(z)], axis=-1), (*z.shape[:-1], 2 * d)
        )


# }}}


# {{{ sign


def sign(
    x: ArrayND[np.number[Any]],
    /,
    *,
    xp: Any = None,
) -> ArrayND[np.number[Any]]:
    r"""The sign of *x*, with a safe complex case.

    For real-valued inputs this is exactly the ``xp.sign``. However, for
    complex values ``xp.sign`` computes ``x / abs(x)`` for all entries then
    masks out the zeros, so the division still evaluates ``0 / 0``. As a
    result, e.g. PyTorch records ``0 / 0`` in the autograd graph and returns a
    ``nan`` gradient at the origin.

    Here the denominator is replaced by one wherever ``abs(x) == 0`` before
    dividing and the zeros are selected afterwards. This avoids the issue
    and gives a zero gradient at the origin.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        return xp.sign(x)

    r = xp.abs(x)
    is_zero = r == 0
    r_safe = xp.where(is_zero, xp.ones_like(r), r)
    return xp.where(is_zero, xp.zeros_like(x), x / r_safe)


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

        if x.device.type == "cpu":
            # NOTE: torch.histogram does not work for integers, so we cast it
            if not (x.is_floating_point() or x.is_complex()):
                x = x.to(torch.get_default_dtype())

            return torch.histogram(x, bins, range=range, density=density)
        else:
            log.debug(
                "'torch.histogram' does not support %s: using fallback.", x.device
            )

    if xp is None:
        xp = array_api_compat.array_namespace(x)

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
    edges = xp.linspace(
        xmin, xmax, bins + 1, dtype=dtype, device=array_api_compat.device(x)
    )
    idx = xp.searchsorted(edges, x, side="right") - 1

    # clamp values at the last edge into the last bin, like numpy does
    idx = xp.minimum(
        idx,
        xp.asarray(bins - 1, dtype=idx.dtype, device=array_api_compat.device(x)),
    )

    # drop any elements outside of the given range
    mask = (x >= xmin) & (x <= xmax)  # ty: ignore[unsupported-operator]
    idx = xp.where(
        mask, idx, xp.asarray(bins, dtype=idx.dtype, device=array_api_compat.device(x))
    )

    # count elements per bin by sorting the bin indices and using searchsorted
    # (O(n log n)) instead of building an O(n * bins) boolean matrix
    sorted_idx = xp.sort(idx)
    boundaries = xp.searchsorted(
        sorted_idx,
        xp.arange(1, bins + 1, dtype=idx.dtype, device=array_api_compat.device(x)),
        side="left",
    )
    pad = xp.concatenate([
        xp.zeros((1,), dtype=boundaries.dtype, device=array_api_compat.device(x)),
        boundaries,
    ])
    counts = xp.diff(pad)

    if density:
        counts = counts / (xp.sum(counts) * (edges[1] - edges[0]))

    return counts, edges


# }}}
