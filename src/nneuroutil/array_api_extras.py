# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import array_api_compat
import numpy as np

from nneuroutil.helpers import register_dataclass
from nneuroutil.typing import Array, Array1D, ArrayND, DTypeLike, ScalarTypeT, ShapeT

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)


# {{{ to_dtype


def to_namespace_dtype(xp: Any, dtype: DTypeLike) -> DTypeLike:
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


# {{{ permutation


def permutation(
    rng: RandomGenerator,
    x: int | ArrayND[ScalarTypeT],
    *,
    axis: int = 0,
    xp: Any = None,
) -> ArrayND[ScalarTypeT]:
    """Return a random permutation of *x* using *rng*.

    If *x* is an integer, a permutation of ``range(x)`` is returned, otherwise
    the entries of *x* are shuffled along *axis*.

    Note that the permutation is generated by sorting uniform random values,
    so it is always a valid permutation, but can be biased by tied values when
    the floating point type of the backend is too coarse for the axis size.

    :param axis: the axis of *x* to shuffle along.
    :param rng: random generator used to draw the permutation.
    """
    if xp is None:
        ref = rng.uniform(size=(0,))
        if isinstance(x, numbers.Integral):
            xp = array_api_compat.array_namespace(ref)
        else:
            xp = array_api_compat.array_namespace(x, ref)

    if isinstance(x, numbers.Integral):
        n = int(x)
        if n < 0:
            raise ValueError(f"'x' must be non-negative integer: {x}")

        if axis not in {0, -1}:
            raise ValueError(f"'axis' is out of bounds for a permutation: {axis}")

        values = rng.uniform(size=(n,))
        return xp.argsort(values)
    else:
        assert isinstance(x, Array)
        if not -x.ndim <= axis < x.ndim:
            raise ValueError(
                f"'axis' is out of bounds for an array with {x.ndim} dimensions: {axis}"
            )

        axis = axis % x.ndim
        values = rng.uniform(size=(x.shape[axis],))

        indices = [slice(None)] * x.ndim
        indices[axis] = xp.argsort(values)

        return x[tuple(indices)]


# }}}


# {{{ generator


def default_rng(
    xp: Any, seed: int | None = None, device: Any = None
) -> RandomGenerator:
    """Create a :class:`RandomGenerator` for the Array API namespace *xp*.

    :param seed: seed used to initialize the generator.
    :param device: device used for the generated arrays, as supported by each
        backend
    """
    if not hasattr(xp, "__name__"):
        raise TypeError(f"unsupported Array API namespace: {xp}")

    if seed is None:
        # NOTE: jax and torch take seeds as uint64
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0] >> 1)

    if not isinstance(seed, numbers.Integral):
        raise TypeError(f"'seed' is not an integer type: {seed!r}")
    seed = int(seed)

    if array_api_compat.is_numpy_namespace(xp):
        return NumpyGenerator.from_seed(seed, device=device)
    elif array_api_compat.is_torch_namespace(xp):
        return TorchGenerator.from_seed(seed, device=device)
    elif array_api_compat.is_jax_namespace(xp):
        return JaxGenerator.from_seed(seed, device=device)
    else:
        raise TypeError(f"unsupported Array API namespace: {xp}")


@dataclass(frozen=True)
class RandomGenerator(ABC):
    """A backend-agnostic random number generator.

    Each generator wraps an opaque, backend-specific *key* that is created from
    a seed by :meth:`from_seed`. The key is used to draw samples and can be
    split into independent generators with :meth:`split` (e.g. for parallel or
    per-object streams) or derived for indexed streams with :meth:`fold_in`
    (e.g. one per epoch). This is heavily inspired by the :mod:`jax`
    implementation.

    The *key* and *device* are backend-specific, so generators should normally
    be created with :meth:`from_seed` (or :func:`default_rng`) instead of
    calling the constructor directly. Draws are fully reproducible for a given
    key and backend, but the streams themselves are *not* interchangeable
    between backends.
    """

    key: Any
    device: Any

    def __init__(self, key: Any, device: Any = None) -> None:
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "device", device)

    @classmethod
    @abstractmethod
    def from_seed(cls, seed: int, device: Any = None) -> Self:
        pass

    @abstractmethod
    def split(self, num: int = 2) -> tuple[Self, ...]:
        """Return *num* (strictly positive) independent generators from *key*.

        This does not modify or consume *key*, so splitting the same generator
        twice returns equivalent generators.
        """

    @abstractmethod
    def fold_in(self, index: int) -> Self:
        """Return a new generator derived from *key* and *index*.

        This is useful for deriving independent streams from a stable base key,
        such as one stream per epoch, without repeatedly splitting the
        generator. The *index* must be an integer in ``[0, 2**32)``.

        This does not modify or consume *key*.
        """

    @abstractmethod
    def integers(
        self,
        low: int,
        high: int | None = None,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.integer[Any]]]:
        """Draw random integers from ``[low, high)``.

        If *high* is ``None``, then values are drawn from ``[0, low)``. A
        *size* of ``None`` returns a zero-dimensional array and a *dtype* of
        ``None`` uses the backend's default integer type.
        """

    @abstractmethod
    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        """Draw random floats from the interval ``[low, high)``.

        A *size* of ``None`` returns a zero-dimensional array and a *dtype* of
        ``None`` uses the backend's default floating point type.
        """

    @abstractmethod
    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        """Draw normally distributed values with mean *loc* and standard
        deviation *scale*.

        A *size* of ``None`` returns a zero-dimensional array and a *dtype* of
        ``None`` uses the backend's default floating point type.
        """


# {{{ NumpyGenerator


@dataclass(frozen=True)
class SeedSequenceKey:
    __slots__ = ("entropy", "spawn_key")

    entropy: tuple[int, ...]
    spawn_key: tuple[int, ...]

    def __init__(
        self,
        entropy: int | Sequence[int] | None = None,
        spawn_key: Sequence[int] = (),
    ) -> None:
        if entropy is None:
            entropy = np.random.SeedSequence().entropy

        if isinstance(entropy, int):
            entropy = (entropy,)

        object.__setattr__(self, "entropy", tuple(entropy))  # ty: ignore[invalid-argument-type]
        object.__setattr__(self, "spawn_key", tuple(spawn_key))

    def derive(self, index: int) -> Self:
        if not isinstance(index, numbers.Integral):
            raise TypeError(f"'index' is not an integer type: {index!r}")

        index = int(index)
        if not 0 <= index < 2**32:
            raise ValueError(f"'index' must be in [0, 2**32): {index}")

        ss = np.random.SeedSequence(self.entropy, spawn_key=(*self.spawn_key, index))
        new_entropy = tuple(ss.generate_state(4, dtype=np.uint32).tolist())
        return type(self)(new_entropy)

    def split(self, num: int = 2) -> tuple[Self, ...]:
        if num <= 0:
            raise ValueError(f"'num' must be strictly positive: {num!r}")

        return tuple(self.derive(i) for i in range(num))

    def fold_in(self, index: int) -> Self:
        return self.derive(index)


@dataclass(frozen=True)
class NumpyGenerator(RandomGenerator):
    @classmethod
    def from_seed(cls, seed: int, device: Any = None) -> Self:
        if device is not None and device != "cpu":
            raise ValueError(f"'numpy' only supports a 'cpu' device: {device!r}")

        return cls(SeedSequenceKey(int(seed)), device=device)

    def split(self, num: int = 2) -> tuple[NumpyGenerator, ...]:
        return tuple(
            NumpyGenerator(key, device=self.device) for key in self.key.split(num)
        )

    def fold_in(self, index: int) -> Self:
        return type(self)(self.key.fold_in(index), device=self.device)

    @property
    def _rng(self) -> np.random.Generator:
        seed = np.random.SeedSequence(self.key.entropy, spawn_key=self.key.spawn_key)
        return np.random.Generator(np.random.PCG64(seed))

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.integer[Any]]]:
        size = cast("ShapeT", ()) if size is None else size

        if dtype is None:
            dtype = np.dtype(np.int_)
        dtype = np.dtype(dtype)

        if dtype.kind not in "iu":
            raise ValueError(f"'dtype' must be an integer dtype: {dtype}")

        if high is None:
            low, high = 0, low

        if low >= high:
            raise ValueError(f"'low' >= 'high': {low} >= {high}")

        return self._rng.integers(low=low, high=high, size=size, dtype=dtype)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        size = cast("ShapeT", ()) if size is None else size

        dtype = np.dtype(dtype)
        if dtype.kind != "f":
            raise ValueError(f"'dtype' must be a float dtype: {dtype!r}")

        result = self._rng.uniform(low, high, size)
        if dtype != result.dtype:
            result = result.astype(dtype)

        return result

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        size = cast("ShapeT", ()) if size is None else size

        dtype = np.dtype(dtype)
        if dtype.kind != "f":
            raise ValueError(f"'dtype' must be a float dtype: {dtype!r}")

        if scale <= 0.0:
            raise ValueError(f"'scale' cannot be negative: {scale}")

        result = self._rng.normal(loc, scale, size)
        if dtype != result.dtype:
            result = result.astype(dtype)

        return result


# }}}


# {{{ JaxGenerator


@register_dataclass
@dataclass(frozen=True)
class JaxGenerator(RandomGenerator):
    def __hash__(self) -> int:
        import jax

        # NOTE: self.key is not hashable, so we implement our own
        return hash((
            jax.random.key_data(self.key).tobytes(),
            str(self.key.dtype),
            self.device,
        ))

    @classmethod
    def from_seed(cls, seed: int, device: Any = None) -> Self:
        import jax

        if device is not None:
            raise ValueError(f"'jax' does not take a device: {device!r}")

        return cls(jax.random.key(int(seed)), device=device)

    def split(self, num: int = 2) -> tuple[JaxGenerator, ...]:
        if num <= 0:
            raise ValueError(f"'num' must be strictly positive: {num!r}")

        import jax.random

        return tuple(
            JaxGenerator(key, device=self.device)
            for key in jax.random.split(self.key, num=num)
        )

    def fold_in(self, index: int) -> Self:
        if not isinstance(index, numbers.Integral):
            raise TypeError(f"'index' is not an integer type: {index!r}")

        index = int(index)
        if not 0 <= index < 2**32:
            raise ValueError(f"'index' must be in [0, 2**32): {index}")

        import jax.random

        return type(self)(jax.random.fold_in(self.key, index), device=self.device)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.integer[Any]]]:
        import jax.numpy as jnp
        from jax.random import randint

        size = cast("ShapeT", ()) if size is None else size

        if dtype is None:
            dtype = jnp.array(0).dtype
        dtype = np.dtype(dtype)

        if dtype.kind not in "iu":
            raise ValueError(f"'dtype' must be an integer dtype: {dtype}")

        if high is None:
            low, high = 0, low

        if low >= high:
            raise ValueError(f"'low' >= 'high': {low} >= {high}")

        return randint(self.key, shape=size, minval=low, maxval=high, dtype=dtype)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        import jax.numpy as jnp
        from jax.random import uniform

        size = cast("ShapeT", ()) if size is None else size
        if dtype is None:
            dtype = jnp.array(0.0).dtype
        dtype = np.dtype(dtype)

        if dtype.kind != "f":
            raise ValueError(f"'dtype' must be a float dtype: {dtype!r}")

        if low >= high:
            raise ValueError(f"'low' >= 'high': {low} >= {high}")

        return uniform(self.key, shape=size, minval=low, maxval=high, dtype=dtype)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        import jax.numpy as jnp
        from jax.random import normal

        size = cast("ShapeT", ()) if size is None else size

        if dtype is None:
            dtype = jnp.array(0.0).dtype
        dtype = np.dtype(dtype)

        if dtype.kind != "f":
            raise ValueError(f"'dtype' must be a float dtype: {dtype!r}")

        if scale <= 0.0:
            raise ValueError(f"'scale' cannot be negative: {scale}")

        result = normal(self.key, shape=size, dtype=dtype)
        return loc + scale * result


# }}}


# {{{ TorchGenerator


@dataclass(frozen=True)
class TorchGenerator(RandomGenerator):
    @classmethod
    def from_seed(cls, seed: int, device: Any = None) -> Self:
        return cls(SeedSequenceKey(int(seed)), device=device)

    def split(self, num: int = 2) -> tuple[TorchGenerator, ...]:
        return tuple(
            TorchGenerator(key, device=self.device) for key in self.key.split(num)
        )

    def fold_in(self, index: int) -> Self:
        return type(self)(self.key.fold_in(index), device=self.device)

    @property
    def _rng(self) -> Any:
        import torch

        ss = np.random.SeedSequence(self.key.entropy, spawn_key=self.key.spawn_key)
        seed = ss.generate_state(1, dtype=np.uint64).item() & (2**63 - 1)
        return torch.Generator(device=self.device).manual_seed(seed)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.integer[Any]]]:
        import torch

        size = cast("ShapeT", ()) if size is None else size

        if dtype is None:
            dtype = torch.int64

        if (
            not isinstance(dtype, torch.dtype)
            or dtype.is_floating_point
            or dtype.is_complex
            or dtype == torch.bool
        ):
            raise ValueError(f"'dtype' must be an integer dtype: {dtype!r}")

        if high is None:
            low, high = 0, low

        if low >= high:
            raise ValueError(f"'low' >= 'high': {low} >= {high}")

        return torch.randint(
            low, high, size, generator=self._rng, dtype=dtype, device=self.device
        )

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        import torch

        size = cast("ShapeT", ()) if size is None else size
        if dtype is None:
            dtype = torch.zeros(0).dtype

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"'dtype' must be a float dtype: {dtype!r}")

        if low >= high:
            raise ValueError(f"'low' >= 'high': {low} >= {high}")

        result = torch.rand(size, dtype=dtype, generator=self._rng, device=self.device)
        return low + (high - low) * result

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: ShapeT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> Array[ShapeT, np.dtype[np.floating[Any]]]:
        import torch

        size = cast("ShapeT", ()) if size is None else size
        if dtype is None:
            dtype = torch.zeros(0).dtype

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"'dtype' must be a float dtype: {dtype!r}")

        if scale <= 0.0:
            raise ValueError(f"'scale' cannot be negative: {scale}")

        result = torch.randn(size, dtype=dtype, generator=self._rng, device=self.device)
        return loc + scale * result


# }}}

# }}}
