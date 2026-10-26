# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""Backend-agnostic activation functions.

All functions operate on arrays from any backend that implements the `Array
API Standard <https://data-apis.org/array-api/latest>`__ (e.g. ``numpy``,
``torch`` or ``jax``) and are dispatched using :mod:`array_api_compat`.

Complex-valued functions expect arrays with a complex data type (e.g.
``complex64`` or ``complex128``). Real-valued storage can be converted by the
caller, for example with :func:`torch.view_as_complex` for an interleaved
layout or by assembling the real and imaginary parts into a complex array.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import array_api_compat
import numpy as np

from nneuroutil.array_api_extras import sign, view_as_complex, view_as_real
from nneuroutil.typing import Array, ArrayND, ScalarTypeT

log = logging.getLogger(__name__)


# {{{ real activations


def quadratic(x: ArrayND[np.floating[Any]]) -> ArrayND[np.floating[Any]]:
    r"""The quadratic activation :math:`x^2`.

    .. warning::

        The quadratic activation is not "depth-stable", meaning in a deep
        network it will eventually explode the variance of the inputs.
    """
    return x * x


def blended_quadratic(
    x: ArrayND[np.floating[Any]],
    *,
    alpha: float | ArrayND[np.floating[Any]] = 0.5,
) -> ArrayND[np.floating[Any]]:
    r"""A convex combination of a linear and a quadratic activation.

    .. math::

        f(x; \alpha) = \alpha x + (1 - \alpha) x^2.

    .. warning::

        The blended quadratic activation is not "depth-stable", meaning in a
        deep network it will eventually explode the variance of the inputs. The
        number of layers allowed before divergence can be increased by taking
        :math:`\alpha \to 1`.
    """
    return alpha * x + (1.0 - alpha) * x * x


# }}}


# {{{ complex storage


def complex_from_real(
    x: ArrayND[np.floating[Any]],
    *,
    interleaved: bool = False,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    """Convert real-valued complex storage into a complex array.

    The last dimension of *x* must have even size ``2 d``.

    * If *interleaved*, the real and imaginary parts are stored as ``(Re z[0],
      Im z[0], ...)`` and the conversion is a zero-copy view.
    * Otherwise, they are stored as ``(Re z[0], ..., Re z[d - 1], Im z[0], ...,
      Im z[d - 1])`` and the conversion allocates a new complex array.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "real floating"):
        raise TypeError(f"'x' must have a real dtype: {x.dtype}")

    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError(f"the last dimension of 'x' must be even: {x.shape}")

    if interleaved:
        return view_as_complex(x, copy=False, xp=xp)
    else:
        d = x.shape[-1] // 2
        return x[..., :d] + 1j * x[..., d:]


def real_from_complex(
    z: ArrayND[np.complexfloating[Any]],
    *,
    interleaved: bool = False,
    xp: Any = None,
) -> ArrayND[np.floating[Any]]:
    """Convert a complex array into real-valued complex storage.

    This is the inverse of :func:`complex_from_real`. If *interleaved* is
    *True*, the conversion is a zero-copy view. Otherwise, it allocates a new
    real array.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(z)

    if not xp.isdtype(z.dtype, "complex floating"):
        raise TypeError(f"'z' must have a complex dtype: {z.dtype}")

    if interleaved:
        return view_as_real(z, copy=False, xp=xp)
    else:
        return xp.concat([xp.real(z), xp.imag(z)], axis=-1)


def apply_complex(
    fn: Callable[[ArrayND[ScalarTypeT]], ArrayND[ScalarTypeT]],
    x: ArrayND[ScalarTypeT],
    *,
    interleaved: bool = False,
    xp: Any = None,
) -> ArrayND[ScalarTypeT]:
    """Apply a complex-valued activation *fn* to *x*.

    If *x* is already a complex array, it is passed to *fn* directly and the
    complex result is returned. Otherwise, *x* is converted with
    :func:`complex_from_real`, *fn* is applied, and the result is converted back
    with :func:`real_from_complex`, preserving the original real storage layout.

    *fn* is called with the complex array as its only positional argument. Bind
    any additional parameters with :func:`functools.partial`.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if xp.isdtype(x.dtype, "complex floating"):
        return fn(x)

    z = complex_from_real(x, interleaved=interleaved, xp=xp)
    return real_from_complex(fn(z), interleaved=interleaved, xp=xp)


# }}}


# {{{ complex activations


def complex_quadratic(
    x: ArrayND[np.complexfloating[Any]],
    *,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""Compute the complex square :math:`x^2` of a complex array *x*.

    .. warning::

        The quadratic activation is not "depth-stable", meaning in a deep
        network it will eventually explode the variance of the inputs.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    return x * x


def complex_blended_quadratic(
    x: ArrayND[np.complexfloating[Any]],
    *,
    alpha: float | ArrayND[np.floating[Any]] = 0.5,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""A convex combination of a linear and a quadratic activation for complex
    arrays.

    .. math::

        f(z; \alpha) = \alpha z + (1 - \alpha) z^2,

    :arg alpha: a scalar or an array with one entry per complex channel (i.e.
        of size ``d`` for an input with last dimension ``d``).
    """

    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    # NOTE: we need to make sure alpha broadcasts correctly to the complex variable
    if array_api_compat.is_array_api_obj(alpha):
        assert isinstance(alpha, Array)
        if alpha.ndim != 0 and alpha.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"'alpha' last dimension ({alpha.shape[-1]}) does not match input "
                f"features ({x.shape[-1]})"
            )

    return alpha * x + (1.0 - alpha) * x * x


def complex_tanh(
    x: ArrayND[np.complexfloating[Any]],
    *,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""A split hyperbolic tangent for complex arrays.

    .. math::

        f(z) = \tanh(\Re z) + i \tanh(\Im z).

    The hyperbolic tangent is applied to the real and imaginary components
    separately, so this is different from the analytic :math:`\tanh(z)` for a
    complex :math:`z`.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    return xp.tanh(xp.real(x)) + 1j * xp.tanh(xp.imag(x))


def modrelu(
    x: ArrayND[np.complexfloating[Any]],
    b: float | ArrayND[np.floating[Any]] = 0.0,
    *,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""The modReLU activation function for complex arrays from [Arjovsky2015]_.

    .. math::

        f(z; b) = \operatorname{ReLU}(|z| + b) \operatorname{sgn}(z),

    where :math:`\operatorname{sgn}(z) = z / |z|` (taken to be zero at the
    origin) preserves the phase of the input. For :math:`b < 0`, all inputs in
    the disk :math:`|z| < -b` are zeroed out, analogously to the standard
    :class:`~torch.nn.ReLU`. For :math:`b \ge 0`, the function reduces to
    :math:`f(z) = z + b \operatorname{sgn}(z)`.

    .. warning::

        Note that for :math:`b < 0`, this activation function is not
        "depth-stable", meaning that in a deep network it will eventually
        explode the variance of the inputs.

    .. [Arjovsky2015] M. Arjovsky, A. Shah, Y. Bengio,
        *Unitary Evolution Recurrent Neural Networks*,
        2015,
        `URL <http://arxiv.org/abs/1511.06464v4>`__.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    r = xp.abs(x)
    m = xp.maximum(r + b, xp.zeros_like(r))
    return m * sign(x, xp=xp)


def leaky_modrelu(
    x: ArrayND[np.complexfloating[Any]],
    b: float | ArrayND[np.floating[Any]] = 0.0,
    *,
    alpha: float = 0.1,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""A leaky variant of the :func:`modrelu` activation function.

    .. math::

        f(z; b, \alpha) = \operatorname{sgn}(z) \times
        \begin{cases}
        |z| + b, & \quad |z| + b \ge 0, \\
        \alpha |z|, & \quad \text{otherwise}.
        \end{cases}

    Note that this is not the same as applying a standard leaky ReLU to the
    :math:`|z| + b` term of :func:`modrelu`, which would give the negative
    magnitude :math:`\alpha (|z| + b) < 0` and thereby flip the phase by
    :math:`\pi`. Here the magnitude itself is scaled by :math:`\alpha`, so the
    phase is always maintained. The trade-off is a jump discontinuity of size
    :math:`\alpha |b|` at :math:`|z| = -b` when :math:`b < 0`. For :math:`b \ge
    0`, the leaky branch never triggers and the function is identical to
    :func:`modrelu`.

    .. warning::

        Note that for :math:`b < 0`, this activation function is not
        "depth-stable", meaning that in a deep network it will eventually
        explode the variance of the inputs.
    """

    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    r = xp.abs(x)
    rb = r + b
    m = xp.where(rb >= 0.0, rb, alpha * r)
    return m * sign(x, xp=xp)


def complex_cardioid(
    x: ArrayND[np.complexfloating[Any]],
    *,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""A complex cardioid activation function from [Virtue2017]_.

    .. math::

        f(z) = \frac{1}{2} (1 + \cos \theta(z)) z,

    where :math:`\theta(z) = \operatorname{arg} z` is the phase of the input.
    The magnitude is attenuated based on the phase, while the phase itself is
    preserved: inputs on the positive real axis are kept as is and inputs on
    the negative real axis are zeroed out.

    At the origin, the scale takes its limiting value :math:`1/2`.

    .. [Virtue2017] P. Virtue, S. X. Yu, M. Lustig,
        *Better Than Real: Complex-Valued Neural Nets for MRI Fingerprinting*,
        2017,
        `URL <https://arxiv.org/abs/1707.00070>`__.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    r = xp.abs(x)
    r_safe = xp.where(r == 0, 1.0, r)
    scale = 0.5 * (1.0 + xp.real(x) / r_safe)
    return scale * x


def zrelu(
    x: ArrayND[np.complexfloating[Any]],
    *,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""A complex ReLU function from [Guberman2016]_ that keeps the first quadrant.

    .. math::

        f(z) =
        \begin{cases}
        z, & \quad \Re(z) \geq 0, \Im(z) \geq 0, \\
        0, & \quad \text{otherwise}.
        \end{cases}

    Inputs with a phase outside of :math:`[0, \pi / 2]` are zeroed out.

    .. [Guberman2016] N. Guberman,
        *On Complex Valued Convolutional Neural Networks*,
        2016,
        `URL <https://arxiv.org/abs/1602.09046>`__.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    mask = (xp.real(x) >= 0.0) & (xp.imag(x) >= 0.0)
    return xp.where(mask, x, xp.zeros_like(x))


def complex_log(
    x: ArrayND[np.complexfloating[Any]],
    *,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""A phase-preserving logarithmic activation for complex arrays.

    .. math::

        f(z) = \frac{\log(1 + |z|)}{|z|} z,

    which is equivalent to :math:`\log(1 + |z|) \operatorname{sgn}(z)` for
    :math:`z \neq 0` and has the continuous extension :math:`f(0) = 0` with
    :math:`f'(0) = 1`.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)
    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    r = xp.abs(x)
    is_zero = r == 0
    r_safe = xp.where(is_zero, 1.0, r)
    scale = xp.where(is_zero, 1.0, xp.log1p(r) / r_safe)
    return scale * x


def complex_exp(
    x: ArrayND[np.complexfloating[Any]],
    *,
    xp: Any = None,
) -> ArrayND[np.complexfloating[Any]]:
    r"""A phase-preserving exponential activation for complex arrays.

    .. math::

        f(z) = \frac{e^{|z|} - 1}{|z|} z,

    which is equivalent to :math:`(e^{|z|} - 1) \operatorname{sgn}(z)` for
    :math:`z \neq 0` and has the continuous extension :math:`f(0) = 0` with
    :math:`f'(0) = 1`.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    if not xp.isdtype(x.dtype, "complex floating"):
        raise TypeError(f"'x' must have a complex dtype: {x.dtype}")

    r = xp.abs(x)
    is_zero = r == 0
    r_safe = xp.where(is_zero, 1.0, r)
    scale = xp.where(is_zero, 1.0, xp.expm1(r) / r_safe)
    return scale * x


# }}}
