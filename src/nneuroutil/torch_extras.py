# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeVar

import torch
import torch.utils._pytree as pytree  # ruff:ignore[import-private-name]
from torch import nn
from torch.utils._python_dispatch import (  # ruff:ignore[import-private-name]
    TorchDispatchMode,
)

from nneuroutil.helpers import MemorySnapshot, MemoryTracker, module_logger

log = module_logger(__name__)


# {{{ activation functions


def view_as_complex(x: torch.Tensor) -> torch.Tensor:
    """View a real tensor of shape ``(..., 2d)`` as an interleaved complex tensor.

    This is mostly :func:`torch.view_as_complex` with a reshape.
    """
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())


def view_as_real(x: torch.Tensor) -> torch.Tensor:
    """View a complex tensor of shape ``(..., d)`` as an interleaved real tensor.

    This is mostly :func:`torch.view_as_real` with a reshape.
    """
    return torch.view_as_real(x).reshape(*x.shape[:-1], -1)


class Quadratic(nn.Module):
    """A quadratic :math:`x^2` activation function.

    Note that a quadratic activation is not "depth-stable", meaning in a deep
    network it will eventually explode the variance of the inputs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ruff:ignore[no-self-use]
        """Define the computation performed at every call."""
        return x * x


class ComplexQuadratic(nn.Module):
    r"""A quadratic activation :math:`f(z) = z^2` for complex values stored in
    real tensors (see :func:`cquadratic`).

    Note that a quadratic activation is not "depth-stable", meaning in a deep
    network it will eventually explode the variance of the inputs.
    """

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return cquadratic(x, interleaved=self.interleaved)


def cquadratic(x: torch.Tensor, *, interleaved: bool = False) -> torch.Tensor:
    r"""Compute the complex square of a real vector *x*.

    The storage of the array depends on *interleaved*. If *True*, we assume that
    the array is stored as ``[z[0].real, z[0].imag, ..., z[d - 1].real,
    z[d - 1].imag]``. Otherwise, we assume that it is stacked as
    ``[z[0].real, ..., z[d - 1].real, z[0].imag, ..., z[d - 1].imag]``.

    The result uses the same storage as the input.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"dimension 'd' of 'x[..., d]' must be even: {x.shape}")

    if interleaved:
        z = view_as_complex(x)
        result = torch.view_as_real(z * z).reshape(*x.shape)
    else:
        x_re, x_im = torch.chunk(x, 2, dim=-1)
        result = torch.concat([x_re * x_re - x_im * x_im, 2 * x_re * x_im], dim=-1)

    return result


class BlendedQuadratic(nn.Module):
    r"""A convex combination of a linear and a quadratic activation.

    .. math::

        f(x; \alpha) = \alpha x + (1 - \alpha) x^2.

    Note that a quadratic activation is not "depth-stable", meaning in a deep
    network it will eventually explode the variance of the inputs. For the
    blended case, setting :math:`\alpha` close to 1 will delay the explosion,
    allowing for deeper networks, but it does not solve the issue.
    """

    alpha: float
    """A non-learnable hyperparameter with values in :math:`[0, 1]`."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"'alpha' must be in [0, 1]: {alpha}")

        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return self.alpha * x + (1.0 - self.alpha) * x * x


class ComplexBlendedQuadratic(nn.Module):
    r"""A convex combination of a linear and a quadratic activation, where the
    quadratic term uses :func:`cquadratic`.

    .. math::

        f(z; \alpha) = \alpha z + (1 - \alpha) z^2.

    Note that a quadratic activation is not "depth-stable", meaning in a deep
    network it will eventually explode the variance of the inputs. For the
    blended case, setting :math:`\alpha` close to 1 will delay the explosion,
    allowing for deeper networks, but it does not solve the issue.
    """

    alpha: float
    """A non-learnable hyperparameter with values in :math:`[0, 1]`."""
    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, alpha: float = 0.5, *, interleaved: bool = False) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"'alpha' must be in [0, 1]: {alpha}")

        super().__init__()
        self.alpha = alpha
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return cblended_quadratic(x, interleaved=self.interleaved, alpha=self.alpha)


def cblended_quadratic(
    x: torch.Tensor,
    *,
    interleaved: bool = False,
    alpha: float = 0.5,
) -> torch.Tensor:
    x2 = cquadratic(x, interleaved=interleaved)
    return alpha * x + (1.0 - alpha) * x2


class ComplexTanh(nn.Module):
    r"""A split hyperbolic tangent for complex tensors.

    .. math::

        f(z) = \tanh(\Re z) + i \tanh(\Im z)

    The hyperbolic tangent is applied to the real and imaginary components
    separately, so this is different from the analytic :math:`\tanh(z)` for a
    complex :math:`z`. Unlike the other activations in this module, the input
    is expected to be a complex tensor; real inputs are promoted to complex
    tensors with a zero imaginary part.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ruff:ignore[no-self-use]
        """Define the computation performed at every call."""
        if x.is_complex():
            return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))
        else:
            return torch.complex(torch.tanh(x), torch.zeros_like(x))


class ModReLU(nn.Module):
    r"""A modified ReLU activation function for complex networks from [Arjovsky2015]_.

    .. math::

        f(z; b) = \operatorname{ReLU}(|z| + b) \operatorname{sgn}(z),

    where :math:`\operatorname{sgn}(z) = z / |z|` (taken to be zero at the
    origin) preserves the phase of the input. For :math:`b < 0`, all inputs in
    the disk :math:`|z| < -b` are zeroed out, analogously to the standard
    :class:`~torch.nn.ReLU`. For :math:`b \ge 0`, the function reduces to
    :math:`f(z) = z + b \operatorname{sgn}(z)`.

    Note that for :math:`b < 0`, this activation function is not "depth-stable",
    meaning that in a deep network it will eventually explode the variance. The
    growth per layer is small (2% or so for :math:`b \sim -1`), but it compounds
    with depth and can still cause problems. This is not an issue for
    :math:`b \ge 0`.

    .. [Arjovsky2015] M. Arjovsky, A. Shah, Y. Bengio,
        *Unitary Evolution Recurrent Neural Networks*,
        2015,
        `URL <http://arxiv.org/abs/1511.06464v4>`__.
    """

    bias: float
    """The bias :math:`b`. A non-learnable hyperparameter."""
    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, bias: float = 0.0, *, interleaved: bool = False) -> None:
        super().__init__()
        self.bias = bias
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        if self.interleaved:
            return view_as_real(modrelu(view_as_complex(x), self.bias))
        else:
            result = modrelu(torch.complex(*torch.chunk(x, 2, dim=-1)), self.bias)
            return torch.cat([result.real, result.imag], dim=-1)


def modrelu(z: torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    """Functional version of :class:`ModReLU` for a complex tensor *z*."""
    return torch.relu(torch.abs(z) + b) * torch.sgn(z)


class LeakyModReLU(nn.Module):
    r"""A leaky variant of the :class:`ModReLU` activation function.

    .. math::

        f(z; b, \alpha) = \operatorname{sgn}(z) \times
        \begin{cases}
        |z| + b, & \quad |z| + b \ge 0, \\
        \alpha |z|, & \quad \text{otherwise}.
        \end{cases}

    Note that this is not the same as applying a standard
    :class:`~torch.nn.LeakyReLU` to the :math:`|z| + b` term of :class:`ModReLU`,
    which would give the negative magnitude :math:`\alpha (|z| + b) < 0` and
    thereby flip the phase by :math:`\pi`. Here the magnitude itself is scaled
    by :math:`\alpha`, so the phase is always maintained. The trade-off is a
    jump discontinuity of size :math:`\alpha |b|` at :math:`|z| = -b` when
    :math:`b < 0`. For :math:`b \ge 0`, the leaky branch never triggers and the
    function is identical to :class:`ModReLU`.

    Note that for :math:`b < 0`, this activation function is not "depth-stable",
    meaning that in a deep network it will eventually explode the variance. The
    leak only enters the signal statistics at :math:`O(\alpha^2)`, so it does
    not fix this instability. This is not an issue for :math:`b \ge 0`.
    """

    bias: float
    """The bias :math:`b`. A non-learnable hyperparameter."""
    alpha: float
    r"""The leak slope :math:`\alpha`, with values in :math:`[0, 1]`. A
    non-learnable hyperparameter."""
    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(
        self,
        bias: float = 0.0,
        *,
        alpha: float = 0.1,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.bias = bias
        self.alpha = alpha
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        if self.interleaved:
            return view_as_real(
                leaky_modrelu(view_as_complex(x), self.bias, alpha=self.alpha)
            )
        else:
            result = leaky_modrelu(
                torch.complex(*torch.chunk(x, 2, dim=-1)), self.bias, alpha=self.alpha
            )
            return torch.cat([result.real, result.imag], dim=-1)


def leaky_modrelu(
    z: torch.Tensor,
    b: float | torch.Tensor,
    *,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Functional version of :class:`LeakyModReLU` for a complex tensor *z*."""
    r = torch.abs(z)
    rb = r + b
    return torch.where(rb >= 0.0, rb, alpha * r) * torch.sgn(z)


class ComplexCardioid(nn.Module):
    r"""A complex cardioid activation function from [Virtue2017]_.

    .. math::

        f(z) = \frac{1}{2} (1 + \cos \theta(z)) z,

    where :math:`\theta(z) = \operatorname{arg} z` is the phase of the input.
    The magnitude is attenuated based on the phase, while the phase itself is
    preserved: inputs on the positive real axis are kept as is and inputs on
    the negative real axis are zeroed out. For real inputs, this reduces to
    the standard :class:`~torch.nn.ReLU`.

    .. [Virtue2017] P. Virtue, S. X. Yu, M. Lustig,
        *Better Than Real: Complex-Valued Neural Nets for MRI Fingerprinting*,
        2017,
        `URL <https://arxiv.org/abs/1707.00070>`__.
    """

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        if self.interleaved:
            return view_as_real(ccardioid(view_as_complex(x)))
        else:
            result = ccardioid(torch.complex(*torch.chunk(x, 2, dim=-1)))
            return torch.cat([result.real, result.imag], dim=-1)


def ccardioid(z: torch.Tensor) -> torch.Tensor:
    """Functional version of :class:`ComplexCardioid` for a complex tensor *z*."""
    return 0.5 * (1 + torch.cos(torch.angle(z))) * z


class zReLU(nn.Module):  # ruff:ignore[invalid-class-name]
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

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        if self.interleaved:
            return view_as_real(zrelu(view_as_complex(x)))
        else:
            result = zrelu(torch.complex(*torch.chunk(x, 2, dim=-1)))
            return torch.cat([result.real, result.imag], dim=-1)


def zrelu(z: torch.Tensor) -> torch.Tensor:
    """Functional version of :class:`zReLU` for a complex tensor *z*."""
    return z * ((z.real >= 0.0) & (z.imag >= 0)).to(z.real.dtype)


# }}}


# {{{ layers


class Bias(nn.Module):
    """A bias-only layer: :math:`y = x + b`."""

    size: int
    """The size of the bias vector."""
    bias: torch.Tensor
    """The learnable bias of shape ``(size,)``. The values are initialized to zero."""

    def __init__(
        self,
        size: int,
        *,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()

        self.size = size
        self.bias = nn.Parameter(torch.zeros(size, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return x + self.bias


class Residual(nn.Module):
    """A basic residual layer around a given *model*."""

    model: nn.Module
    """The wrapped model."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return self.model(x) + x


class ComplexLinear(nn.Module):
    """A linear layer applied to stacked complex variables.

    We assume that the input vector is of shape ``(2 d,)`` representing a
    stacked complex vector of shape ``(d,)``. This linear layer treats it as such
    and computes

    .. code:: python

        y = torch.cat([x[..., :d] @ W.T + b_re, x[..., d:] @ W.T + b_im], dim=-1)
    """

    in_features: int
    """Size of each input sample."""
    out_features: int
    """Size of each output sample."""
    weight: torch.Tensor
    """The learnable weights of the module of shape ``(out_features, in_features)``.
    The values are initialized using :func:`kaiming_uniform_`.
    """
    bias_re: torch.Tensor
    """The learnable bias of shape ``(out_features,)`` for the real part of the
    input vector. The values are initialized to 0.
    """
    bias_im: torch.Tensor
    """The learnable bias of shape ``(out_features,)`` for the imaginary part of the
    input vector. The values are initialized to 0.
    """
    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        interleaved: bool = False,
        bias: bool = False,
        device: str | torch.device | None = None,
        dtype: Any | None = None,
    ) -> None:
        super().__init__()

        ftype = None if dtype is None else dtype.to_real()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=ftype)
        )
        if bias:
            self.bias_re = nn.Parameter(
                torch.empty(out_features, device=device, dtype=ftype)
            )
            self.bias_im = nn.Parameter(
                torch.empty(out_features, device=device, dtype=ftype)
            )
        else:
            self.bias_re = self.bias_im = None

        self.in_features = in_features
        self.out_features = out_features
        self.interleaved = interleaved

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaiming uniform (same as nn.Linear default) as a safe fallback
        nn.init.kaiming_uniform_(self.weight, math.sqrt(5))
        if self.bias_re is not None:
            nn.init.zeros_(self.bias_re)
        if self.bias_im is not None:
            nn.init.zeros_(self.bias_im)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        # x: [batch, 2 * in_features]
        if self.interleaved:
            z = view_as_complex(x)

            bias = None
            if self.bias_re is not None:
                bias = torch.complex(self.bias_re, self.bias_im)

            result = nn.functional.linear(z, self.weight.to(z.dtype), bias)
            return view_as_real(result)
        else:
            x_re, x_im = x.chunk(2, dim=-1)

            result_re = nn.functional.linear(x_re, self.weight, self.bias_re)
            result_im = nn.functional.linear(x_im, self.weight, self.bias_im)

            return torch.cat([result_re, result_im], dim=-1)


class SymmetricLinear(nn.Module):
    """A symmetric linear layer."""

    in_features: int
    """Size of each input sample."""
    out_features: int
    """Size of each output sample."""

    bias: torch.Tensor | None
    """The learnable bias of shape ``(out_features,)``. The values are
    initialized to 0.
    """

    def __init__(
        self,
        features: int,
        *,
        bias: bool = False,
        device: str | torch.device | None = None,
        dtype: Any | None = None,
    ) -> None:
        super().__init__()

        self.in_features = features
        self.out_features = features

        self._weight = nn.Parameter(
            torch.empty(features, features, device=device, dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(features, device=device, dtype=dtype))
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        """The learnable weights of the module of shape ``(out_features, in_features)``.
        The values are initialized using :func:`kaiming_uniform_`.
        """
        return self._weight.triu() + self._weight.triu().transpose(-1, -2)

    def reset_parameters(self) -> None:
        # NOTE: this is the same init as nn.Linear. Turns out that symmetrizing
        # it doesn't change the row-wise variances, so the same init works here
        nn.init.kaiming_uniform_(self._weight, math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return nn.functional.linear(x, self.weight, self.bias)


# }}}


# {{{ init

NONLINEARITY_TYPE_NAME = {
    BlendedQuadratic: "blended_quadratic",
    ComplexBlendedQuadratic: "cblended_quadratic",
    ComplexCardioid: "ccardioid",
    ComplexQuadratic: "cquadratic",
    ComplexTanh: "ctanh",
    LeakyModReLU: "leaky_modrelu",
    ModReLU: "modrelu",
    Quadratic: "quadratic",
    zReLU: "zrelu",
}


def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    atol: float = 1.0e-6,
) -> float:
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a) and f(b) must have opposite signs: {fa} and {fb}")

    while 0.5 * (b - a) > atol:
        m = (a + b) / 2.0
        fm = f(m)
        if fa * fm <= 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm

    return (a + b) / 2.0


def complex_kaiming_uniform_(
    x: torch.Tensor,
    *,
    nonlinearity: str = "modrelu",
    param: float | None = None,
    paramb: float | None = None,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(x)
    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity == "cquadratic":
        bound = math.sqrt(3) / math.sqrt(2 * fan)
    elif nonlinearity == "cblended_quadratic":
        # NOTE: if not given, use the default from BlendedQuadratic
        if param is None:
            param = 0.5

        if abs(param - 1.0) < 1.0e-8:
            var_x = 1.0 / fan
        else:
            b = param**2
            a = 4 * (1 - param) ** 2
            var_x = (-b + math.sqrt(b**2 + 4 * a)) / (2 * a * fan)

        bound = math.sqrt(3 * var_x)
    elif nonlinearity == "modrelu":
        if param is None:
            param = 0.0

        b = param
        if b > math.sqrt(2):
            raise ValueError(f"No solution exists for b>=sqrt(2): {b}")

        if b >= 0:
            var_w = 0.5 * (
                math.sqrt(b**2 * math.pi / 2.0 + 4 - 2 * b**2)
                - b * math.sqrt(math.pi / 2.0)
            )
        else:
            # NOTE: these seems to be no exact solution for this case, so we just
            # bisect it. Theoretically this doesn't happen a lot of times, so
            # it should be fine to do a slower algorithm
            result = bisect(
                lambda s: (
                    s * math.exp(-(b**2) / (2 * s))
                    + b * math.sqrt(math.pi * s / 2) * math.erfc(-b / math.sqrt(2 * s))
                    - 1
                ),
                1.0e-6,
                max(100.0, b**2 + 10),
            )
            var_w = math.sqrt(result)

        bound = math.sqrt(3) * var_w / math.sqrt(fan)
    elif nonlinearity == "leaky_modrelu":
        if param is None:
            param = 0.0

        if paramb is None:
            paramb = 0.1

        b = param
        alpha = paramb

        if b >= 0:
            # NOTE: in this case, the leaky branch never actually triggers, so
            # there is nothing extra to do compared to standard modReLU
            var_w = 0.5 * (
                math.sqrt(b**2 * math.pi / 2.0 + 4 - 2 * b**2)
                - b * math.sqrt(math.pi / 2.0)
            )
        else:
            result = bisect(
                lambda s: (
                    s * math.exp(-(b**2) / (2 * s))
                    + b * math.sqrt(math.pi * s / 2) * math.erfc(-b / math.sqrt(2 * s))
                    + alpha**2 * (s - (s + b**2 / 2) * math.exp(-(b**2) / (2 * s)))
                    - 1
                ),
                1.0e-6,
                max(100.0, b**2 + 10),
            )
            var_w = math.sqrt(result)

        bound = math.sqrt(3) * var_w / math.sqrt(fan)
    elif nonlinearity == "ccardioid":
        bound = math.sqrt(3) / math.sqrt(0.375 * fan)
    elif nonlinearity == "zrelu":
        bound = math.sqrt(12) / math.sqrt(fan)
    else:
        raise ValueError(f"unknown nonlinearity: {nonlinearity!r}")

    return nn.init.uniform_(x, -bound, bound, generator=generator)


def kaiming_uniform_(
    x: torch.Tensor,
    *,
    nonlinearity: str = "quadratic",
    param: float | None = None,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A wrapper around :func:`torch.nn.init.kaiming_uniform_` that supports
    our activation functions.
    """

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(x)
    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity == "quadratic":
        bound = math.sqrt(3) / math.sqrt(math.sqrt(3) * fan)
        return nn.init.uniform_(x, -bound, bound, generator=generator)
    elif nonlinearity == "blended_quadratic":
        # NOTE: if not given, use the default from BlendedQuadratic
        if param is None:
            param = 0.5

        if abs(param - 1.0) < 1.0e-8:
            var_x = 1.0 / fan
        else:
            b = param**2
            a = 3 * (1 - param) ** 2
            var_x = (-b + math.sqrt(b**2 + 4 * a)) / (2 * a * fan)

        bound = math.sqrt(3 * var_x)
        return nn.init.uniform_(x, -bound, bound, generator=generator)
    else:
        # NOTE: this is the default for kaiming_uniform_
        if param is None:
            param = 0.0

        return nn.init.kaiming_uniform_(
            x,
            a=param,
            mode=mode,
            nonlinearity=nonlinearity,
            generator=generator,
        )


def kaiming_normal_(
    x: torch.Tensor,
    *,
    nonlinearity: str = "quadratic",
    param: float | None = None,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A wrapper around :func:`torch.nn.init.kaiming_normal_` that supports
    our activation functions.
    """

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(x)
    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity == "quadratic":
        std = 1.0 / math.sqrt(math.sqrt(3) * fan)
        return nn.init.normal_(x, 0.0, std, generator=generator)
    elif nonlinearity == "blended_quadratic":
        # NOTE: if not given, use the default from BlendedQuadratic
        if param is None:
            param = 0.5

        if abs(param - 1.0) < 1.0e-8:
            var_x = 1.0 / fan
        else:
            b = param**2
            a = 3 * (1 - param) ** 2
            var_x = (-b + math.sqrt(b**2 + 4 * a)) / (2 * a * fan)

        return nn.init.normal_(x, 0.0, math.sqrt(var_x), generator=generator)
    else:
        # NOTE: this is the default for kaiming_normal_
        if param is None:
            param = 0.0

        return nn.init.kaiming_normal_(
            x,
            a=param,
            mode=mode,
            nonlinearity=nonlinearity,
            generator=generator,
        )


# }}}


# {{{ available_devices


def available_devices() -> frozenset[torch.device]:
    """Get a list of all supported devices."""

    devices = [torch.device("cpu")]
    devices += [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    if hasattr(torch.backends, "mtia") and torch.backends.mtia.is_available():
        devices.append(torch.device("mtia"))

    if hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
        devices.append(torch.device("xpu"))

    return frozenset(devices)


def available_device_names() -> frozenset[str]:
    """Get a list of available device names."""
    return frozenset([
        d.type if d.index is None else f"{d.type}:{d.index}"
        for d in available_devices()
    ])


@contextmanager
def torch_default_device(
    device: str | torch.device | None = None,
) -> Generator[torch.device]:
    """Context manager to set the default device.

    Newer versions of ``pytorch`` can use ``torch.device`` for this.
    """
    prev_device = torch.get_default_device()

    torch.set_default_device(device)
    new_device = torch.get_default_device()
    assert new_device == get_default_device()

    try:
        yield new_device
    finally:
        torch.set_default_device(prev_device)


def get_default_device() -> torch.device:
    """Get the default device used on tensor creation.

    This creates a dummy tensor and returns its device. It should be equivalent
    to `torch.get_default_device()`, but that does not seem to work across versions.
    """
    return torch.zeros(1).device


# }}}


# {{{ get_memory_usage


@dataclass(frozen=True)
class CUDASnapshot(MemorySnapshot):
    cuda_mb: float
    delta_cuda_mb: float
    peak_cuda_mb: float

    def as_row(self) -> tuple[str, ...]:
        return (
            *super().as_row(),
            f"{self.cuda_mb:.2f}",
            f"{self.delta_cuda_mb:+.2f}",
            f"{self.peak_cuda_mb:.2f}",
        )


class CUDAMemoryTracker(MemoryTracker[CUDASnapshot]):
    def __init__(self, device: Any = None) -> None:
        if not getattr(device, "type", "") == "cuda":
            raise ValueError(f"{type(self).__name__} does not support device: {device}")

        super().__init__(device)

    def record(self, tag: str) -> CUDASnapshot:
        mem = super().record(tag)

        cuda_mb = torch.cuda.memory_allocated(self.device) / (1024.0**2)
        peak_cuda_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
        delta_cuda_mb = 0.0
        if self.snapshots:
            delta_cuda_mb = cuda_mb - self.snapshots[-1].cuda_mb

        snapshot = CUDASnapshot(
            lineno=mem.lineno,
            tag=tag,
            rss_mb=mem.rss_mb,
            delta_rss_mb=mem.delta_rss_mb,
            peak_rss_mb=mem.peak_rss_mb,
            cuda_mb=cuda_mb,
            delta_cuda_mb=delta_cuda_mb,
            peak_cuda_mb=peak_cuda_mb,
        )

        return snapshot

    def labels(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        labels = (
            *super().labels(),
            ("CUDA (MiB)", {"justify": "right"}),
            ("Δ CUDA (MiB)", {"justify": "right"}),
            ("Peak CUDA (MiB)", {"justify": "right"}),
        )

        return labels


def make_memory_tracker(device: torch.device | None = None) -> MemoryTracker:
    if getattr(device, "type", "") == "cuda":
        return CUDAMemoryTracker(device)
    else:
        return MemoryTracker(device)


# }}}


# {{{ DeviceCheckMode


R = TypeVar("R")


class DeviceMismatchError(RuntimeError):
    """Error raised when a Tensor's device does not match the expected one."""


class DeviceCheckMode(TorchDispatchMode):
    """A context manager that can be used to check that all operations are taking
    places on a given device.

    .. code:: python

        with DeviceCheckMode("cuda:0"):
            x = torch.linspace(0.0, 1.0, 32, device="cuda:0")
            y = 2 * x
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = torch.get_default_device()

        self.device = torch.device(device)

    def __torch_dispatch__(  # ruff:ignore[bad-dunder-method-name]
        self,
        func: Callable[..., R],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> R:
        for tensor in pytree.tree_leaves((args, kwargs)):
            if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                raise DeviceMismatchError(
                    f"{func}: tensor on {tensor.device}, expected {self.device}"
                )

        return func(*args, **(kwargs or {}))


# }}}


# {{{ gather_model_signal_statistics


class LayerStatistics(NamedTuple):
    """Simple statistics for a layer output."""

    mean: float
    """The mean value of the output over all dimensions."""
    var: float
    """The (unbiased) variance of the output over all dimensions."""
    msq: float
    """The second-moment :math:`E[y^2]` of the output over all dimensions."""


def rayleigh(
    *shape: int,
    sigma: float = 1.0,
    device: str | torch.device | None = None,
    dtype: Any = None,
) -> torch.Tensor:
    U = torch.rand(*shape, device=device, dtype=dtype)
    torch.clamp_min_(U, 1.0e-12)

    return sigma * torch.sqrt(-2.0 * torch.log(U))


def gather_model_signal_statistics(
    model: nn.Module,
    shape: tuple[int, ...],
    *,
    dtype: Any = None,
    device: str | torch.device | None = None,
    batch: int = 4096,
) -> dict[str, LayerStatistics]:
    r"""Gather statistics over the whole *model* starting from a randomly
    distributed input in :math:`\mathcal{N}(0, 1)`.

    :returns: a dictionary with statistics for each module in the *model*. The
        names of the modules are expected to be unique, see
        :meth:`torch.nn.Module.named_modules`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    stats = {}

    def hook(name: str) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def fn(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
            o = out.detach()
            stats[name] = LayerStatistics(
                mean=torch.mean(o).item(),
                var=torch.var(o, correction=1).item(),
                msq=torch.mean(torch.pow(o, 2)).item(),
            )

        return fn

    # register our hook for each layer
    handles = []
    for name, m in model.named_modules():
        if not name or list(m.children()):
            continue

        handles.append(m.register_forward_hook(hook(name)))

    # run model and some nice data

    if dtype.is_floating_point:
        x = torch.randn(batch, *shape, device=device, dtype=dtype)
    else:
        r = rayleigh(batch, *shape, device=device, dtype=dtype.to_real())
        theta = (2.0 * torch.rand_like(r) - 1.0) * math.pi
        x = view_as_real(torch.polar(r, theta))

    model(x)

    # remove remove hooks from the layers
    for h in handles:
        h.remove()

    return stats


# }}}
