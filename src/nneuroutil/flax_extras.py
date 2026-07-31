# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from nneuroutil.array_api_extras import deinterleave, interleave
from nneuroutil.helpers import module_logger, to_real

log = module_logger(__name__)


# {{{ activation functions


class Quadratic(nnx.Module):
    """A quadratic :math:`x^2` activation function."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return quadratic(x)


class BlendedQuadratic(nnx.Module):
    r"""A convex combination of a linear and a quadratic activation.

    .. math::

        f(x; \alpha) = \alpha x + (1 - \alpha) x^2.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def __call__(self, x: jax.Array) -> jax.Array:
        return blended_quadratic(x, alpha=self.alpha)


class ComplexQuadratic(nnx.Module):
    """An activation function that uses :func:`complex_quadratic`."""

    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, *, interleaved: bool = False) -> None:
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return complex_quadratic(x, interleaved=self.interleaved)


class ComplexBlendedQuadratic(nnx.Module):
    r"""A convex combination of a linear and a quadratic activation, where the
    quadratic term uses :func:`complex_quadratic`.

    .. math::

        f(x; \alpha) = \alpha x + (1 - \alpha) x^2.
    """

    alpha: float
    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, alpha: float = 0.1, *, interleaved: bool = False) -> None:
        self.alpha = alpha
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return complex_blended_quadratic(
            x, alpha=self.alpha, interleaved=self.interleaved
        )


class ComplexTanh(nnx.Module):
    r"""A hyperbolic tangent for a complex vector of shape ``(d,)``.

    This applies the hyperbolic tangent to the real and imaginary components
    separately. Note that this is different than :math:`\tanh(z)` for a complex
    :math:`z`.
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        return complex_tanh(x)


def quadratic(x: jax.Array) -> jax.Array:
    """The quadratic activation :math:`x^2`."""
    return x * x


def blended_quadratic(x: jax.Array, *, alpha: float = 0.1) -> jax.Array:
    r"""The blended quadratic activation :math:`\alpha x + (1 - \alpha) x^2`."""
    return alpha * x + (1 - alpha) * x * x


def complex_quadratic(x: jax.Array, *, interleaved: bool = False) -> jax.Array:
    """Treat *x* as a ``(2 d,)`` shaped complex array and compute its square.

    The storage of the array depends on *interleaved*. If *True*, we assume that
    the array is stored as ``[x[0].real, x[0].imag, ..., x[n].real,
    x[n].imag]``. Otherwise, we assume that it is stacked as ``[x[0].real, ...,
    x[n].real, x[0].imag, ..., x[n].imag]``.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"dimension 'd' of 'x[..., d]' must be even: {x.shape}")

    if interleaved:
        x_re, x_im = deinterleave(x)
        result_re = x_re * x_re - x_im * x_im
        result_im = 2.0 * x_re * x_im
        return interleave(result_re, result_im)  # ty: ignore[invalid-return-type]
    else:
        x_re, x_im = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x_re * x_re - x_im * x_im, 2.0 * x_re * x_im], axis=-1)


def complex_blended_quadratic(
    x: jax.Array, *, alpha: float = 0.1, interleaved: bool = False
) -> jax.Array:
    r"""Blended quadratic for complex-valued inputs."""
    return alpha * x + (1 - alpha) * complex_quadratic(x, interleaved=interleaved)


def complex_tanh(x: jax.Array) -> jax.Array:
    """Hyperbolic tangent applied separately to real and imaginary parts."""
    return jnp.tanh(x.real) + 1j * jnp.tanh(x.imag)


# }}}


# {{{ layers


class Bias(nnx.Module):
    """A bias-only layer: :math:`y = x + b`."""

    bias: nnx.Param
    """The learnable bias of shape ``(size,)``."""

    def __init__(
        self,
        size: int,
        *,
        dtype: Any = None,
        init: Initializer | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if init is None:
            init = nnx.initializers.zeros_init()

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.size = size
        self.bias = nnx.Param(init(rngs.params(), (size,), dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies a translation to the inputs along the last dimension."""
        return x + self.bias


class Residual(nnx.Module):
    """A basic residual layer around a given module."""

    model: nnx.Module
    """The wrapped model."""

    def __init__(self, m: nnx.Module) -> None:
        super().__init__()
        self.module = m

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies a residual to the :attr:`model`."""
        return self.module(x) + x


class ComplexLinear(nnx.Module):
    """A linear layer applied to stacked complex variables.

    We assume that the input vector is of shape ``(2 d,)`` representing a
    stacked complex vector of shape ``(d,)``.
    """

    in_features: int
    """Size of each input sample."""
    out_features: int
    """Size of each output sample."""

    kernel: nnx.Param
    """The learnable weights of the module of shape ``(out_features, in_features)``."""
    bias_re: nnx.Param | None
    """The learnable bias of shape ``(out_features,)`` for the real part of the
    input vector.
    """
    bias_im: nnx.Param | None
    """The learnable bias of shape ``(out_features,)`` for the imaginary part of the
    input vector.
    """
    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = False,
        interleaved: bool = False,
        dtype: Any | None = None,
        kernel_init: Initializer | None = None,
        bias_init: Initializer | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        super().__init__()

        if rngs is None:
            rngs = nnx.Rngs(0)

        if kernel_init is None:
            kernel_init = nnx.initializers.lecun_normal()

        if bias_init is None:
            bias_init = nnx.initializers.zeros_init()

        ftype = None if dtype is None else to_real(dtype)

        self.in_features = in_features
        self.out_features = out_features
        self.interleaved = interleaved
        self.kernel = nnx.Param(
            kernel_init(rngs.params(), (in_features, out_features), ftype)
        )

        if use_bias:
            self.bias_re = nnx.Param(bias_init(rngs.params(), (out_features,), ftype))
            self.bias_im = nnx.Param(bias_init(rngs.params(), (out_features,), ftype))
        else:
            self.bias_re = self.bias_im = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies a complex linear transformation to the inputs along the last
        dimension.
        """
        # x: [batch, 2 * in_features]
        if self.interleaved:
            x_re, x_im = deinterleave(x)
        else:
            x_re, x_im = jnp.split(x, 2, axis=-1)

        result_re = x_re @ self.kernel.value
        if self.bias_re is not None:
            result_re = result_re + self.bias_re.value  # ruff:ignore[non-augmented-assignment]

        result_im = x_im @ self.kernel.value
        if self.bias_im is not None:
            result_im = result_im + self.bias_im.value  # ruff:ignore[non-augmented-assignment]

        if self.interleaved:
            return interleave(result_re, result_im)  # ty: ignore[invalid-return-type]
        else:
            return jnp.concatenate([result_re, result_im], axis=-1)


# }}}


# {{{ init


def quadratic_uniform(
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    """A uniform init that preserves variance through :func:`quadratic`."""

    # NOTE just a small math derivation for that scale=1/sqrt(3).
    # 1. We assume that we have a zero-mean Gaussian pre-activation.
    # 2. Then, we have that
    #   E[y^2] = E[x^4] = 3 Var[x]^2
    # 3. We want to impose a unit second moment, so
    #   E[y^2] = 3 Var[x]^2 = 1      =>  scale = Var[x] = 1 / sqrt(3)
    # That's it! This matches what the Kaiming inits do for ReLU.

    return nnx.initializers.variance_scaling(
        scale=1 / math.sqrt(3),
        mode="fan_in",
        distribution="uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def quadratic_normal(
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    """A normal init that preserves variance through :func:`quadratic`."""
    return nnx.initializers.variance_scaling(
        scale=1 / math.sqrt(3),
        mode="fan_in",
        distribution="truncated_normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def blended_quadratic_uniform(
    alpha: float = 1.0,
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    """A uniform init that preserves variance through :func:`blended_quadratic`."""

    # NOTE: blended_quadratic has a nonzero mean for alpha != 1, so we use the
    # second-moment convention (like He/Kaiming for ReLU) to keep the next
    # layer's pre-activation at unit variance.
    # 1. In the alpha = 1 case, we have a simple linear => scale = 1.0
    # 2. Otherwise, with scale = Var(z) and y = alpha z + (1 - alpha) z^2,
    #   E[y^2] = alpha^2 v + 3 (1 - alpha)^2 v^2
    # 3. E[y^2] = 1 gives a quadratic in s = scale:
    #   3 (1 - alpha)^2 s^2 + alpha^2 s - 1 = 0
    # of which we take the positive root.

    if abs(alpha - 1.0) < 1.0e-8:
        scale = 1.0
    else:
        a = 3 * (1 - alpha) ** 2
        b = alpha**2
        scale = (-b + math.sqrt(b**2 + 4 * a)) / (2 * a)

    return nnx.initializers.variance_scaling(
        scale=scale,
        mode="fan_in",
        distribution="uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def blended_quadratic_normal(
    alpha: float = 1.0,
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    """A normal init that preserves variance through :func:`blended_quadratic`."""
    if abs(alpha - 1.0) < 1.0e-8:
        scale = 1.0
    else:
        a = 3 * (1 - alpha) ** 2
        b = alpha**2
        scale = (-b + math.sqrt(b**2 + 4 * a)) / (2 * a)

    return nnx.initializers.variance_scaling(
        scale=scale,
        mode="fan_in",
        distribution="truncated_normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


# }}}


# {{{ available_device_names


class UnknownDeviceError(RuntimeError):
    """Error raised when a device is not known."""


def available_device_names() -> frozenset[str]:
    """
    :returns: a set of ``platform:id`` device names supported by ``jax``. If
        a device object list is needed, just use :func:`jax.devices`.
    """
    return frozenset([f"{d.platform}:{d.id}" for d in jax.devices()])


@contextmanager
def jax_default_device(device: str) -> Generator[None]:
    """A context manager similar to ``jax.default_device``.

    :arg device: a device name in the form ``platform[:id]``.
    """
    if ":" in device:
        platform, ids = device.split(":", maxsplit=1)
        if ids:
            try:
                dev_id = int(ids)
            except ValueError as exc:
                raise UnknownDeviceError(f"could not parse device {device:r}") from exc
        else:
            dev_id = 0
    else:
        platform = device
        dev_id = 0

    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        raise UnknownDeviceError(
            f"cannot set device {device!r}: unknown platform"
        ) from exc

    if not 0 <= dev_id < len(devices):
        raise UnknownDeviceError(
            f"cannot set device {device!r}: id out of bounds for {len(devices)} devices"
        )

    prev_device = jax.config.jax_default_device
    jax.config.update("jax_default_device", devices[dev_id])
    try:
        yield
    finally:
        jax.config.update("jax_default_device", prev_device)


def set_jax_config(
    *,
    platform: str | None = None,
) -> None:
    """Set up any ``jax`` related functionality.

    This should be called after JAX is imported. It will mainly enable ``float64``
    mode, register any required PyTrees, etc.
    """
    if platform is not None:
        jax.config.update("jax_platform_name", platform)

    from nneuroutil.helpers import _PENDING_JAX_REGISTER_DATACLASS

    jax.config.update("jax_enable_x64", val=True)
    while _PENDING_JAX_REGISTER_DATACLASS:
        jax.tree_util.register_dataclass(_PENDING_JAX_REGISTER_DATACLASS.pop())


# }}}
