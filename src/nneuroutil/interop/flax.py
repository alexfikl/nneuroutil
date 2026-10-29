# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import Initializer

from nneuroutil import activations
from nneuroutil.helpers import MemorySnapshot, MemoryTracker, calculate_gain, to_real

log = logging.getLogger(__name__)


# {{{ activation functions


class Quadratic(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.quadratic`."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.quadratic(x)  # ty: ignore[invalid-return-type]


class BlendedQuadratic(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.blended_quadratic`."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.blended_quadratic(x, alpha=self.alpha)  # ty: ignore[invalid-return-type]


class ComplexQuadratic(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.complex_quadratic`."""

    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, *, interleaved: bool = False) -> None:
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            activations.complex_quadratic, x, interleaved=self.interleaved
        )


class ComplexBlendedQuadratic(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.complex_blended_quadratic`."""

    alpha: float
    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, alpha: float = 0.1, *, interleaved: bool = False) -> None:
        self.alpha = alpha
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            partial(activations.complex_blended_quadratic, alpha=self.alpha),
            x,
            interleaved=self.interleaved,
        )


class ComplexTanh(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.complex_tanh`."""

    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, *, interleaved: bool = False) -> None:
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            activations.complex_tanh, x, interleaved=self.interleaved
        )


class ModReLU(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.modrelu`."""

    bias: float
    """The bias :math:`b`."""
    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, bias: float = 0.0, *, interleaved: bool = False) -> None:
        self.bias = bias
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            partial(activations.modrelu, b=self.bias),
            x,
            interleaved=self.interleaved,
        )


class LeakyModReLU(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.leaky_modrelu`."""

    bias: float
    """The bias :math:`b`."""
    alpha: float
    r"""The leak slope :math:`\alpha`."""
    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(
        self,
        bias: float = 0.0,
        *,
        alpha: float = 0.1,
        interleaved: bool = False,
    ) -> None:
        self.bias = bias
        self.alpha = alpha
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            partial(activations.leaky_modrelu, b=self.bias, alpha=self.alpha),
            x,
            interleaved=self.interleaved,
        )


class ComplexCardioid(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.complex_cardioid`."""

    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, *, interleaved: bool = False) -> None:
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            activations.complex_cardioid, x, interleaved=self.interleaved
        )


class zReLU(nnx.Module):  # ruff:ignore[invalid-class-name]
    """A wrapper around :func:`~nneuroutil.activations.zrelu`."""

    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, *, interleaved: bool = False) -> None:
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            activations.zrelu, x, interleaved=self.interleaved
        )


class ComplexLog(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.complex_log`."""

    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, *, interleaved: bool = False) -> None:
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            activations.complex_log, x, interleaved=self.interleaved
        )


class ComplexExp(nnx.Module):
    """A wrapper around :func:`~nneuroutil.activations.complex_exp`."""

    interleaved: bool
    """If *True*, assume that the tensors have interleaved real and complex parts."""

    def __init__(self, *, interleaved: bool = False) -> None:
        self.interleaved = interleaved

    def __call__(self, x: jax.Array) -> jax.Array:
        return activations.apply_complex(  # ty: ignore[invalid-return-type]
            activations.complex_exp, x, interleaved=self.interleaved
        )


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
        # x: [..., 2 * in_features]
        batch_shape = x.shape[:-1]

        if self.interleaved:
            # (..., d, 2) -> (..., 2, d)
            x = jnp.swapaxes(
                jnp.reshape(x, (*batch_shape, self.in_features, 2)), -1, -2
            )
        else:
            # (..., 2, d) is already the storage order
            x = jnp.reshape(x, (*batch_shape, 2, self.in_features))

        # one (2 * batch, in_features) @ (in_features, out_features) GEMM
        result = jnp.reshape(x, (-1, self.in_features)) @ self.kernel.value
        result = jnp.reshape(result, (*batch_shape, 2, self.out_features))

        if self.bias_re is not None:
            assert self.bias_im is not None
            result = result + jnp.stack(
                [self.bias_re.value, self.bias_im.value], axis=0
            )

        if self.interleaved:
            result = jnp.swapaxes(result, -1, -2)  # (..., out_features, 2)

        return jnp.reshape(result, (*batch_shape, 2 * self.out_features))


# }}}


# {{{ init


def quadratic_uniform(
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    """A uniform init that preserves variance through :func:`quadratic`."""

    return nnx.initializers.variance_scaling(
        scale=calculate_gain("quadratic"),
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
        scale=calculate_gain("quadratic"),
        mode="fan_in",
        distribution="truncated_normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def blended_quadratic_uniform(
    alpha: float = 0.5,
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    """A uniform init that preserves variance through
    :func:`nneuroutil.activations.blended_quadratic`."""

    return nnx.initializers.variance_scaling(
        scale=calculate_gain("blended_quadratic", alpha=alpha),
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
    """A normal init that preserves variance through
    :func:`nneuroutil.activations.blended_quadratic`."""
    return nnx.initializers.variance_scaling(
        scale=calculate_gain("blended_quadratic", alpha=alpha),
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


# {{{ JaxMemoryTracker


class JaxMemoryTracker(MemoryTracker):
    """A :class:`~nneuroutil.helpers.MemoryTracker` that also records the GPU
    memory usage of *device*.
    """

    def __init__(self, device: Any = None) -> None:
        super().__init__(device)

    def make_record(self, tag: str, *, stacklevel: int = 2) -> MemorySnapshot:
        mem = super().make_record(tag, stacklevel=stacklevel + 1)
        if self.device and (stats := self.device.memory_stats()) is not None:
            bytes_in_use = stats["bytes_in_use"]
            prev_bytes_in_use = (
                self.snapshots[-1].memory["GPU"] if self.snapshots else bytes_in_use
            )

            mem.memory.update({
                "GPU": bytes_in_use,
                "Delta GPU": bytes_in_use - prev_bytes_in_use,
                "Peak GPU": stats["peak_bytes_in_use"],
            })

        return mem


# }}}
