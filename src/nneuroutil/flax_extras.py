# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer

from nneuroutil.helpers import module_logger, to_real

log = module_logger(__name__)


# {{{ activation functions


class Quadratic(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return quadratic(x)


class LeakyQuadratic(nnx.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def __call__(self, x: jax.Array) -> jax.Array:
        return leaky_quadratic(x, alpha=self.alpha)


class ComplexQuadratic(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return complex_quadratic(x)


class ComplexLeakyQuadratic(nnx.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def __call__(self, x: jax.Array) -> jax.Array:
        return complex_leaky_quadratic(x, alpha=self.alpha)


class ComplexTanh(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return complex_tanh(x)


def quadratic(x: jax.Array) -> jax.Array:
    return x * x


def leaky_quadratic(x: jax.Array, *, alpha: float = 0.1) -> jax.Array:
    return alpha * x + (1 - alpha) * x * x


def complex_quadratic(x: jax.Array) -> jax.Array:
    x_re, x_im = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x_re * x_re + x_im * x_im, 2.0 * x_re * x_im], axis=-1)


def complex_leaky_quadratic(x: jax.Array, *, alpha: float = 0.1) -> jax.Array:
    return alpha * x + (1 - alpha) * complex_quadratic(x)


def complex_tanh(x: jax.Array) -> jax.Array:
    return jnp.tanh(x.real) + 1j * jnp.tanh(x.imag)


# }}}


# {{{ layers


class Bias(nnx.Module):
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

        self.bias = nnx.Param(init(rngs.params(), (size,), dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.bias  # ty: ignore[unsupported-operator]


class ComplexLinear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = False,
        dtype: Any | None = None,
        kernel_init: Initializer | None = None,
        bias_init: Initializer | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)

        if kernel_init is None:
            kernel_init = nnx.initializers.lecun_normal()

        if bias_init is None:
            bias_init = nnx.initializers.zeros_init()

        ftype = None if dtype is None else to_real(dtype)

        self.in_features = in_features
        self.out_features = out_features
        self.kernel = nnx.Param(
            kernel_init(rngs.params(), (in_features, out_features), ftype)
        )

        if use_bias:
            self.bias_re = nnx.Param(bias_init(rngs.params(), (out_features,), ftype))
            self.bias_im = nnx.Param(bias_init(rngs.params(), (out_features,), ftype))
        else:
            self.bias_re = self.bias_im = None

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [batch, 2 * in_features]
        x_re, x_im = jnp.split(x, 2, axis=-1)

        result_re = x_re @ self.kernel.value
        if self.bias_re is not None:
            result_re = result_re + self.bias_re.value  # noqa: PLR6104

        result_im = x_im @ self.kernel.value
        if self.bias_im is not None:
            result_im = result_im + self.bias_im.value  # noqa: PLR6104

        return jnp.concatenate([result_re, result_im], axis=-1)


# }}}


# {{{ init


def quadratic_uniform(
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    # NOTE: some simple math to construct that scale=0.5 for our quadratic activation.
    # Assuming a zero-mean Gaussian pre-activation, we have that
    #   Var(x^2) = 2 Var(x)^2
    # To maintain unit variance we'd need
    #   2 Var(x)^2 = Var(x)     => Var(x) = 0.5
    # That's it!

    return nnx.initializers.variance_scaling(
        scale=0.5,
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
    return nnx.initializers.variance_scaling(
        scale=0.5,
        mode="fan_in",
        distribution="truncated_normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def leaky_quadratic_uniform(
    alpha: float = 1.0,
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    if abs(alpha - 1.0) < 1.0e-8:
        scale = 1.0
    else:
        a = 2 * (1 - alpha) ** 2
        b = alpha**2
        scale = (-b + (b**2 + 8 * a) ** 0.5) / (2 * a)

    return nnx.initializers.variance_scaling(
        scale=scale,
        mode="fan_in",
        distribution="uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def leaky_quadratic_normal(
    alpha: float = 1.0,
    in_axis: int | tuple[int, ...] = -2,
    out_axis: int | tuple[int, ...] = -1,
    batch_axis: int | tuple[int, ...] = (),
    dtype: Any = None,
) -> Initializer:
    # NOTE: some simple math for this as well.
    # 1. In the alpha = 1 case, we have a simple linear => scale = 1.0
    # 2. For alpha in (0, 1), we have that
    #   Var(x^2) = a^2 Var(x)^2 + 2 (1 - a)^2 Var(x)^4
    # if we want to maintain unit variance again, we get a nice quadratic in scale
    #   2 (1 - a)^2 s^2 + a^2 s - 1 = 0
    # and pick the positive root.

    if abs(alpha - 1.0) < 1.0e-8:
        scale = 1.0
    else:
        a = 2 * (1 - alpha) ** 2
        b = alpha**2
        scale = (-b + (b**2 + 8 * a) ** 0.5) / (2 * a)

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
