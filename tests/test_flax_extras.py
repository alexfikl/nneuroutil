# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import pytest

from nneuroutil.helpers import module_logger

pytest.importorskip("flax")

import jax
import jax.numpy as jnp
from flax import nnx

from nneuroutil.flax_extras import (
    Bias,
    ComplexLinear,
    ComplexTanh,
    LeakyQuadratic,
    Quadratic,
    complex_tanh,
    leaky_quadratic,
    leaky_quadratic_normal,
    leaky_quadratic_uniform,
    quadratic,
    quadratic_normal,
    quadratic_uniform,
)

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_activation_functions


def test_quadratic() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 5))
    mod = Quadratic()
    res_mod = mod(x)
    res_func = quadratic(x)

    assert jnp.allclose(res_mod, x * x)
    assert jnp.allclose(res_func, x * x)


def test_leaky_quadratic() -> None:
    alpha = 0.2
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 5))

    mod = LeakyQuadratic(alpha=alpha)
    res_mod = mod(x)
    res_func = leaky_quadratic(x, alpha=alpha)
    expected = alpha * x + (1.0 - alpha) * x * x

    assert jnp.allclose(res_mod, expected)
    assert jnp.allclose(res_func, expected)


def test_complex_tanh() -> None:
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x_real = jax.random.normal(key1, (5, 5))
    x_imag = jax.random.normal(key2, (5, 5))
    x = x_real + 1j * x_imag

    mod = ComplexTanh()
    res_mod = mod(x)
    res_func = complex_tanh(x)
    expected = jnp.tanh(x_real) + 1j * jnp.tanh(x_imag)

    assert jnp.allclose(res_mod, expected)
    assert jnp.allclose(res_func, expected)


# }}}


# {{{ test_layers


def test_bias() -> None:
    size = 10
    mod = Bias(size)
    assert mod.bias.value.shape == (size,)
    assert jnp.allclose(mod.bias.value, 0.0)


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear(bias: bool) -> None:  # noqa: FBT001
    in_features = 4
    out_features = 6
    batch_size = 3

    # Input: [batch, 2 * in_features]
    key_x, key_rngs = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key_x, (batch_size, 2 * in_features))
    rngs = nnx.Rngs(params=key_rngs)
    mod = ComplexLinear(in_features, out_features, use_bias=bias, rngs=rngs)

    assert mod.kernel.value.shape == (in_features, out_features)
    if bias:
        assert mod.bias_re is not None
        assert mod.bias_im is not None
        assert mod.bias_re.value.shape == (out_features,)
        assert mod.bias_im.value.shape == (out_features,)
    else:
        assert mod.bias_re is None
        assert mod.bias_im is None

    res = mod(x)
    assert res.shape == (batch_size, 2 * out_features)

    mod_custom_dtype = ComplexLinear(
        in_features,
        out_features,
        use_bias=bias,
        dtype=jnp.complex64,
        rngs=nnx.Rngs(0),
    )
    assert mod_custom_dtype.kernel.value.dtype == jnp.float32


# }}}


# {{{ test_init


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_kaiming_init(alpha: float) -> None:
    shape = (10, 10)
    key = jax.random.PRNGKey(0)

    # 1. quadratic_uniform / quadratic_normal
    init_qu = quadratic_uniform()
    res_qu = init_qu(key, shape)
    assert res_qu.shape == shape
    assert not jnp.allclose(res_qu, 0.0)

    init_qn = quadratic_normal()
    res_qn = init_qn(key, shape)
    assert res_qn.shape == shape
    assert not jnp.allclose(res_qn, 0.0)

    # 2. leaky_quadratic_uniform / leaky_quadratic_normal
    init_lqu = leaky_quadratic_uniform(alpha=alpha)
    res_lqu = init_lqu(key, shape)
    assert res_lqu.shape == shape
    assert not jnp.allclose(res_lqu, 0.0)

    init_lqn = leaky_quadratic_normal(alpha=alpha)
    res_lqn = init_lqn(key, shape)
    assert res_lqn.shape == shape
    assert not jnp.allclose(res_lqn, 0.0)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
