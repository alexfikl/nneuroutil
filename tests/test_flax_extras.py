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

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_activation_functions


def test_quadratic() -> None:
    from nneuroutil.flax_extras import Quadratic, quadratic

    x = jax.random.normal(jax.random.PRNGKey(0), (5, 5))
    mod = Quadratic()
    res_mod = mod(x)
    res_func = quadratic(x)

    assert jnp.allclose(res_mod, x * x)
    assert jnp.allclose(res_func, x * x)


def test_blended_quadratic() -> None:
    from nneuroutil.flax_extras import BlendedQuadratic, blended_quadratic

    alpha = 0.2
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 5))

    mod = BlendedQuadratic(alpha=alpha)
    res_mod = mod(x)
    res_func = blended_quadratic(x, alpha=alpha)
    expected = alpha * x + (1.0 - alpha) * x * x

    assert jnp.allclose(res_mod, expected)
    assert jnp.allclose(res_func, expected)


def test_complex_tanh() -> None:
    from nneuroutil.flax_extras import ComplexTanh, complex_tanh

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


def test_complex_quadratic_interleave() -> None:
    from nneuroutil.flax_extras import ComplexQuadratic, complex_quadratic

    n = 5
    key_re, key_im = jax.random.split(jax.random.PRNGKey(0))
    z_re = jax.random.normal(key_re, (3, n))
    z_im = jax.random.normal(key_im, (3, n))
    z2_re = z_re * z_re - z_im * z_im
    z2_im = 2.0 * z_re * z_im

    # interleaved storage: [re[0], im[0], ..., re[n], im[n]]
    x = jnp.empty((3, 2 * n))
    x = x.at[:, ::2].set(z_re).at[:, 1::2].set(z_im)
    res = complex_quadratic(x, interleaved=True)
    expected = jnp.empty((3, 2 * n))
    expected = expected.at[:, ::2].set(z2_re).at[:, 1::2].set(z2_im)
    assert res.shape == x.shape
    assert jnp.allclose(res, expected)

    # stacked storage: [re[0], ..., re[n], im[0], ..., im[n]]
    x = jnp.concatenate([z_re, z_im], axis=-1)
    res = complex_quadratic(x, interleaved=False)
    expected = jnp.concatenate([z2_re, z2_im], axis=-1)
    assert jnp.allclose(res, expected)

    # test module
    x = jnp.empty((3, 2 * n))
    x = x.at[:, ::2].set(z_re).at[:, 1::2].set(z_im)
    mod = ComplexQuadratic(interleaved=True)
    res = mod(x)
    expected = jnp.empty((3, 2 * n))
    expected = expected.at[:, ::2].set(z2_re).at[:, 1::2].set(z2_im)
    assert jnp.allclose(res, expected)

    # test odd dimension raises
    with pytest.raises(ValueError, match="must be even"):
        complex_quadratic(jnp.empty((3, 2 * n + 1)), interleaved=True)


# }}}


# {{{ test_layers


def test_bias() -> None:
    from nneuroutil.flax_extras import Bias

    size = 10
    mod = Bias(size)
    assert mod.bias.value.shape == (size,)
    assert jnp.allclose(mod.bias.value, 0.0)


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear(bias: bool) -> None:  # ruff:ignore[boolean-type-hint-positional-argument]
    from nneuroutil.flax_extras import ComplexLinear

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


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear_interleave(bias: bool) -> None:  # ruff:ignore[boolean-type-hint-positional-argument]
    from nneuroutil.flax_extras import ComplexLinear

    in_features = 4
    out_features = 6
    batch_size = 3

    key_re, key_im, key_rngs = jax.random.split(jax.random.PRNGKey(0), 3)
    z_re = jax.random.normal(key_re, (batch_size, in_features))
    z_im = jax.random.normal(key_im, (batch_size, in_features))

    rngs = nnx.Rngs(params=key_rngs)

    # interleaved input: [batch, 2 * in_features]
    x = jnp.empty((batch_size, 2 * in_features))
    x = x.at[:, ::2].set(z_re).at[:, 1::2].set(z_im)
    mod = ComplexLinear(
        in_features, out_features, interleaved=True, use_bias=bias, rngs=rngs
    )

    res = mod(x)
    assert res.shape == (batch_size, 2 * out_features)

    expected_re = z_re @ mod.kernel.value
    expected_im = z_im @ mod.kernel.value
    if bias:
        assert mod.bias_re is not None
        assert mod.bias_im is not None
        expected_re += mod.bias_re.value
        expected_im += mod.bias_im.value

    expected = jnp.empty((batch_size, 2 * out_features))
    expected = expected.at[:, ::2].set(expected_re).at[:, 1::2].set(expected_im)
    assert jnp.allclose(res, expected)

    # check consistency with the stacked layout
    mod.interleaved = False
    x = jnp.concatenate([z_re, z_im], axis=-1)

    res = mod(x)
    expected = jnp.concatenate([expected_re, expected_im], axis=-1)
    assert jnp.allclose(res, expected)


# }}}


# {{{ test_init


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_kaiming_init(alpha: float) -> None:
    from nneuroutil.flax_extras import (
        blended_quadratic_normal,
        blended_quadratic_uniform,
        quadratic_normal,
        quadratic_uniform,
    )

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

    # 2. blended_quadratic_uniform / blended_quadratic_normal
    init_bqu = blended_quadratic_uniform(alpha=alpha)
    res_bqu = init_bqu(key, shape)
    assert res_bqu.shape == shape
    assert not jnp.allclose(res_bqu, 0.0)

    init_bqn = blended_quadratic_normal(alpha=alpha)
    res_bqn = init_bqn(key, shape)
    assert res_bqn.shape == shape
    assert not jnp.allclose(res_bqn, 0.0)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
