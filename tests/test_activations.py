# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import array_api_compat
import numpy as np
import pytest

from nneuroutil import activations

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = logging.getLogger(__name__)


# {{{ helpers


def z_complex(xp: Any, rng: np.random.Generator) -> Any:
    z = rng.standard_normal((3, 5)) + 1j * rng.standard_normal((3, 5))

    # the origin, to test sgn(0) = 0
    z[0, 0] = 0.0
    # purely real
    z[1, 2] = 0.5 + 0.0j
    # purely imaginary
    z[2, 4] = 0.0 - 0.25j

    return xp.asarray(z)


# }}}


# {{{ test_real_activations


def test_quadratic(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x_np = rng.standard_normal((4, 6))
    x = xp.asarray(x_np)

    res = activations.quadratic(x)
    assert np.allclose(res, x_np * x_np)


def test_blended_quadratic(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    x_np = rng.standard_normal((4, 6))
    x = xp.asarray(x_np)

    # scalar alpha
    res = activations.blended_quadratic(x, alpha=0.3)
    expected = 0.3 * x_np + 0.7 * x_np * x_np
    assert np.allclose(res, expected)

    # the default is alpha = 0.5
    res = activations.blended_quadratic(x)
    expected = 0.5 * x_np + 0.5 * x_np * x_np
    assert np.allclose(res, expected)

    # per-feature alpha
    alpha_np = rng.uniform(0.0, 1.0, size=(6,))
    res = activations.blended_quadratic(x, alpha=xp.asarray(alpha_np))
    expected = alpha_np * x_np + (1.0 - alpha_np) * x_np * x_np
    assert np.allclose(res, expected)


# }}}


# {{{ test_complex_activations


def test_complex_quadratic(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.complex_quadratic(z)

    assert res.shape == z.shape
    assert np.allclose(res, z_np * z_np)


def test_complex_blended_quadratic(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)
    d = z.shape[-1]

    # scalar alpha
    res = activations.complex_blended_quadratic(z, alpha=0.3)
    expected = 0.3 * z_np + 0.7 * z_np * z_np
    assert np.allclose(res, expected)

    # per-complex-channel alpha
    alpha_np = np.linspace(0.1, 0.9, d)
    res = activations.complex_blended_quadratic(z, alpha=xp.asarray(alpha_np))
    expected = alpha_np * z_np + (1.0 - alpha_np) * z_np * z_np
    assert np.allclose(res, expected)

    # invalid alpha shape
    with pytest.raises(ValueError, match="does not match"):
        activations.complex_blended_quadratic(z, alpha=xp.zeros((d + 1,)))


def test_complex_tanh(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.complex_tanh(z)

    assert res.shape == z.shape
    expected = np.tanh(z_np.real) + 1j * np.tanh(z_np.imag)
    assert np.allclose(res, expected)


@pytest.mark.parametrize("bias", [0.0, 0.5, -0.5])
def test_modrelu(xp: Any, bias: float) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.modrelu(z, bias)

    assert res.shape == z.shape
    expected = np.maximum(np.abs(z_np) + bias, 0.0) * np.sign(z_np)
    assert np.allclose(res, expected)


@pytest.mark.parametrize("bias", [-2.0, 0.0, 1.0])
@pytest.mark.parametrize("alpha", [0.1, 0.5])
def test_leaky_modrelu(xp: Any, bias: float, alpha: float) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.leaky_modrelu(z, bias, alpha=alpha)

    assert res.shape == z.shape
    r = np.abs(z_np)
    rb = r + bias
    expected = np.where(rb >= 0.0, rb, alpha * r) * np.sign(z_np)
    assert np.allclose(res, expected)


def test_complex_cardioid(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.complex_cardioid(z)

    assert res.shape == z.shape
    expected = 0.5 * (1.0 + np.cos(np.angle(z_np))) * z_np
    assert np.allclose(res, expected)


def test_zrelu(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.zrelu(z)

    assert res.shape == z.shape
    expected = z_np * ((z_np.real >= 0.0) & (z_np.imag >= 0.0))
    assert np.allclose(res, expected)


def test_complex_log(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.complex_log(z)

    assert res.shape == z.shape
    expected = np.log1p(np.abs(z_np)) * np.sign(z_np)
    assert np.allclose(res, expected)


def test_complex_exp(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.complex_exp(z)

    assert res.shape == z.shape
    expected = np.expm1(np.abs(z_np)) * np.sign(z_np)
    assert np.allclose(res, expected)


def test_complex_exp_log_are_inverses(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.complex_exp(activations.complex_log(z))
    assert np.allclose(res, z_np)

    res = activations.complex_log(activations.complex_exp(z))
    assert np.allclose(res, z_np)


# }}}


# {{{ test_validation


@pytest.mark.parametrize(
    "fn",
    [
        activations.complex_quadratic,
        activations.complex_blended_quadratic,
        activations.complex_tanh,
        activations.modrelu,
        activations.leaky_modrelu,
        activations.complex_cardioid,
        activations.zrelu,
        activations.complex_log,
        activations.complex_exp,
    ],
)
def test_complex_dtype_required(xp: Any, fn: Any) -> None:
    x = xp.asarray(np.ones((3, 5)))

    with pytest.raises(TypeError, match="complex"):
        fn(x)


def test_explicit_namespace(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    z = z_complex(xp, rng)
    z_np = np.asarray(z)

    res = activations.complex_quadratic(z, xp=xp)
    assert np.allclose(res, z_np * z_np)


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (activations.complex_log, 1.0),
        (activations.complex_exp, 1.0),
        (activations.complex_cardioid, 0.5),
        (lambda x: activations.modrelu(x, 0.5), 0.0),
    ],
    ids=["log", "exp", "cardioid", "modrelu"],
)
def test_origin_gradient(xp: Any, fn: Any, expected: float) -> None:
    x = xp.asarray([0.0 + 0.0j], dtype=xp.complex64)

    if array_api_compat.is_torch_array(x):
        x.requires_grad_(requires_grad=True)
        fn(x).real.sum().backward()
        grad = x.grad
    elif array_api_compat.is_jax_array(x):
        import jax

        grad = jax.grad(lambda z: xp.real(xp.sum(fn(z))))(x)
    elif array_api_compat.is_numpy_array(x):
        pytest.skip("numpy does not provide autodiff")

    assert np.allclose(grad, expected)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
