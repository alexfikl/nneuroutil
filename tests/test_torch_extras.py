# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Literal

import pytest

torch = pytest.importorskip("torch")

from nneuroutil.helpers import module_logger

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_activation_functions


def test_quadratic() -> None:
    from nneuroutil.torch_extras import ComplexQuadratic, Quadratic

    # test real only
    x = torch.randn(5, 5)

    mod = Quadratic()
    res = mod(x)
    assert torch.allclose(res, x * x)

    # test complex
    x = torch.randn(5, 10)
    x[:, 5:] = 0.0

    mod = ComplexQuadratic()
    res = mod(x)
    assert torch.allclose(res, x * x)


def test_complex_quadratic_interleave() -> None:
    from nneuroutil.torch_extras import ComplexQuadratic, complex_quadratic

    n = 5
    z = torch.complex(torch.randn(3, n), torch.randn(3, n))
    z2 = z * z

    # test interleaved storage: [re[0], im[0], ..., re[n], im[n]]
    x = torch.view_as_real(z).reshape(3, 2 * n)
    res = complex_quadratic(x, interleaved=True)
    expected = torch.view_as_real(z2).reshape(3, 2 * n)
    assert res.shape == x.shape
    assert torch.allclose(res, expected)

    # test stacked storage: [re[0], ..., re[n], im[0], ..., im[n]]
    x = torch.cat([z.real, z.imag], dim=-1)
    res = complex_quadratic(x, interleaved=False)
    expected = torch.cat([z2.real, z2.imag], dim=-1)
    assert torch.allclose(res, expected)

    # test module
    x = torch.view_as_real(z).reshape(3, 2 * n)
    mod = ComplexQuadratic(interleaved=True)
    res = mod(x)
    expected = torch.view_as_real(z2).reshape(3, 2 * n)
    assert torch.allclose(res, expected)

    # test odd dimension raises
    with pytest.raises(ValueError, match="must be even"):
        complex_quadratic(torch.randn(3, 2 * n + 1), interleaved=True)


def test_linear_quadratic() -> None:
    from nneuroutil.torch_extras import BlendedQuadratic, ComplexBlendedQuadratic

    n = 5
    alpha = 0.2
    x = torch.randn(n, n)

    mod = BlendedQuadratic(alpha=alpha)
    res = mod(x)
    expected = alpha * x + (1.0 - alpha) * x * x

    assert torch.allclose(res, expected)

    # test complex
    x = torch.randn(n, 2 * n)
    x[:, n:] = 0.0

    mod = ComplexBlendedQuadratic(alpha=alpha)
    res = mod(x)
    assert torch.allclose(res, alpha * x + (1.0 - alpha) * x * x)


def test_complex_tanh() -> None:
    from nneuroutil.torch_extras import ComplexTanh

    x_real = torch.randn(5, 5)
    x_imag = torch.randn(5, 5)
    x = torch.complex(x_real, x_imag)

    mod = ComplexTanh()
    res = mod(x)
    expected = torch.complex(torch.tanh(x_real), torch.tanh(x_imag))

    assert torch.allclose(res, expected)


@pytest.mark.parametrize("bias", [0.0, 0.5, 2.0])
def test_modrelu(bias: float) -> None:
    from nneuroutil.torch_extras import ModReLU

    n = 5
    z = torch.complex(torch.randn(3, n), torch.randn(3, n))

    expected = torch.relu(torch.abs(z) + bias) * torch.sgn(z)

    # interleaved storage: [re[0], im[0], ..., re[n], im[n]]
    x = torch.view_as_real(z).reshape(3, 2 * n)
    res = ModReLU(bias=bias, interleaved=True)(x)
    expected_inter = torch.view_as_real(expected).reshape(3, 2 * n)
    assert res.shape == x.shape
    assert torch.allclose(res, expected_inter)

    # stacked storage: [re[0], ..., re[n], im[0], ..., im[n]]
    x = torch.cat([z.real, z.imag], dim=-1)
    res = ModReLU(bias=bias, interleaved=False)(x)
    expected_stack = torch.cat([expected.real, expected.imag], dim=-1)
    assert torch.allclose(res, expected_stack)


def test_modrelu_known_values() -> None:
    from nneuroutil.torch_extras import ModReLU

    # z = 1 + 1j has |z| = sqrt(2); with bias = -sqrt(2) the magnitude vanishes
    z = torch.tensor([1.0 + 1.0j])
    xi = torch.view_as_real(z).reshape(2)
    res = ModReLU(bias=-(2.0**0.5), interleaved=True)(xi)
    assert torch.allclose(res, torch.zeros(2))

    # with bias = 0, ModReLU(z) = |z| * z / |z| = z (for |z| > 0)
    res = ModReLU(interleaved=True)(xi)
    assert torch.allclose(res, xi)

    # purely imaginary input is preserved (phase preserved) for bias = 0
    z = torch.tensor([0.0 + 3.0j])
    xi = torch.view_as_real(z).reshape(2)
    res = ModReLU(interleaved=True)(xi)
    assert torch.allclose(res, torch.tensor([0.0, 3.0]))


@pytest.mark.parametrize("bias", [-2.0, 0.0, 1.0])
@pytest.mark.parametrize("alpha", [0.1, 0.5])
def test_leaky_modrelu(bias: float, alpha: float) -> None:
    from nneuroutil.torch_extras import LeakyModReLU

    n = 5
    z = torch.complex(torch.randn(3, n), torch.randn(3, n))

    r = torch.abs(z)
    expected = torch.where(r + bias >= 0.0, r + bias, alpha * r) * torch.sgn(z)

    # interleaved storage: [re[0], im[0], ..., re[n], im[n]]
    x = torch.view_as_real(z).reshape(3, 2 * n)
    res = LeakyModReLU(bias=bias, alpha=alpha, interleaved=True)(x)
    expected_inter = torch.view_as_real(expected).reshape(3, 2 * n)
    assert res.shape == x.shape
    assert torch.allclose(res, expected_inter)

    # stacked storage: [re[0], ..., re[n], im[0], ..., im[n]]
    x = torch.cat([z.real, z.imag], dim=-1)
    res = LeakyModReLU(bias=bias, alpha=alpha, interleaved=False)(x)
    expected_stack = torch.cat([expected.real, expected.imag], dim=-1)
    assert torch.allclose(res, expected_stack)


def test_leaky_modrelu_known_values() -> None:
    from nneuroutil.torch_extras import LeakyModReLU

    # with bias = 0: |z| >= 0 always, so we are always in the positive branch and
    # LeakyModReLU(z) = |z| * sgn(z) = z, independent of alpha
    z = torch.tensor([1.0 + 1.0j, -2.0 + 0.5j, 0.0 + 3.0j])
    xi = torch.view_as_real(z).reshape(6)
    res = LeakyModReLU(bias=0.0, alpha=0.1, interleaved=True)(xi)
    assert torch.allclose(res, xi)

    # negative branch: with |z| = 0.5 and bias = -1.0 we get |z| + b < 0, so the
    # output is alpha * |z| * sgn(z) = alpha * z (phase preserved)
    z = torch.tensor([0.3 + 0.4j])
    xi = torch.view_as_real(z).reshape(2)
    res = LeakyModReLU(bias=-1.0, alpha=0.5, interleaved=True)(xi)
    assert torch.allclose(res, torch.tensor([0.15, 0.2]))

    # boundary: |z| + b == 0 falls in the positive (>=) branch, giving 0
    z = torch.tensor([1.0 + 1.0j])  # |z| = sqrt(2)
    xi = torch.view_as_real(z).reshape(2)
    res = LeakyModReLU(bias=-(2.0**0.5), alpha=0.5, interleaved=True)(xi)
    assert torch.allclose(res, torch.zeros(2))


def test_leaky_modrelu_matches_modrelu() -> None:
    # with alpha = 0, LeakyModReLU coincides with ModReLU
    from nneuroutil.torch_extras import LeakyModReLU, ModReLU

    n = 5
    z = torch.complex(torch.randn(3, n), torch.randn(3, n))
    x = torch.cat([z.real, z.imag], dim=-1)

    bias = -1.5
    res_leaky = LeakyModReLU(bias=bias, alpha=0.0, interleaved=False)(x)
    res_mod = ModReLU(bias=bias, interleaved=False)(x)
    assert torch.allclose(res_leaky, res_mod)


def test_complex_cardioid() -> None:
    from nneuroutil.torch_extras import ComplexCardioid

    n = 5
    z = torch.complex(torch.randn(3, n), torch.randn(3, n))

    expected = 0.5 * (1 + torch.cos(torch.angle(z))) * z

    # interleaved storage
    x = torch.view_as_real(z).reshape(3, 2 * n)
    res = ComplexCardioid(interleaved=True)(x)
    expected_inter = torch.view_as_real(expected).reshape(3, 2 * n)
    assert res.shape == x.shape
    assert torch.allclose(res, expected_inter)

    # stacked storage
    x = torch.cat([z.real, z.imag], dim=-1)
    res = ComplexCardioid(interleaved=False)(x)
    expected_stack = torch.cat([expected.real, expected.imag], dim=-1)
    assert torch.allclose(res, expected_stack)


def test_complex_cardioid_known_values() -> None:
    from nneuroutil.torch_extras import ComplexCardioid

    # for real positive z (angle 0): factor = 0.5 * (1 + 1) = 1, so f(z) = z
    z = torch.tensor([2.0 + 0.0j])
    xi = torch.view_as_real(z).reshape(2)
    res = ComplexCardioid(interleaved=True)(xi)
    assert torch.allclose(res, torch.tensor([2.0, 0.0]))

    # for purely imaginary z (angle pi/2): factor = 0.5 * (1 + 0) = 0.5
    z = torch.tensor([0.0 + 2.0j])
    xi = torch.view_as_real(z).reshape(2)
    res = ComplexCardioid(interleaved=True)(xi)
    assert torch.allclose(res, torch.tensor([0.0, 1.0]))


def test_zrelu() -> None:
    from nneuroutil.torch_extras import zReLU

    n = 5
    z = torch.complex(torch.randn(3, n), torch.randn(3, n))

    mask = ((z.real >= 0.0) & (z.imag >= 0.0)).to(z.real.dtype)
    expected = z * mask

    # interleaved storage
    x = torch.view_as_real(z).reshape(3, 2 * n)
    res = zReLU(interleaved=True)(x)
    expected_inter = torch.view_as_real(expected).reshape(3, 2 * n)
    assert res.shape == x.shape
    assert torch.allclose(res, expected_inter)

    # stacked storage
    x = torch.cat([z.real, z.imag], dim=-1)
    res = zReLU(interleaved=False)(x)
    expected_stack = torch.cat([expected.real, expected.imag], dim=-1)
    assert torch.allclose(res, expected_stack)


def test_zrelu_known_values() -> None:
    from nneuroutil.torch_extras import zReLU

    # interleaved input z = [1+1j, -1+1j, 1-1j, -1-1j]; only the first quadrant
    # entry (1 + 1j) passes through, the rest are zeroed.
    x = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    res = zReLU(interleaved=True)(x)
    expected = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(res, expected)

    # boundary: values on the axes (Re = 0 or Im = 0) are kept by the >= test
    x = torch.tensor([0.0, 5.0, 5.0, 0.0])
    res = zReLU(interleaved=True)(x)
    assert torch.allclose(res, x)


# }}}


# {{{ test_layers


def test_bias() -> None:
    from nneuroutil.torch_extras import Bias

    size = 10
    mod = Bias(size)

    assert mod.bias.shape == (size,)
    assert torch.allclose(mod.bias, torch.zeros_like(mod.bias))


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear(bias: bool) -> None:  # ruff:ignore[boolean-type-hint-positional-argument]
    from nneuroutil.torch_extras import ComplexLinear

    in_features = 4
    out_features = 6
    batch_size = 3

    # Input: [batch, 2 * in_features]
    x = torch.randn(batch_size, 2 * in_features)
    mod = ComplexLinear(in_features, out_features, bias=bias)

    assert mod.weight.shape == (out_features, in_features)
    if bias:
        assert mod.bias_re is not None
        assert mod.bias_im is not None
        assert mod.bias_re.shape == (out_features,)
        assert mod.bias_im.shape == (out_features,)
    else:
        assert mod.bias_re is None
        assert mod.bias_im is None

    res = mod(x)
    assert res.shape == (batch_size, 2 * out_features)

    # Let's also check with a custom dtype object that supports to_real()
    class MockDtype:
        def to_real(self) -> torch.dtype:  # ruff:ignore[no-self-use]
            return torch.float32

    mod_custom_dtype = ComplexLinear(
        in_features, out_features, bias=bias, dtype=MockDtype()
    )
    assert mod_custom_dtype.weight.dtype == torch.float32


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear_interleave(bias: bool) -> None:  # ruff:ignore[boolean-type-hint-positional-argument]
    from nneuroutil.torch_extras import ComplexLinear

    in_features = 4
    out_features = 6
    batch_size = 3

    z = torch.complex(
        torch.randn(batch_size, in_features), torch.randn(batch_size, in_features)
    )

    # input: [batch, 2 * in_features] with interleaved storage
    x = torch.view_as_real(z).reshape(batch_size, 2 * in_features)
    mod = ComplexLinear(in_features, out_features, interleaved=True, bias=bias)

    res = mod(x)
    assert res.shape == (batch_size, 2 * out_features)

    expected_re = torch.nn.functional.linear(z.real, mod.weight, mod.bias_re)
    expected_im = torch.nn.functional.linear(z.imag, mod.weight, mod.bias_im)
    expected = torch.stack([expected_re, expected_im], dim=-1).reshape(
        batch_size, 2 * out_features
    )
    assert torch.allclose(res, expected)

    # check consistency with the stacked layout
    mod.interleaved = False
    x = torch.cat([z.real, z.imag], dim=-1)

    res = mod(x)
    expected = torch.cat([expected_re, expected_im], dim=-1)
    assert torch.allclose(res, expected)


# }}}


# {{{ test_init


@pytest.mark.parametrize("nonlinearity", ["quadratic", "blended_quadratic", "relu"])
@pytest.mark.parametrize("param", [None, 0.0, 0.2])
@pytest.mark.parametrize("mode", ["fan_in", "fan_out"])
def test_kaiming_init(
    nonlinearity: str,
    param: float | None,
    mode: Literal["fan_in", "fan_out"],
) -> None:
    from nneuroutil.torch_extras import kaiming_normal_, kaiming_uniform_

    shape = (10, 10)
    x = torch.empty(shape)

    res_uniform = kaiming_uniform_(
        x.clone(),
        nonlinearity=nonlinearity,
        param=param,
        mode=mode,
    )
    assert res_uniform.shape == shape
    assert not torch.allclose(res_uniform, torch.zeros_like(res_uniform))

    res_normal = kaiming_normal_(
        x.clone(),
        nonlinearity=nonlinearity,
        param=param,
        mode=mode,
    )
    assert res_normal.shape == shape
    assert not torch.allclose(res_normal, torch.zeros_like(res_normal))


# }}}


# {{{ test_device_check_mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU backend not available")
def test_device_check_mode() -> None:
    from nneuroutil.torch_extras import DeviceCheckMode, DeviceMismatchError

    torch.set_default_device("cuda")
    device = torch.get_default_device()

    with DeviceCheckMode(device):
        # this should pass
        x = torch.linspace(0.0, 1.0, 32, device=device)
        assert x.device.type.startswith("cuda")

        # this should also pass, since it's all on the default device
        y = 2 * x
        assert y is not None

    with pytest.raises(DeviceMismatchError):  # ruff:ignore[multiple-with-statements, pytest-raises-with-multiple-statements]
        with DeviceCheckMode():
            # this should pass
            x = torch.linspace(0.0, 1.0, 32, device="cpu")
            assert x.device.type == "cpu"

            # this goes through the backend and will get caught by DeviceCheckMode
            y = 2 * x
            assert y is not None


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
