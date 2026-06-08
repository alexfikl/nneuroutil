# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Literal

import pytest

from nneuroutil.helpers import module_logger

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_activation_functions


def test_quadratic() -> None:
    torch = pytest.importorskip("torch")

    from nneuroutil.torch_extras import Quadratic

    x = torch.randn(5, 5)

    mod = Quadratic()
    res = mod(x)

    assert torch.allclose(res, x * x)


def test_leaky_quadratic() -> None:
    torch = pytest.importorskip("torch")

    from nneuroutil.torch_extras import LeakyQuadratic

    alpha = 0.2
    x = torch.randn(5, 5)

    mod = LeakyQuadratic(alpha=alpha)
    res = mod(x)
    expected = alpha * x + (1.0 - alpha) * x * x

    assert torch.allclose(res, expected)


def test_complex_tanh() -> None:
    torch = pytest.importorskip("torch")

    from nneuroutil.torch_extras import ComplexTanh

    x_real = torch.randn(5, 5)
    x_imag = torch.randn(5, 5)
    x = torch.complex(x_real, x_imag)

    mod = ComplexTanh()
    res = mod(x)
    expected = torch.complex(torch.tanh(x_real), torch.tanh(x_imag))

    assert torch.allclose(res, expected)


# }}}


# {{{ test_layers


def test_bias() -> None:
    torch = pytest.importorskip("torch")

    from nneuroutil.torch_extras import Bias

    size = 10
    mod = Bias(size)

    assert mod.bias.shape == (size,)
    assert torch.allclose(mod.bias, torch.zeros_like(mod.bias))


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear(bias: bool) -> None:  # noqa: FBT001
    torch = pytest.importorskip("torch")

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
        def to_real(self) -> torch.dtype:  # noqa: PLR6301
            return torch.float32

    mod_custom_dtype = ComplexLinear(
        in_features, out_features, bias=bias, dtype=MockDtype()
    )
    assert mod_custom_dtype.weight.dtype == torch.float32


# }}}


# {{{ test_init


@pytest.mark.parametrize("nonlinearity", ["quadratic", "leaky_quadratic", "relu"])
@pytest.mark.parametrize("param", [None, 0.0, 0.2])
@pytest.mark.parametrize("mode", ["fan_in", "fan_out"])
def test_kaiming_init(
    nonlinearity: str,
    param: float | None,
    mode: Literal["fan_in", "fan_out"],
) -> None:
    torch = pytest.importorskip("torch")

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
