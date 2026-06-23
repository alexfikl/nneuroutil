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


# }}}


# {{{ test_layers


def test_bias() -> None:
    from nneuroutil.torch_extras import Bias

    size = 10
    mod = Bias(size)

    assert mod.bias.shape == (size,)
    assert torch.allclose(mod.bias, torch.zeros_like(mod.bias))


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear(bias: bool) -> None:  # noqa: FBT001
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

    with pytest.raises(DeviceMismatchError):  # noqa: SIM117,PT012
        with DeviceCheckMode():
            # this should pass
            x = torch.linspace(0.0, 1.0, 32, device="cpu")
            assert x.device.type == "cpu"

            # this goes through the backend and will get caught by DeviceCheckMode
            y = 2 * x
            assert y is not None


# }}}


# {{{ test_sliding_window_dataset


def test_sliding_window_dataset() -> None:
    from nneuroutil.torch_extras import SlidingWindowDataset

    nruns, maxit, dim = 2, 5, 3
    window_size = 3
    nwindows = maxit - window_size + 1

    x = torch.arange(nruns * maxit * dim, dtype=torch.float64)
    x = x.reshape(nruns, maxit, dim)

    dataset = SlidingWindowDataset(x, window_size=window_size)

    assert len(dataset) == nruns * nwindows

    # first window of first realization starts at 0
    (win,) = dataset[0]
    assert torch.allclose(win, x[0, 0:window_size, :])

    # first window of second realization
    (win,) = dataset[nwindows]
    assert torch.allclose(win, x[1, 0:window_size, :])

    # last window ends at maxit
    (win,) = dataset[-1]
    assert torch.allclose(win, x[-1, maxit - window_size : maxit, :])


# }}}


# {{{ test_sliding_window_dataset_nwindows


def test_sliding_window_dataset_nwindows() -> None:
    from nneuroutil.torch_extras import SlidingWindowDataset

    nruns, maxit, dim = 2, 10, 3
    window_size = 3
    nwindows = 4

    x = torch.arange(nruns * maxit * dim, dtype=torch.float64)
    x = x.reshape(nruns, maxit, dim)
    dataset = SlidingWindowDataset(x, window_size=window_size, nwindows=nwindows)

    assert len(dataset) == nruns * nwindows

    for r in range(nruns):
        # first window always starts at 0
        (first,) = dataset[r * nwindows]
        assert torch.allclose(first, x[r, 0:window_size, :])

        # last window always ends at maxit
        (last,) = dataset[r * nwindows + nwindows - 1]
        assert torch.allclose(last, x[r, maxit - window_size : maxit, :])

    # spot-check intermediate windows use round() spacing
    step = (maxit - window_size) / (nwindows - 1)
    for r in range(nruns):
        for i in range(nwindows):
            (win,) = dataset[r * nwindows + i]
            start = round(i * step)
            assert torch.allclose(win, x[r, start : start + window_size, :])


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
