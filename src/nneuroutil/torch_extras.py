# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, TypeVar

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC2701
from torch.utils.data import Dataset

from nneuroutil.helpers import module_logger

log = module_logger(__name__)


# {{{ activation functions


def complex_quadratic(x: torch.Tensor) -> torch.Tensor:
    x_re, x_im = torch.chunk(x, 2, dim=-1)
    return torch.concat([x_re * x_re - x_im * x_im, 2 * x_re * x_im], dim=-1)


class Quadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Define the computation performed at every call."""
        return x * x


class ComplexQuadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Define the computation performed at every call."""
        return complex_quadratic(x)


class LeakyQuadratic(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return self.alpha * x + (1.0 - self.alpha) * x * x


class ComplexLeakyQuadratic(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return self.alpha * x + (1.0 - self.alpha) * complex_quadratic(x)


class ComplexTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Define the computation performed at every call."""
        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


# }}}


# {{{ layers


class Bias(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return x + self.bias


class ComplexLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
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
        x_re, x_im = x.chunk(2, dim=-1)

        result_re = nn.functional.linear(x_re, self.weight, self.bias_re)
        result_im = nn.functional.linear(x_im, self.weight, self.bias_im)

        return torch.cat([result_re, result_im], dim=-1)


class SymmetricLinear(nn.Module):
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
        return self._weight.triu() + self._weight.triu().transpose(-1, -2)

    def reset_parameters(self) -> None:
        # NOTE: this is the same init as nn.Linear. Turns out that symmetrizing
        # it doesn't change the row-wise variances, so the same init works here
        nn.init.kaiming_uniform_(self._weight, math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


# }}}


# {{{ init


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
    elif nonlinearity == "leaky_quadratic":
        # NOTE: if not given, use the default from LeakyQuadratic
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
    elif nonlinearity == "leaky_quadratic":
        # NOTE: if not given, use the default from LeakyQuadratic
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


# {{{ SlidingWindowDataset


class SlidingWindowDataset(Dataset):
    def __init__(self, x: torch.Tensor, window_size: int) -> None:
        self.x = x
        self.window_size = window_size

        self.nrealizations, self.maxit, self.dim = x.shape
        self.windows_per_realization = self.maxit - self.window_size + 1
        self.total_windows = self.nrealizations * self.windows_per_realization

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int) -> torch.Tensor:
        ridx = index // self.windows_per_realization
        n = index % self.windows_per_realization

        window = self.x[ridx, n : n + self.window_size, :]
        return window


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

    def __torch_dispatch__(  # noqa: PLW3201
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
    mean: float
    var: float
    msq: float


def gather_model_signal_statistics(
    model: nn.Module,
    shape: tuple[int, ...],
    *,
    dtype: Any = None,
    device: str | torch.device | None = None,
    batch: int = 4096,
) -> dict[str, LayerStatistics]:
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
    x = torch.randn(batch, *shape, device=device, dtype=dtype)
    model(x)

    # remove remove hooks from the layers
    for h in handles:
        h.remove()

    return stats


# }}}
