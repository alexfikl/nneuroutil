# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC2701

from nneuroutil.helpers import module_logger

log = module_logger(__name__)


# {{{ activation functions


def complex_quadratic(x: torch.Tensor) -> torch.Tensor:
    x_re, x_im = torch.chunk(x, 2, dim=-1)
    return torch.concat([x_re * x_re - x_im * x_im, 2 * x_re * x_im], dim=-1)


class Quadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        return x * x


class ComplexQuadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        return complex_quadratic(x)


class LeakyQuadratic(nn.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * x * x


class ComplexLeakyQuadratic(nn.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * complex_quadratic(x)


class ComplexTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


# }}}


# {{{ layers


class Bias(nn.Module):
    def __init__(self, size: int, *, dtype: Any = None) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class ComplexLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: Any | None = None,
    ):
        super().__init__()

        ftype = None if dtype is None else dtype.to_real()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=ftype))
        if bias:
            self.bias_re = nn.Parameter(torch.empty(out_features))
            self.bias_im = nn.Parameter(torch.empty(out_features))
        else:
            self.bias_re = self.bias_im = None

        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaiming uniform (same as nn.Linear default) as a safe fallback
        nn.init.kaiming_uniform_(self.weight, math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 2 * in_features]
        x_re, x_im = x.chunk(2, dim=-1)

        result_re = nn.functional.linear(x_re, self.weight, self.bias_re)
        result_im = nn.functional.linear(x_im, self.weight, self.bias_im)

        return torch.cat([result_re, result_im], dim=-1)


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
        bound = math.sqrt(3) / math.sqrt(math.sqrt(2) * fan)
        return nn.init.uniform_(x, -bound, bound, generator=generator)
    elif nonlinearity == "leaky_quadratic":
        # NOTE: if not given, we go back to a straight quadratic
        if param is None:
            param = 1.0

        if abs(param) < 1.0e-8:
            var_x = 1.0 / fan
        else:
            a = param
            b = (1 - a) ** 2
            var_x = (-b + math.sqrt(b**2 + 8 * a**2)) / (4 * a**2 * fan)

        bound = math.sqrt(3) * math.sqrt(var_x)
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
        std = 1.0 / math.sqrt(math.sqrt(2) * fan)
        return nn.init.normal_(x, 0.0, std, generator=generator)
    elif nonlinearity == "leaky_quadratic":
        # NOTE: if not given, we go back to a straight quadratic
        if param is None:
            param = 1.0

        if abs(param) < 1.0e-8:
            var_x = 1.0 / fan
        else:
            a = param
            b = (1 - a) ** 2
            var_x = (-b + math.sqrt(b**2 + 8 * a**2)) / (4 * a**2 * fan)

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


# {{{ available_devices


def available_devices() -> frozenset[torch.device]:
    """Get a list of all supported devices."""

    devices = [torch.device("cpu")]
    devices += [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    if torch.backends.mtia.is_available():
        devices.append(torch.device("mtia"))

    if torch.backends.xpu.is_available():
        devices.append(torch.device("xpu"))

    return frozenset(devices)


def available_device_names() -> frozenset[str]:
    """Get a list of available device names."""
    return frozenset([d.type for d in available_devices()])


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
