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
    """Treat *x* as a ``(2 d,)`` shaped complex array and compute its square.

    We assume that ``x[..., :d]`` is the real part and ``x[..., d:]`` is the
    imaginary part and compute the standard complex square.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"dimension 'd' of 'x[..., d]' must be even: {x.shape}")

    x_re, x_im = torch.chunk(x, 2, dim=-1)
    return torch.concat([x_re * x_re - x_im * x_im, 2 * x_re * x_im], dim=-1)


class Quadratic(nn.Module):
    """A quadratic :math:`x^2` activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Define the computation performed at every call."""
        return x * x


class ComplexQuadratic(nn.Module):
    """An activation function that uses :func:`complex_quadratic`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Define the computation performed at every call."""
        return complex_quadratic(x)


class BlendedQuadratic(nn.Module):
    r"""A convex combination of a linear and a quadratic activation.

    .. math::

        f(x; \alpha) = \alpha x + (1 - \alpha) x^2.
    """

    alpha: float
    """A non-learnable hyperparameter with values in :math:`[0, 1]`."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"'alpha' must be in [0, 1]: {alpha}")

        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return self.alpha * x + (1.0 - self.alpha) * x * x


class ComplexBlendedQuadratic(nn.Module):
    r"""A convex combination of a linear and a quadratic activation, where the
    quadratic term uses :func:`complex_quadratic`.

    .. math::

        f(x; \alpha) = \alpha x + (1 - \alpha) x^2.
    """

    alpha: float
    """A non-learnable hyperparameter with values in :math:`[0, 1]`."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"'alpha' must be in [0, 1]: {alpha}")

        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return self.alpha * x + (1.0 - self.alpha) * complex_quadratic(x)


class ComplexTanh(nn.Module):
    r"""A hyperbolic tangent for a complex vector of shape ``(d,)``.

    This applies the hyperbolic tangent to the real and imaginary components
    separately. Note that this is different than :math:`\tanh(z)` for a complex
    :math:`z`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Define the computation performed at every call."""
        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


# }}}


# {{{ layers


class Bias(nn.Module):
    """A bias-only layer: :math:`y = x + b`."""

    size: int
    """The size of the bias vector."""
    bias: torch.Tensor
    """The learnable bias of shape ``(size,)``. The values are initialized to zero."""

    def __init__(
        self,
        size: int,
        *,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()

        self.size = size
        self.bias = nn.Parameter(torch.zeros(size, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return x + self.bias


class ComplexLinear(nn.Module):
    """A linear layer applied to stacked complex variables.

    We assume that the input vector is of shape ``(2 d,)`` representing a
    stacked complex vector of shape ``(d,)``. This linear layer treats it as such
    and computes

    .. code:: python

        y = torch.cat([x[..., :d] @ W.T + b_re, x[..., d:] @ W.T + b_im], dim=-1)
    """

    in_features: int
    """Size of each input sample."""
    out_features: int
    """Size of each output sample."""
    weight: torch.Tensor
    """The learnable weights of the module of shape ``(out_features, in_features)``.
    The values are initialized using :func:`kaiming_uniform_`.
    """
    bias_re: torch.Tensor
    """The learnable bias of shape ``(out_features,)`` for the real part of the
    input vector. The values are initialized to 0.
    """
    bias_im: torch.Tensor
    """The learnable bias of shape ``(out_features,)`` for the imaginary part of the
    input vector. The values are initialized to 0.
    """

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
    """A symmetric linear layer."""

    in_features: int
    """Size of each input sample."""
    out_features: int
    """Size of each output sample."""

    bias: torch.Tensor | None
    """The learnable bias of shape ``(out_features,)``. The values are
    initialized to 0.
    """

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
        """The learnable weights of the module of shape ``(out_features, in_features)``.
        The values are initialized using :func:`kaiming_uniform_`.
        """
        return self._weight.triu() + self._weight.triu().transpose(-1, -2)

    def reset_parameters(self) -> None:
        # NOTE: this is the same init as nn.Linear. Turns out that symmetrizing
        # it doesn't change the row-wise variances, so the same init works here
        nn.init.kaiming_uniform_(self._weight, math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
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
    elif nonlinearity == "blended_quadratic":
        # NOTE: if not given, use the default from BlendedQuadratic
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
    elif nonlinearity == "blended_quadratic":
        # NOTE: if not given, use the default from BlendedQuadratic
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
    """A sliding window dataset for tensors of shape ``(nrealizations, maxit, d)``.

    This constructs sliding windows of the shape ``(L, d)`` for each realization.
    """

    xs: tuple[torch.Tensor, ...]
    """The tensors for which to compute sliding windows."""
    window_size: int
    """The size of each window."""
    nwindows: int
    """The number of windows per realization."""

    def __init__(
        self,
        *xs: torch.Tensor,
        window_size: int,
        nwindows: int | None = None,
    ) -> None:
        if not xs:
            raise ValueError(f"{type(self).__name__}: no tensors are provided")

        shape = xs[0].shape
        if not len(shape) == 3:
            raise ValueError(f"tensors must be 3d: {shape}")

        if not all(shape == x.shape for x in xs[1:]):
            raise ValueError("tensors must have the same shape")

        self.xs = xs
        self.window_size = window_size
        self.nrealizations, self.maxit, self.dim = shape

        if nwindows is None:
            nwindows = self.maxit - self.window_size + 1

        if nwindows < 1:
            raise ValueError(f"'nwindows' must be >= 1: got {nwindows}")

        self.nwindows = nwindows
        self.total_windows = self.nrealizations * self.nwindows
        self.window_step = (
            (self.maxit - self.window_size) / (nwindows - 1) if nwindows > 1 else 0.0
        )

        if self.window_step > self.window_size - 1:
            raise ValueError(
                "'nwindows' is too large for 'window_size': consecutive windows "
                "do not overlap (min overlap is 1)"
            )

    def __len__(self) -> int:
        return self.total_windows

    def window(self, index: int) -> slice:
        if not -len(self) <= index < len(self):
            raise IndexError(
                f"index {index} is out of bounds for dataset of length {len(self)}"
            )

        i = index % self.nwindows
        n = round(i * self.window_step)

        return slice(n, n + self.window_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        ridx = index // self.nwindows
        window = self.window(index)

        return tuple(x[ridx, window, :] for x in self.xs)


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
    """Simple statistics for a layer output."""

    mean: float
    """The mean value of the output over all dimensions."""
    var: float
    """The (unbiased) variance of the output over all dimensions."""
    msq: float
    """The second-moment :math:`E[y^2]` of the output over all dimensions."""


def gather_model_signal_statistics(
    model: nn.Module,
    shape: tuple[int, ...],
    *,
    dtype: Any = None,
    device: str | torch.device | None = None,
    batch: int = 4096,
) -> dict[str, LayerStatistics]:
    r"""Gather statistics over the whole *model* starting from a randomly
    distributed input in :math:`\mathcal{N}(0, 1)`.

    :returns: a dictionary with statistics for each module in the *model*. The
        names of the modules are expected to be unique, see
        :meth:`torch.nn.Module.named_modules`.
    """
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
