# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal, NamedTuple, TypeVar

import torch
import torch.utils._pytree as pytree  # ruff:ignore[import-private-name]
from torch import nn
from torch.utils._python_dispatch import (  # ruff:ignore[import-private-name]
    TorchDispatchMode,
)

from nneuroutil import activations
from nneuroutil.array_api_extras import view_as_complex, view_as_real
from nneuroutil.helpers import MemorySnapshot, MemoryTracker

log = logging.getLogger(__name__)


# {{{ activation functions


class Quadratic(nn.Module):
    """A wrapper around :func:`~nneuroutil.activations.quadratic`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ruff:ignore[no-self-use]
        """Define the computation performed at every call."""
        return activations.quadratic(x)


class ComplexQuadratic(nn.Module):
    """A wrapper around :func:`~nneuroutil.activations.complex_quadratic`."""

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            activations.complex_quadratic, x, interleaved=self.interleaved
        )


class BlendedQuadratic(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.blended_quadratic`.

    When *learnable* is *True*, :math:`\alpha` is parameterized as
    :math:`\alpha = \sigma(\theta)` with learnable logits :math:`\theta`.
    """

    learnable: bool
    """If *True*, :math:`\alpha` is a learnable parameter."""
    num_parameters: int
    """Number of learnable parameters (1 for a shared scalar, or number of channels)."""

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        num_parameters: int = 1,
        learnable: bool = False,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()

        self.learnable = learnable
        self.num_parameters = num_parameters

        if learnable:
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"'alpha' must be in (0, 1) exclusive: {alpha}")

            init_theta = math.log(alpha / (1.0 - alpha))
            shape = (num_parameters,) if num_parameters > 1 else ()
            self.alpha_param = nn.Parameter(
                torch.full(shape, init_theta, device=device, dtype=dtype)
            )
        else:
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"'alpha' must be in [0, 1]: {alpha}")
            self.alpha_value = alpha

    @property
    def alpha(self) -> float | torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.alpha_param)
        return self.alpha_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.blended_quadratic(x, alpha=self.alpha)


class ComplexBlendedQuadratic(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.complex_blended_quadratic`.

    When *learnable* is *True*, :math:`\alpha` is parameterized as
    :math:`\alpha = \sigma(\theta)` with learnable logits :math:`\theta`.
    """

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """
    learnable: bool
    """If *True*, :math:`\alpha` is a learnable parameter."""
    num_parameters: int
    """Number of learnable parameters (1 for a shared scalar, or number of
    complex channels).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        num_parameters: int = 1,
        learnable: bool = False,
        interleaved: bool = False,
        device: str | torch.device | None = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()

        self.interleaved = interleaved
        self.learnable = learnable
        self.num_parameters = num_parameters

        if learnable:
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"'alpha' must be in (0, 1) exclusive: {alpha}")

            init_theta = math.log(alpha / (1.0 - alpha))
            shape = (num_parameters,) if num_parameters > 1 else ()
            self.alpha_param = nn.Parameter(
                torch.full(shape, init_theta, device=device, dtype=dtype)
            )
        else:
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"'alpha' must be in [0, 1]: {alpha}")
            self.alpha_value = alpha

    @property
    def alpha(self) -> float | torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.alpha_param)
        return self.alpha_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            partial(activations.complex_blended_quadratic, alpha=self.alpha),
            x,
            interleaved=self.interleaved,
        )


class ComplexTanh(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.complex_tanh`."""

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            activations.complex_tanh, x, interleaved=self.interleaved
        )


class ModReLU(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.modrelu`."""

    bias: float
    """The bias :math:`b`. A non-learnable hyperparameter."""
    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, bias: float = 0.0, *, interleaved: bool = False) -> None:
        super().__init__()
        self.bias = bias
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            partial(activations.modrelu, b=self.bias),
            x,
            interleaved=self.interleaved,
        )


class LeakyModReLU(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.leaky_modrelu`."""

    bias: float
    """The bias :math:`b`. A non-learnable hyperparameter."""
    alpha: float
    r"""The leak slope :math:`\alpha`, with values in :math:`[0, 1]`. A
    non-learnable hyperparameter."""
    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(
        self,
        bias: float = 0.0,
        *,
        alpha: float = 0.1,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.bias = bias
        self.alpha = alpha
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            partial(activations.leaky_modrelu, b=self.bias, alpha=self.alpha),
            x,
            interleaved=self.interleaved,
        )


class ComplexCardioid(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.complex_cardioid`."""

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            activations.complex_cardioid, x, interleaved=self.interleaved
        )


class zReLU(nn.Module):  # ruff:ignore[invalid-class-name]
    r"""A wrapper around :func:`~nneuroutil.activations.zrelu`."""

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            activations.zrelu, x, interleaved=self.interleaved
        )


class ComplexLog(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.complex_log`."""

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            activations.complex_log, x, interleaved=self.interleaved
        )


class ComplexExp(nn.Module):
    r"""A wrapper around :func:`~nneuroutil.activations.complex_exp`."""

    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(self, *, interleaved: bool = False) -> None:
        super().__init__()
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return activations.apply_complex(
            activations.complex_exp, x, interleaved=self.interleaved
        )


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


class Residual(nn.Module):
    """A basic residual layer around a given *model*."""

    model: nn.Module
    """The wrapped model."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        return self.model(x) + x


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
    interleaved: bool
    """If *True*, assume tensors store complex values interleaved as
    ``[z[0].real, z[0].imag, z[1].real, ...]``. Otherwise, assume the real and
    imaginary parts are stacked as ``[z[0].real, ..., z[d - 1].real, z[0].imag, ...]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        interleaved: bool = False,
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
        self.interleaved = interleaved

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaiming uniform (same as nn.Linear default) as a safe fallback
        kaiming_uniform_(self.weight)
        if self.bias_re is not None:
            nn.init.zeros_(self.bias_re)
        if self.bias_im is not None:
            nn.init.zeros_(self.bias_im)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        # x: [batch, 2 * in_features]
        if self.interleaved:
            z = view_as_complex(x)

            bias = None
            if self.bias_re is not None:
                bias = torch.complex(self.bias_re, self.bias_im)

            result = nn.functional.linear(z, self.weight.to(z.dtype), bias)
            return view_as_real(result)
        else:
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

        self.weight = nn.Parameter(
            torch.empty(features, features, device=device, dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(features, device=device, dtype=dtype))
        else:
            self.bias = None

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call."""
        weight = self.weight.triu() + self.weight.triu().transpose(-1, -2)
        return nn.functional.linear(x, weight, self.bias)


# }}}


# {{{ init

NONLINEARITY_TYPE_NAME = {
    BlendedQuadratic: "blended_quadratic",
    ComplexBlendedQuadratic: "cblended_quadratic",
    ComplexCardioid: "ccardioid",
    ComplexQuadratic: "cquadratic",
    ComplexTanh: "ctanh",
    LeakyModReLU: "leaky_modrelu",
    ModReLU: "modrelu",
    Quadratic: "quadratic",
    zReLU: "zrelu",
}

NONLINEARITY_TYPE = {name: cls for cls, name in NONLINEARITY_TYPE_NAME.items()}


def kaiming_uniform_(
    x: torch.Tensor,
    *,
    nonlinearity: str = "relu",
    param: float | None = None,
    paramb: float | None = None,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Fill the input tensor *x* using a Kaiming uniform distribution.

    This does not use :func:`torch.nn.init.kaiming_uniform_`, but works the
    same and supports the custom activation functions defined here. See
    :ref:`notes-initializers` for a derivation and other aspects of
    initialization. We assume that the inputs to each layer are

    * for real inputs: Gaussian with zero mean and unit variance.
    * for complex inputs :math:`z = r e^{\mathrm{i} \theta}`: :math:`r` is
      Rayleigh with scale :math:`1/\sqrt{2}` and :math:`\theta` is uniform on
      :math:`[0, 2 \pi]`. This means that :math:`\mathbb{E}[z \bar{z}] = 1` and
      **not** :math:`\mathbb{E}[z z] = 1`!

    :arg param: parameter used in activations functions that require it, such as
        "blended_quadratic", "leaky_relu", etc.
    :arg paramb: additional parameter for "leaky_modrelu".
    """
    from nneuroutil.helpers import calculate_gain

    # FIXME: this is pretty horrible, need a better api or something
    kwargs = {}
    if nonlinearity in {"blended_quadratic", "cblended_quadratic", "leaky_relu"}:
        if param is not None:
            kwargs["alpha"] = param
    elif nonlinearity in {"modrelu"}:
        if param is not None:
            kwargs["b"] = param
    elif nonlinearity in {"leaky_modrelu"}:
        if param is not None:
            kwargs["b"] = param

        if paramb is not None:
            kwargs["alpha"] = paramb

    fan = nn.init._calculate_correct_fan(x, mode)
    gain = calculate_gain(nonlinearity, **kwargs)

    with torch.no_grad():
        bound = math.sqrt(3.0 * gain / fan)
        return nn.init.uniform_(x, -bound, bound, generator=generator)


def kaiming_normal_(
    x: torch.Tensor,
    *,
    nonlinearity: str = "quadratic",
    param: float | None = None,
    paramb: float | None = None,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Fill the input tensor *x* using a Kaiming normal distribution.

    This does not use :func:`torch.nn.init.kaiming_normal_`, but works the
    same and supports the custom activation functions defined here. The input
    assumptions match those in :func:`kaiming_uniform_`.

    :arg param: parameter used in activations functions that require it, such as
        "blended_quadratic", "leaky_relu", etc.
    :arg paramb: additional parameter for "leaky_modrelu".
    """
    from nneuroutil.helpers import calculate_gain

    # FIXME: this is pretty horrible, need a better api or something
    kwargs = {}
    if nonlinearity in {"blended_quadratic", "cblended_quadratic", "leaky_relu"}:
        if param is not None:
            kwargs["alpha"] = param
    elif nonlinearity in {"modrelu"}:
        if param is not None:
            kwargs["b"] = param
    elif nonlinearity in {"leaky_modrelu"}:
        if param is not None:
            kwargs["b"] = param

        if paramb is not None:
            kwargs["alpha"] = paramb

    fan = nn.init._calculate_correct_fan(x, mode)
    gain = calculate_gain(nonlinearity, **kwargs)

    with torch.no_grad():
        std = math.sqrt(gain / fan)
        return nn.init.normal_(x, 0.0, std, generator=generator)


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


@contextmanager
def torch_default_device(
    device: str | torch.device | None = None,
) -> Generator[torch.device]:
    """Context manager to set the default device.

    Newer versions of ``pytorch`` can use ``torch.device`` for this.
    """
    prev_device = torch.get_default_device()

    torch.set_default_device(device)
    new_device = torch.get_default_device()
    assert new_device == get_default_device()

    try:
        yield new_device
    finally:
        torch.set_default_device(prev_device)


def get_default_device() -> torch.device:
    """Get the default device used on tensor creation.

    This creates a dummy tensor and returns its device. It should be equivalent
    to `torch.get_default_device()`, but that does not seem to work across versions.
    """
    return torch.zeros(1).device


# }}}


# {{{ TorchMemoryTracker


class TorchMemoryTracker(MemoryTracker):
    """A :class:`~nneuroutil.helpers.MemoryTracker` that also records the CUDA
    memory usage of *device*.
    """

    def __init__(self, device: Any = None) -> None:
        super().__init__(device)

    def make_record(self, tag: str, *, stacklevel: int = 2) -> MemorySnapshot:
        mem = super().make_record(tag, stacklevel=stacklevel + 1)
        if getattr(self.device, "type", "") != "cuda":
            return mem

        bytes_in_use = torch.cuda.memory_allocated(self.device)
        peak_bytes_in_use = torch.cuda.max_memory_allocated(self.device)
        prev_bytes_in_use = (
            self.snapshots[-1].memory["CUDA"] if self.snapshots else bytes_in_use
        )

        mem.memory.update({
            "CUDA": bytes_in_use,
            "Delta CUDA": bytes_in_use - prev_bytes_in_use,
            "Peak CUDA": peak_bytes_in_use,
        })

        return mem


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

    def __torch_dispatch__(  # ruff:ignore[bad-dunder-method-name]
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


def rayleigh(
    *shape: int,
    sigma: float = 1.0 / math.sqrt(2),
    device: str | torch.device | None = None,
    dtype: Any = None,
) -> torch.Tensor:
    U = torch.rand(*shape, device=device, dtype=dtype)
    torch.clamp_min_(U, 1.0e-12)

    return sigma * torch.sqrt(-2.0 * torch.log(U))


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

    For complex dtypes, the input is a circularly-symmetric complex normal
    distribution with :math:`\mathbb{E}[\|z\|^2] = 1` and all tensors are
    assumed to store complex values interleaved (see :func:`view_as_complex`).
    The statistics are computed per complex entry: the mean is the modulus of
    the complex mean, while the variance and second moment are computed from
    :math:`\|z\|^2`.

    :returns: a dictionary with statistics for each module in the *model*. The
        names of the modules are expected to be unique, see
        :meth:`torch.nn.Module.named_modules`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    stats = {}

    def hook(name: str) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def fn(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
            o = out.detach()
            if o.is_complex() or dtype.is_complex:
                z = o if o.is_complex() else view_as_complex(o)
                stats[name] = LayerStatistics(
                    mean=torch.abs(torch.mean(z)).item(),
                    var=torch.var(z, correction=1).item(),
                    msq=torch.mean(torch.pow(torch.abs(z), 2)).item(),
                )
            else:
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

    if dtype.is_floating_point:
        x = torch.randn(batch, *shape, device=device, dtype=dtype)
    else:
        r = rayleigh(batch, *shape, device=device, dtype=dtype.to_real())
        theta = (2.0 * torch.rand_like(r) - 1.0) * math.pi
        x = view_as_real(torch.polar(r, theta))

    model(x)

    # remove remove hooks from the layers
    for h in handles:
        h.remove()

    return stats


# }}}
