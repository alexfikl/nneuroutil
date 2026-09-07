# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from collections.abc import Callable

from rich.table import Table

from nneuroutil.helpers import module_logger, stringify_table

log = module_logger(__name__)

try:
    import torch
    from torch import nn
except ImportError:
    log.error("This example requires 'pytorch'.")
    raise SystemExit(0) from None

import nneuroutil.torch_extras as nnx

# {{{ activation

activation_cls = nnx.LeakyModReLU
activation_id = nnx.NONLINEARITY_TYPE_NAME[activation_cls]

param = -0.5
paramb = 0.1

# kwargs = {"interleaved": True}
kwargs = {"interleaved": True, "bias": param, "alpha": paramb}
# kwargs = {"interleaved": True, "bias": param}

# }}}

# {{{ make testing MLPs


def make_mlp(
    template: nn.Module,
    init_fn: Callable[[nn.Module], None],
    *,
    depth: int = 10,
) -> nn.Module:
    layers = []
    for i in range(depth):
        layer = copy.deepcopy(template)
        if i == 0:
            # NOTE: first layer doesn't have any activation before it, so we
            # just initialize it with a linear "activation"
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
        else:
            init_fn(layer.weight)
        layers.append(layer)

        layers.append(activation_cls(**kwargs))  # ty: ignore[invalid-argument-type]

    return nn.Sequential(*layers)


n = 256
dtype = torch.complex64
layer = nnx.ComplexLinear(n, n, interleaved=True, bias=False, dtype=dtype.to_real())

# n = n if dtype.is_floating_point else 2 * n
# layer = nn.Linear(n, n, bias=False, dtype=dtype.to_real())

default_model = make_mlp(
    layer,
    lambda w: None,
)
activated_model = make_mlp(
    layer,
    lambda w: nnx.kaiming_uniform_(
        w, nonlinearity=activation_id, param=param, paramb=paramb
    ),
)

# }}}

# {{{ gather statistics

default_stats = nnx.gather_model_signal_statistics(default_model, (n,), dtype=dtype)
activated_stats = nnx.gather_model_signal_statistics(activated_model, (n,), dtype=dtype)

table = Table("depth", "mean", "var", "msq")
for name, stats in default_stats.items():
    depth = int(name)
    lid = (
        f"{depth // 2}/{type(layer).__name__}"
        if depth % 2 == 0
        else f"  {activation_cls.__name__}"
    )
    table.add_row(lid, f"{stats.mean:9.5f}", f"{stats.var:9.5f}", f"{stats.msq:9.5f}")
log.info("Default:\n%s", stringify_table(table))

table = Table("depth", "mean", "var", "msq")
for name, stats in activated_stats.items():
    depth = int(name)
    lid = (
        f"{depth // 2}/{type(layer).__name__}"
        if depth % 2 == 0
        else f"  {activation_cls.__name__}"
    )
    table.add_row(lid, f"{stats.mean:9.5f}", f"{stats.var:9.5f}", f"{stats.msq:9.5f}")
log.info("Activated:\n%s", stringify_table(table))

# }}}
