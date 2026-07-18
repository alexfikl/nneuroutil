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
kwargs = {"bias": param, "alpha": 0.1}
# kwargs = {}

# }}}

# {{{ make testing MLPs


def make_mlp(
    template: nn.Module,
    init_fn: Callable[[nn.Module], None],
    *,
    depth: int = 10,
) -> nn.Module:
    layers = []
    for _ in range(depth):
        layer = copy.deepcopy(template)
        init_fn(layer.weight)
        layers.append(layer)

        layers.append(activation_cls(**kwargs))  # ty: ignore[invalid-argument-type]

    return nn.Sequential(*layers)


n = 256
layer = nnx.ComplexLinear(n, n, interleaved=True, bias=False)
# layer = nn.Linear(2 * n, 2 * n, bias=False)
default_model = make_mlp(
    layer,
    lambda w: None,
)
activated_model = make_mlp(
    layer,
    lambda w: nnx.complex_kaiming_uniform_(w, nonlinearity=activation_id, param=param),
)

# }}}

# {{{ gather statistics

default_stats = nnx.gather_model_signal_statistics(
    default_model, (n,), dtype=torch.complex64
)
activated_stats = nnx.gather_model_signal_statistics(
    activated_model, (n,), dtype=torch.complex64
)

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
