# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from collections.abc import Callable

from rich.table import Table

from nneuroutil.helpers import module_logger, stringify_table

log = module_logger(__name__)

try:
    from torch import nn
except ImportError:
    log.error("This example requires 'pytorch'.")
    raise SystemExit(0) from None

from nneuroutil.torch_extras import (
    BlendedQuadratic,
    gather_model_signal_statistics,
    kaiming_uniform_,
)


def make_mlp(
    template: nn.Module,
    init_fn: Callable[[nn.Module], None],
    *,
    depth: int = 10,
    alpha: float = 0.85,
) -> nn.Module:
    layers = []
    for _ in range(depth):
        layer = copy.deepcopy(template)
        init_fn(layer.weight)
        layers.append(layer)

        layers.append(BlendedQuadratic(alpha))

    return nn.Sequential(*layers)


# {{{ make testing MLPs

layer_name = "Linear"
funca_name = "BlendedQuadratic"

# NOTE: alpha << 0.75 will explode for depth=10 because of the quadratic, but
# larger values seem to keep it stable for longer.
n = 512
depth = 10
alpha = 0.8

layer = nn.Linear(n, n, bias=False)
default_model = make_mlp(
    layer,
    lambda w: None,
    depth=depth,
)
activated_model = make_mlp(
    layer,
    lambda w: kaiming_uniform_(w, nonlinearity="linear_quadratic", param=alpha),
    depth=depth,
    alpha=alpha,
)

# }}}

# {{{ gather statistics

default_stats = gather_model_signal_statistics(default_model, (n,))
activated_stats = gather_model_signal_statistics(activated_model, (n,))

table = Table("depth", "mean", "var", "msq")
for name, stats in default_stats.items():
    depth = int(name)
    lid = f"{depth // 2}/{layer_name}" if depth % 2 == 0 else f"  {funca_name}"
    table.add_row(lid, f"{stats.mean:9.5f}", f"{stats.var:9.5f}", f"{stats.msq:9.5f}")
log.info("Default:\n%s", stringify_table(table))

table = Table("depth", "mean", "var", "msq")
for name, stats in activated_stats.items():
    depth = int(name)
    lid = f"{depth // 2}/{layer_name}" if depth % 2 == 0 else f"  {funca_name}"
    table.add_row(lid, f"{stats.mean:9.5f}", f"{stats.var:9.5f}", f"{stats.msq:9.5f}")
log.info("Activated:\n%s", stringify_table(table))

# }}}
