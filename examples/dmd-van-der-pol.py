# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
from rich.table import Table
from scipy.integrate import solve_ivp

from nneuroutil import dmd
from nneuroutil.helpers import module_logger, stringify_table
from nneuroutil.typing import Array2D

log = module_logger(__name__)
rng = np.random.default_rng(seed=42)


# {{{ functions


mu = 2.0


def van_der_pol(t: float, x: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
    return np.array([
        x[1],
        mu * (1.0 - x[0] ** 2) * x[1] - x[0],
    ])


def g_identity(x: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
    return x


def g_sqr(x: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
    return np.concat([x**2, x[..., :1] * x[..., 1:]], axis=-1)


def g_cubic(x: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
    return np.concat(
        [
            x**3,
            x[..., :1] * x[..., 1:] ** 2,
            x[..., :1] ** 2 * x[..., 1:],
        ],
        axis=-1,
    )


# }}}

# {{{ evolve

tspan = (0.0, 25.0)
dt = 0.1

x0 = np.array([2.0, 0.0])
t = np.arange(tspan[0], tspan[1] + dt, dt)

result = solve_ivp(
    van_der_pol,
    tspan,
    x0,
    method="RK45",
    t_eval=t,
    rtol=1.0e-10,
    atol=1.0e-12,
    max_step=dt,
)

X = result.y.T
nsnapshots, ndim = X.shape

log.info("Snapshot matrix: (%d, %d)", nsnapshots, ndim)

# rank selection from the snapshot singular values (should always be 2)
_, sv, _ = np.linalg.svd(X, full_matrices=False)
ce = dmd.cumulative_energy(sv)
rank = np.argmax(ce >= 0.999) + 1  # ty: ignore[unsupported-operator]

log.info("Singular values: %s", sv)
log.info("Rank for 99.9%% energy: %d", rank)

# }}}

# {{{ add noise

sigma = 0.01 * np.linalg.norm(X) / np.sqrt(X.size)
X_noisy = X + sigma * rng.standard_normal(X.shape)
X1, X2 = X_noisy[:-1], X_noisy[1:]

# }}}

# {{{ fit models

models: dict[str, Any] = {}
models["DMD"] = dmd.build_full_dmd(X1, X2, method="ridge")
models["fbDMD"] = dmd.build_forward_backward_dmd(X1, X2, method="ridge")
models["EDMD"] = dmd.build_full_extended_dmd(
    [g_identity, g_sqr, g_cubic],
    X1,
    X2,
    first_observable_is_state=True,
    method="ridge",
)

# }}}

# {{{ gather statistics

horizon = int(15.0 / dt)

table = Table("model", "fit residual", "forecast error")
for name, model in models.items():
    resid = dmd.fit_residual(model, X1, X2)
    err = dmd.relative_forecast_error(model, X1)
    table.add_row(name, f"{resid:9.5e}", f"{err[-horizon:].mean():9.5e}")

log.info("Metrics:\n%s", stringify_table(table))

# }}}

# {{{ plot

from nneuroutil.helpers import on_ci

if on_ci():
    raise SystemExit(0)

try:
    import matplotlib.pyplot as mp  # ruff:ignore[unused-import]
except ImportError:
    raise SystemExit(0) from None

from nneuroutil.visualization import figure, set_plotting_defaults

dirname = pathlib.Path(__file__).parent
set_plotting_defaults()

styles = ["", "--", "-.", ":"]

with figure(
    dirname / "dmd_van_der_pol",
    nrows=3,
    figsize=(10, 12),
    overwrite=True,
) as fig:
    ax1, ax2, ax3 = fig.axes

    ax1.plot(X[:, 0], X[:, 1], color="k", label="reference")
    for i, (name, model) in enumerate(models.items()):
        x_pred = model.predict(X1[0], horizon, full=True)
        ax1.plot(
            x_pred[:, 0],
            x_pred[:, 1],
            styles[i % len(styles)],
            label=name,
        )

    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.legend(loc="upper right")

    ax2.plot(t[:-1], X1[:, 0], "k.", markersize=2, label="noisy")
    for i, (name, model) in enumerate(models.items()):
        x_pred = model.predict(X1[0], horizon, full=True)
        ax2.plot(t[: horizon + 1], x_pred[:, 0], styles[i % len(styles)], label=name)

    ax2.set_xlim(t[0], t[-1])
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$x(t)$")
    ax2.legend(loc="upper right")

    for i, (name, model) in enumerate(models.items()):
        err = dmd.relative_forecast_error(model, X1)
        ax3.semilogy(t[:-1], err, styles[i % len(styles)], label=name)

    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(bottom=1.0e-16)
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$\|e_k\| / \|X\|$")
    ax3.legend(loc="upper right")

# }}}
