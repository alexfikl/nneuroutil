# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest

from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array1D, Array2D, ArrayND

log = module_logger(__name__)


def simulate_linear_system(
    nsnapshots: int,
    ndim: int,
    udim: int,
    *,
    xp: Any,
    rng: np.random.Generator,
) -> tuple[
    Array1D[np.floating[Any]],
    Array2D[np.floating[Any]],
    Array2D[np.floating[Any]],
    Array2D[np.floating[Any]],
]:
    A = rng.standard_normal((ndim, ndim)) / ndim
    B = rng.standard_normal((udim, ndim)) / ndim

    X = np.zeros((nsnapshots, ndim))
    U = rng.standard_normal((nsnapshots, udim))
    X[0] = rng.standard_normal(ndim)
    for k in range(nsnapshots - 1):
        X[k + 1] = X[k] @ A + U[k] @ B

    return xp.asarray(X), xp.asarray(U), xp.asarray(A), xp.asarray(B)


# {{{ test_build_full_dmdc


@pytest.mark.parametrize("method", ["pinv", "ridge"])
@pytest.mark.parametrize("nsnapshots", [8, 32])
def test_build_full_dmdc(
    xp: Any, nsnapshots: int, method: Literal["pinv", "ridge"]
) -> None:
    rng = np.random.default_rng(seed=42)
    ndim, udim = 5, 2

    from nneuroutil.dmdc import build_full_dmdc, fit_residual

    # NOTE: the snapshot counts hit both the QR and SVD branches of "pinv"
    X, U, A_ref, B_ref = simulate_linear_system(nsnapshots, ndim, udim, xp=xp, rng=rng)

    dmdc = build_full_dmdc(X, U, method=method, xp=xp)

    assert dmdc.physical_dim == ndim
    assert dmdc.lifted_dim == ndim
    assert dmdc.control_dim == udim
    assert dmdc.A.shape == (ndim, ndim)
    assert dmdc.B.shape == (udim, ndim)

    # the noiseless system is recovered up to solver accuracy
    error_A = xp.linalg.norm(dmdc.A - A_ref) / xp.linalg.norm(A_ref)
    error_B = xp.linalg.norm(dmdc.B - B_ref) / xp.linalg.norm(B_ref)
    log.info(
        "[%s] full DMDc operator errors: A %.3e B %.3e (method=%s nsnapshots=%d)",
        xp.__name__,
        error_A,
        error_B,
        method,
        nsnapshots,
    )
    assert error_A < 1.0e-10
    assert error_B < 1.0e-10

    residual = fit_residual(dmdc, X[:-1], U[:-1], X[1:])
    log.info("[%s] full DMDc fit residual: %.3e", xp.__name__, residual)
    assert residual < 1.0e-10  # ty: ignore[unsupported-operator]


def test_build_full_dmdc_predict(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    ndim, udim = 5, 2
    nsnapshots, maxit = 16, 10

    from nneuroutil.dmdc import build_full_dmdc

    X, U, A_ref, _ = simulate_linear_system(nsnapshots, ndim, udim, xp=xp, rng=rng)
    dmdc = build_full_dmdc(X, U, xp=xp)

    # time-varying control: one control input per time step
    Xpred = dmdc.predict(X[0], U[:maxit], maxit, full=True)
    assert Xpred.shape == (maxit + 1, ndim)

    error = xp.linalg.norm(Xpred - X[: maxit + 1]) / xp.linalg.norm(X[: maxit + 1])
    log.info(
        "[%s] full DMDc forecast error: %.3e (time-varying control)",
        xp.__name__,
        error,
    )
    assert error < 1.0e-10

    # static control: the same control input is used at every step
    A_ref_np = np.asarray(A_ref)
    Xref = np.zeros((maxit + 1, ndim))
    Xref[0] = X[0]
    for k in range(maxit):
        Xref[k + 1] = Xref[k] @ A_ref_np
    Xref = xp.asarray(Xref)

    xpred = dmdc.predict(X[0], xp.zeros(udim, dtype=dmdc.dtype), maxit)
    assert xpred.shape == (ndim,)

    error = xp.linalg.norm(xpred - Xref[-1]) / xp.linalg.norm(Xref[-1])
    log.info(
        "[%s] full DMDc forecast error: %.3e (static control)",
        xp.__name__,
        error,
    )
    assert error < 1.0e-10


def test_build_full_dmdc_errors(xp: Any) -> None:
    from nneuroutil.dmdc import build_full_dmdc

    rng = np.random.default_rng(seed=42)
    X = xp.asarray(rng.standard_normal((16, 5)))
    U = xp.asarray(rng.standard_normal((16, 2)))

    with pytest.raises(ValueError, match="must be of shape"):
        build_full_dmdc(X[..., None], U, xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_full_dmdc(X, U[..., None], xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_full_dmdc(X, U, X[..., None], xp=xp)

    with pytest.raises(ValueError, match="matching snapshot counts"):
        build_full_dmdc(X, U[:-1], X, xp=xp)

    with pytest.raises(ValueError, match="'eps' must be positive"):
        build_full_dmdc(X, U, eps=-1.0, xp=xp)

    with pytest.raises(ValueError, match="unknown method"):
        build_full_dmdc(X, U, method="garbage", xp=xp)  # ty: ignore[invalid-argument-type]


# }}}


# {{{ test_build_full_extended_dmdc


@pytest.mark.parametrize("method", ["pinv", "ridge"])
@pytest.mark.parametrize("first_observable_is_state", [False, True])
@pytest.mark.parametrize("use_trajectory", [False, True])
def test_build_full_extended_dmdc(
    xp: Any,
    method: Literal["pinv", "ridge"],
    *,
    first_observable_is_state: bool,
    use_trajectory: bool,
) -> None:
    rng = np.random.default_rng(seed=42)
    nsnapshots = 12

    # the lifted state [x1, x2, x2^2] has exactly linear controlled dynamics:
    # x1_{k+1} = 2 x1_k + u_k and x2_{k+1} = 2 x2_k
    X = np.zeros((nsnapshots, 2))
    X[0] = [0.3, 0.7]
    U = 0.1 * rng.standard_normal((nsnapshots, 1))

    for k in range(nsnapshots - 1):
        X[k + 1, 0] = 2.0 * X[k, 0] + U[k, 0]
        X[k + 1, 1] = 2.0 * X[k, 1]

    X = xp.asarray(X)
    U = xp.asarray(U)

    def identity(x: ArrayND[np.floating[Any]]) -> ArrayND[np.floating[Any]]:
        return x

    def square_second(x: ArrayND[np.floating[Any]]) -> ArrayND[np.floating[Any]]:
        return x[..., 1:2] ** 2

    from nneuroutil.dmdc import build_full_extended_dmdc

    observables = [identity, square_second]
    if use_trajectory:
        dmdc = build_full_extended_dmdc(
            observables,
            X,
            U,
            method=method,
            first_observable_is_state=first_observable_is_state,
            xp=xp,
        )
    else:
        dmdc = build_full_extended_dmdc(
            observables,
            X[:-1],
            U[:-1],
            X[1:],
            method=method,
            first_observable_is_state=first_observable_is_state,
            xp=xp,
        )

    assert dmdc.physical_dim == 2
    assert dmdc.lifted_dim == 3
    assert dmdc.control_dim == 1

    A_ref = xp.asarray([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    B_ref = xp.asarray([[1.0, 0.0, 0.0]])
    C_ref = xp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    error_A = xp.linalg.norm(dmdc.A - A_ref)
    error_B = xp.linalg.norm(dmdc.B - B_ref)
    error_C = xp.linalg.norm(dmdc.C - C_ref)
    log.info(
        "[%s] extended DMDc operator errors: A %.3e B %.3e C %.3e "
        "(method=%s first_observable_is_state=%s use_trajectory=%s)",
        xp.__name__,
        error_A,
        error_B,
        error_C,
        method,
        first_observable_is_state,
        use_trajectory,
    )
    assert error_A < 1.0e-8
    assert error_B < 1.0e-8
    assert error_C < 1.0e-8

    maxit = 6
    Xpred = dmdc.predict(X[0], U[:maxit], maxit, full=True)
    assert Xpred.shape == (maxit + 1, 2)

    error = xp.linalg.norm(Xpred - X[: maxit + 1]) / xp.linalg.norm(X[: maxit + 1])
    log.info("[%s] extended DMDc forecast error: %.3e", xp.__name__, error)
    assert error < 1.0e-9


def test_build_full_extended_dmdc_errors(xp: Any) -> None:
    from nneuroutil.dmdc import build_full_extended_dmdc

    rng = np.random.default_rng(seed=42)
    X = xp.asarray(rng.standard_normal((16, 2)))
    U = xp.asarray(rng.standard_normal((16, 1)))

    with pytest.raises(ValueError, match="must be of shape"):
        build_full_extended_dmdc([lambda z: z], X[..., None], U, xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_full_extended_dmdc([lambda z: z], X, U[..., None], xp=xp)

    with pytest.raises(ValueError, match="no 'observables'"):
        build_full_extended_dmdc([], X, U, xp=xp)


# }}}


# {{{ test_diagnostics


def test_relative_forecast_error(xp: Any) -> None:
    from nneuroutil.dmdc import build_full_dmdc, relative_forecast_error

    rng = np.random.default_rng(seed=42)
    ndim, udim = 5, 2
    nsnapshots = 32

    X, U, _, _ = simulate_linear_system(nsnapshots, ndim, udim, xp=xp, rng=rng)
    dmdc = build_full_dmdc(xp.asarray(X), xp.asarray(U), xp=xp)

    # NOTE: the forecast uses one control input per time step
    Xa = xp.asarray(X)
    Uc = xp.asarray(U[:-1])

    err = relative_forecast_error(dmdc, Xa, Uc)
    assert err.shape == (nsnapshots,)
    log.info(
        "[%s] full DMDc forecast error (max): %.3e",
        xp.__name__,
        float(xp.max(err)),
    )
    assert float(xp.max(err)) < 1.0e-10

    maxit = 10
    err_maxit = relative_forecast_error(dmdc, Xa, Uc[:maxit], maxit=maxit)
    assert err_maxit.shape == (maxit + 1,)
    assert float(xp.max(err_maxit)) < 1.0e-10

    Xpred = dmdc.predict(Xa[0], Uc, nsnapshots - 1, full=True)
    err_pred = relative_forecast_error(dmdc, Xa, Uc, xp.asarray(Xpred))
    assert err_pred.shape == (nsnapshots,)
    assert float(xp.max(xp.abs(err_pred - err))) < 1.0e-14

    with pytest.raises(ValueError, match="'maxit' must be in"):
        relative_forecast_error(dmdc, Xa, Uc, maxit=0)

    with pytest.raises(ValueError, match="'maxit' must be in"):
        relative_forecast_error(dmdc, Xa, Uc, maxit=nsnapshots)

    with pytest.raises(ValueError, match="'U' must have"):
        relative_forecast_error(dmdc, Xa, Uc[:-1])

    with pytest.raises(ValueError, match="'U' must have"):
        relative_forecast_error(dmdc, Xa, Uc[:5], maxit=10)


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
