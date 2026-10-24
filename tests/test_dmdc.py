# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pytest

from nneuroutil.typing import Array1D, Array2D, ArrayND

log = logging.getLogger(__name__)


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


def simulate_symmetric_linear_system(
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
    # NOTE: a symmetric random matrix can easily have a spectral radius above
    # one, so it is scaled to keep the simulated trajectory from blowing up.
    A = rng.standard_normal((ndim, ndim))
    A = A + A.T
    A = 0.8 * A / np.linalg.norm(A, 2)
    B = rng.standard_normal((udim, ndim)) / ndim

    X = np.zeros((nsnapshots, ndim))
    U = rng.standard_normal((nsnapshots, udim))
    X[0] = rng.standard_normal(ndim)
    for k in range(nsnapshots - 1):
        X[k + 1] = X[k] @ A + U[k] @ B

    return xp.asarray(X), xp.asarray(U), xp.asarray(A), xp.asarray(B)


# {{{ test_build_dense_dmdc


@pytest.mark.parametrize("method", ["pinv", "ridge"])
@pytest.mark.parametrize("nsnapshots", [8, 32])
def test_build_dense_dmdc(
    xp: Any, nsnapshots: int, method: Literal["pinv", "ridge"]
) -> None:
    rng = np.random.default_rng(seed=42)
    ndim, udim = 5, 2

    from nneuroutil.dmdc import build_dense_dmdc, fit_residual

    # NOTE: the snapshot counts hit both the QR and SVD branches of "pinv"
    X, U, A_ref, B_ref = simulate_linear_system(nsnapshots, ndim, udim, xp=xp, rng=rng)

    dmdc = build_dense_dmdc(X, U, method=method, xp=xp)

    assert dmdc.physical_dim == ndim
    assert dmdc.lifted_dim == ndim
    assert dmdc.control_dim == udim
    assert dmdc.A.shape == (ndim, ndim)
    assert dmdc.B.shape == (udim, ndim)

    # the noiseless system is recovered up to solver accuracy
    error_A = xp.linalg.norm(dmdc.A - A_ref) / xp.linalg.norm(A_ref)
    error_B = xp.linalg.norm(dmdc.B - B_ref) / xp.linalg.norm(B_ref)
    log.info(
        "[%s] Dense DMDc operator errors: A %.3e B %.3e (method=%s nsnapshots=%d)",
        xp.__name__,
        error_A,
        error_B,
        method,
        nsnapshots,
    )
    assert error_A < 1.0e-10
    assert error_B < 1.0e-10

    residual = fit_residual(dmdc, X[:-1], U[:-1], X[1:])
    log.info("[%s] Dense DMDc fit residual: %.3e", xp.__name__, residual)
    assert residual < 1.0e-10  # ty: ignore[unsupported-operator]


def test_build_dense_dmdc_predict(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)
    ndim, udim = 5, 2
    nsnapshots, maxit = 16, 10

    from nneuroutil.dmdc import build_dense_dmdc

    X, U, A_ref, _ = simulate_linear_system(nsnapshots, ndim, udim, xp=xp, rng=rng)
    dmdc = build_dense_dmdc(X, U, xp=xp)

    # time-varying control: one control input per time step
    Xpred = dmdc.predict(X[0], U[:maxit], maxit, full=True)
    assert Xpred.shape == (maxit + 1, ndim)

    error = xp.linalg.norm(Xpred - X[: maxit + 1]) / xp.linalg.norm(X[: maxit + 1])
    log.info(
        "[%s] Dense DMDc forecast error: %.3e (time-varying control)",
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
        "[%s] Dense DMDc forecast error: %.3e (static control)",
        xp.__name__,
        error,
    )
    assert error < 1.0e-10


def test_build_dense_dmdc_errors(xp: Any) -> None:
    from nneuroutil.dmdc import build_dense_dmdc

    rng = np.random.default_rng(seed=42)
    X = xp.asarray(rng.standard_normal((16, 5)))
    U = xp.asarray(rng.standard_normal((16, 2)))

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_dmdc(X[..., None], U, xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_dmdc(X, U[..., None], xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_dmdc(X, U, X[..., None], xp=xp)

    with pytest.raises(ValueError, match="matching snapshot counts"):
        build_dense_dmdc(X, U[:-1], X, xp=xp)

    with pytest.raises(ValueError, match="'eps' must be positive"):
        build_dense_dmdc(X, U, eps=-1.0, xp=xp)

    with pytest.raises(ValueError, match="unknown method"):
        build_dense_dmdc(X, U, method="garbage", xp=xp)  # ty: ignore[invalid-argument-type]


# }}}


# {{{ test_build_dense_symmetric_dmdc


@pytest.mark.parametrize("method", ["pinv", "ridge"])
@pytest.mark.parametrize("nsnapshots", [8, 32])
@pytest.mark.parametrize("use_trajectory", [False, True])
def test_build_dense_symmetric_dmdc(
    xp: Any,
    nsnapshots: int,
    method: Literal["pinv", "ridge"],
    *,
    use_trajectory: bool,
) -> None:
    rng = np.random.default_rng(seed=42)
    ndim, udim = 5, 2

    from nneuroutil.dmdc import build_dense_symmetric_dmdc, fit_residual

    X, U, A_ref, B_ref = simulate_symmetric_linear_system(
        nsnapshots, ndim, udim, xp=xp, rng=rng
    )

    if use_trajectory:
        dmdc = build_dense_symmetric_dmdc(X, U, method=method, xp=xp)
    else:
        dmdc = build_dense_symmetric_dmdc(X[:-1], U[:-1], X[1:], method=method, xp=xp)

    assert dmdc.physical_dim == ndim
    assert dmdc.lifted_dim == ndim
    assert dmdc.control_dim == udim
    assert dmdc.A.shape == (ndim, ndim)
    assert dmdc.B.shape == (udim, ndim)

    # the state operator is symmetric up to roundoff
    error_sym = xp.linalg.norm(dmdc.A - xp.conj(dmdc.A).T)
    log.info("[%s] symmetric DMDc operator error: %.3e", xp.__name__, error_sym)
    assert error_sym < 1.0e-10

    # the noiseless system is recovered up to solver accuracy
    error_A = xp.linalg.norm(dmdc.A - A_ref) / xp.linalg.norm(A_ref)
    error_B = xp.linalg.norm(dmdc.B - B_ref) / xp.linalg.norm(B_ref)
    log.info(
        "[%s] symmetric DMDc operator errors: A %.3e B %.3e (method=%s "
        "nsnapshots=%d use_trajectory=%s)",
        xp.__name__,
        error_A,
        error_B,
        method,
        nsnapshots,
        use_trajectory,
    )
    assert error_A < 1.0e-8
    assert error_B < 1.0e-8

    residual = fit_residual(dmdc, X[:-1], U[:-1], X[1:])
    log.info("[%s] symmetric DMDc fit residual: %.3e", xp.__name__, residual)
    assert residual < 1.0e-8  # ty: ignore[unsupported-operator]


@pytest.mark.parametrize("method", ["pinv", "ridge"])
def test_build_dense_symmetric_dmdc_feasible(
    xp: Any, method: Literal["pinv", "ridge"]
) -> None:
    from nneuroutil.dmdc import (
        DenseDMDc,
        build_dense_dmdc,
        build_dense_symmetric_dmdc,
        fit_residual,
    )

    rng = np.random.default_rng(seed=7)
    X = xp.asarray(rng.standard_normal((32, 4)))
    U = xp.asarray(rng.standard_normal((32, 2)))
    Y = xp.asarray(rng.standard_normal((32, 4)))

    dmdc = build_dense_symmetric_dmdc(X, U, Y, method=method, xp=xp)

    # the state operator is symmetric by construction, even for arbitrary data
    error_sym = xp.linalg.norm(dmdc.A - xp.conj(dmdc.A).T)
    log.info("[%s] symmetric DMDc operator error: %.3e", xp.__name__, error_sym)
    assert error_sym < 1.0e-10

    # symmetrizing the unconstrained fit gives a feasible but generally
    # suboptimal symmetric operator, so it cannot fit better
    dmdc_unc = build_dense_dmdc(X, U, Y, method=method, xp=xp)
    A_feasible = 0.5 * (dmdc_unc.A + xp.conj(dmdc_unc.A).T)
    dmdc_feasible = DenseDMDc(A=A_feasible, B=dmdc_unc.B)

    residual = fit_residual(dmdc, X, U, Y)
    residual_feasible = fit_residual(dmdc_feasible, X, U, Y)
    log.info(
        "[%s] symmetric DMDc residual: %.3e (feasible: %.3e)",
        xp.__name__,
        residual,
        residual_feasible,
    )
    assert residual <= residual_feasible + 1.0e-10


def test_build_dense_symmetric_dmdc_errors(xp: Any) -> None:
    from nneuroutil.dmdc import build_dense_symmetric_dmdc

    rng = np.random.default_rng(seed=42)
    X = xp.asarray(rng.standard_normal((16, 5)))
    U = xp.asarray(rng.standard_normal((16, 2)))

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_symmetric_dmdc(X[..., None], U, xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_symmetric_dmdc(X, U[..., None], xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_symmetric_dmdc(X, U, X[..., None], xp=xp)

    with pytest.raises(ValueError, match="matching snapshot counts"):
        build_dense_symmetric_dmdc(X, U[:-1], X, xp=xp)

    # the state operator must be square, so 'X' and 'Y' need matching columns
    with pytest.raises(ValueError, match="must be square"):
        build_dense_symmetric_dmdc(X, U, X[:, :3], xp=xp)

    with pytest.raises(ValueError, match="'eps' must be positive"):
        build_dense_symmetric_dmdc(X, U, eps=-1.0, xp=xp)

    with pytest.raises(ValueError, match="unknown method"):
        build_dense_symmetric_dmdc(X, U, method="garbage", xp=xp)  # ty: ignore[invalid-argument-type]


# }}}


# {{{ test_build_dense_extended_dmdc


@pytest.mark.parametrize("method", ["pinv", "ridge"])
@pytest.mark.parametrize("first_observable_is_state", [False, True])
@pytest.mark.parametrize("use_trajectory", [False, True])
def test_build_dense_extended_dmdc(
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

    from nneuroutil.dmdc import build_dense_extended_dmdc

    observables = [identity, square_second]
    if use_trajectory:
        dmdc = build_dense_extended_dmdc(
            observables,
            X,
            U,
            method=method,
            first_observable_is_state=first_observable_is_state,
            xp=xp,
        )
    else:
        dmdc = build_dense_extended_dmdc(
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


def test_build_dense_extended_dmdc_errors(xp: Any) -> None:
    from nneuroutil.dmdc import build_dense_extended_dmdc

    rng = np.random.default_rng(seed=42)
    X = xp.asarray(rng.standard_normal((16, 2)))
    U = xp.asarray(rng.standard_normal((16, 1)))

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_extended_dmdc([lambda z: z], X[..., None], U, xp=xp)

    with pytest.raises(ValueError, match="must be of shape"):
        build_dense_extended_dmdc([lambda z: z], X, U[..., None], xp=xp)

    with pytest.raises(ValueError, match="no 'observables'"):
        build_dense_extended_dmdc([], X, U, xp=xp)


# }}}


# {{{ test_diagnostics


def test_relative_forecast_error(xp: Any) -> None:
    from nneuroutil.dmdc import build_dense_dmdc, relative_forecast_error

    rng = np.random.default_rng(seed=42)
    ndim, udim = 5, 2
    nsnapshots = 32

    X, U, _, _ = simulate_linear_system(nsnapshots, ndim, udim, xp=xp, rng=rng)
    dmdc = build_dense_dmdc(xp.asarray(X), xp.asarray(U), xp=xp)

    # NOTE: the forecast uses one control input per time step
    Xa = xp.asarray(X)
    Uc = xp.asarray(U[:-1])

    err = relative_forecast_error(dmdc, Xa, Uc)
    assert err.shape == (nsnapshots,)
    log.info(
        "[%s] Dense DMDc forecast error (max): %.3e",
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
