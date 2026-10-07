# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal

import array_api_compat
import numpy as np

from nneuroutil.helpers import module_logger, register_dataclass
from nneuroutil.typing import Array0D, Array1D, Array2D, ArrayND, ScalarTypeT

log = module_logger(__name__)


# {{{ DMDcBase


@register_dataclass
@dataclass(frozen=True)
class DMDcBase(ABC, Generic[ScalarTypeT]):
    """Base class for linear approximations with control in a lifted space.

    All states and control inputs are assumed to have their spatial/channel
    dimension on the last axis, e.g. snapshots of shape ``(nsnapshots, ndim)``.
    The dynamics are given by:

    .. code:: python

        xhat_next = self.evolve(self.encode(x), u)

    where :meth:`encode` maps the physical state into the lifted space,
    :attr:`~nneuroutil.dmdc.DMDcBase.A` acts on the lifted state, and
    :attr:`~nneuroutil.dmdc.DMDcBase.B` injects the control input :math:`u`.
    """

    A: Array2D[ScalarTypeT]
    """Linear evolution operator acting on the lifted state."""
    B: Array2D[ScalarTypeT]
    """Linear control operator acting on the control/parameter input."""

    @property
    def dtype(self) -> np.dtype[ScalarTypeT]:
        """The :class:`~numpy.dtype` of this operator."""
        return self.A.dtype

    @property
    def lifted_dim(self) -> int:
        """The dimension of the lifted space."""
        return self.A.shape[0]

    @property
    def control_dim(self) -> int:
        """The dimension of the control/parameter input space."""
        return self.B.shape[0]

    @property
    @abstractmethod
    def physical_dim(self) -> int:
        """The dimension of the physical state space."""

    @abstractmethod
    def encode(self, x: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        """Project the physical state ``(..., d)`` into the lifted space."""

    @abstractmethod
    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        """Project the lifted state ``(..., r)`` into the physical space."""

    @abstractmethod
    def evolve(
        self,
        xhat: ArrayND[ScalarTypeT],
        u: ArrayND[ScalarTypeT],
    ) -> ArrayND[ScalarTypeT]:
        """Advance the lifted state with control input *u* by a single time step."""

    def predict(
        self,
        x0: ArrayND[ScalarTypeT],
        u: ArrayND[ScalarTypeT],
        maxit: int,
        *,
        full: bool = False,
    ) -> ArrayND[ScalarTypeT]:
        """Evolve the initial condition *x0* with control *u* forward for *maxit* steps.

        :arg x0: initial condition with its spatial dimension on the last axis.
        :arg u: control inputs. If *u* has a leading time dimension of size *maxit*,
            the step-dependent control ``u[k]`` is used at step :math:`k`; otherwise,
            the same static control parameter *u* is used across all time steps.
        :arg maxit: number of time steps to take.
        :arg full: if ``True``, also include intermediate steps and *x0* itself;
            the result is stacked along a new leading axis. Otherwise, only the
            final state is returned.
        """
        xp = array_api_compat.array_namespace(x0, u)

        assert x0.shape[-1] == self.physical_dim
        assert u.shape[-1] == self.control_dim
        xhat = self.encode(x0)

        is_time_varying = (u.ndim == x0.ndim + 1) and (u.shape[0] == maxit)

        if full:
            result = [x0]
            for i in range(maxit):
                u_i = u[i] if is_time_varying else u
                xhat = self.evolve(xhat, u_i)
                result.append(self.decode(xhat))

            result = xp.stack(result, axis=0)
        else:
            for i in range(maxit):
                u_i = u[i] if is_time_varying else u
                xhat = self.evolve(xhat, u_i)

            result = self.decode(xhat)

        return result


# }}}


# {{{ dense DMDc


@register_dataclass
@dataclass(frozen=True)
class DenseDMDc(DMDcBase[ScalarTypeT]):
    """DMDc model for which the lifted space is the physical space itself.

    The encoding and decoding steps are both the identity and
    :attr:`~nneuroutil.dmdc.DMDcBase.A` acts directly on states of shape
    ``(..., d)`` while :attr:`~nneuroutil.dmdc.DMDcBase.B` acts on control
    inputs of shape ``(..., m)``.
    """

    @property
    def physical_dim(self) -> int:
        return self.A.shape[0]

    def encode(self, x: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert x.shape[-1] == self.physical_dim
        return x

    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim
        return xhat

    def evolve(
        self,
        xhat: ArrayND[ScalarTypeT],
        u: ArrayND[ScalarTypeT],
    ) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim
        assert u.shape[-1] == self.control_dim

        xp = array_api_compat.array_namespace(xhat, u, self.A, self.B)
        Ax = xp.einsum("...j,ji->...i", xhat, self.A)
        Bu = xp.einsum("...j,ji->...i", u, self.B)

        return Ax + Bu


def build_dense_dmdc(
    X: Array2D[ScalarTypeT],
    U: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    method: Literal["pinv", "ridge"] = "ridge",
    eps: float | None = None,
    xp: Any = None,
) -> DenseDMDc[ScalarTypeT]:
    r"""Fit a linear input-driven model :math:`Y = X A + U B` on snapshot pairs.

    :arg X: state snapshot matrix of shape ``(nsnapshots, dim_x)``.
    :arg U: control input snapshot matrix of shape ``(nsnapshots, dim_u)``.
    :arg Y: output state snapshot matrix of shape ``(nsnapshots, dim_y)``.
        If *None*, the pairs ``(X[:-1], X[1:])`` and inputs ``U[:-1]`` are used.
    :arg method: regularized solver (``"ridge"`` or ``"pinv"``).
    :arg eps: regularization tolerance.

    :returns: an approximation of the input-driven state dynamics.
    """
    if X.ndim != 2:
        raise ValueError(
            f"inputs 'X' must be of shape ``(nsnapshots, dim)``: {X.shape}"
        )

    if Y is None:
        Y = X[1:, :]
        X = X[:-1, :]
        U = U[:-1, :]

    if Y.ndim != 2:
        raise ValueError(
            f"outputs 'Y' must be of shape ``(nsnapshots, dim)``: {Y.shape}"
        )

    if U.ndim != 2:
        raise ValueError(
            f"inputs 'U' must be of shape ``(nsnapshots, dim)``: {U.shape}"
        )

    n, dx = X.shape
    _, du = U.shape
    _, dy = Y.shape
    if not (n == U.shape[0] == Y.shape[0]):
        raise ValueError(
            f"inputs 'X', 'U', and 'Y' must have matching snapshot counts: "
            f"{X.shape}, {U.shape}, and {Y.shape}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(X, U, Y)

    # Form augmented matrix Omega = [X, U]
    Omega = xp.concat([X, U], axis=1)
    d_omega = dx + du

    if eps is None:
        eps = max(n, d_omega, dy) * xp.finfo(X.dtype).eps

    if eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    # NOTE: see build_dense_dmd for math, design choices, etc.
    if method == "pinv":
        if n >= 2 * d_omega:
            Q, R = xp.linalg.qr(Omega, mode="reduced")
            Ur, S, Vh = xp.linalg.svd(R, full_matrices=False)
            S = xp.where(eps * S[0] < S, 1.0 / S, xp.zeros_like(S))

            UY = xp.conj(Ur).T @ (xp.conj(Q).T @ Y)
            VS = xp.conj(Vh).T * S
            W = VS @ UY
        else:
            U_om, S, Vh = xp.linalg.svd(Omega, full_matrices=False)
            S = xp.where(eps * S[0] < S, 1.0 / S, xp.zeros_like(S))

            UY = xp.conj(U_om).T @ Y
            VS = xp.conj(Vh).T * S
            W = VS @ UY
    elif method == "ridge":
        Q1, R1 = xp.linalg.qr(Omega, mode="reduced")
        B1 = xp.conj(Q1).T @ Y
        del Q1

        I = xp.eye(d_omega, d_omega, dtype=Omega.dtype, device=Omega.device)  # ruff: ignore[ambiguous-variable-name]
        R_aug = xp.concat([R1, eps**0.5 * I], axis=0)
        del R1

        Q2, R = xp.linalg.qr(R_aug, mode="reduced")
        del R_aug

        rhs = xp.conj(Q2[:d_omega]).T @ B1
        del Q2, B1

        W = xp.linalg.solve(R, rhs)
    else:
        raise ValueError(f"unknown method: {method!r}")

    A = W[:dx, :]
    B = W[dx:, :]

    return DenseDMDc(A=A, B=B)


# }}}


# {{{ symmetric dense DMDc


def build_dense_symmetric_dmdc(
    X: Array2D[ScalarTypeT],
    U: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    method: Literal["pinv", "ridge"] = "ridge",
    eps: float | None = None,
    xp: Any = None,
) -> DenseDMDc[ScalarTypeT]:
    r"""Fit a symmetric linear input-driven model :math:`Y = X A + U B`.

    This is the symmetric counterpart of :func:`build_dense_dmdc`, where the
    state operator is constrained to be symmetric (Hermitian for complex
    inputs), i.e.

    .. math::

        A^*, B^* = \arg\min_{A = A^H} \|Y - X A - U B\|_F^2.

    Eliminating :math:`B` by projecting out the control space spanned by
    :math:`U`, the minimizer satisfies the Sylvester equation

    .. math::

        \tilde{X}^H \tilde{X} A + A \tilde{X}^H \tilde{X}
            = \tilde{X}^H \tilde{Y} + \tilde{Y}^H \tilde{X},

    where :math:`\tilde{X} = (I - P) X` and :math:`\tilde{Y} = (I - P) Y`
    are the snapshots with the control contribution projected out.

    .. note::
        Unlike ``pinv`` in :func:`build_dense_dmdc`, the Sylvester equation
        requires forming the Gram matrix :math:`\tilde{X}^H \tilde{X}`, so the
        condition number of the data is squared.

    .. note::
        The ``ridge`` path avoids materializing the projected snapshots
        :math:`\tilde{X}` and :math:`\tilde{Y}` using the Woodbury identity.
        This is cheaper, but, like any normal-equation solve, loses accuracy
        when :math:`U` and :math:`X` are strongly correlated (i.e. when
        :math:`\|(I - P) X\| \ll \|X\|`). The ``pinv`` path forms the explicit
        projection and does not square that ratio.

    :arg X: state snapshot matrix of shape ``(nsnapshots, dim_x)``.
    :arg U: control input snapshot matrix of shape ``(nsnapshots, dim_u)``.
    :arg Y: output state snapshot matrix of shape ``(nsnapshots, dim_x)``.
        If *None*, the pairs ``(X[:-1], X[1:])`` and inputs ``U[:-1]`` are used.
    :arg method: regularized solver (``"ridge"`` or ``"pinv"``).
    :arg eps: regularization tolerance.

    :returns: a symmetric approximation of the input-driven state dynamics.
    """
    if X.ndim != 2:
        raise ValueError(
            f"inputs 'X' must be of shape ``(nsnapshots, dim)``: {X.shape}"
        )

    if Y is None:
        Y = X[1:, :]
        X = X[:-1, :]
        U = U[:-1, :]

    if Y.ndim != 2:
        raise ValueError(
            f"outputs 'Y' must be of shape ``(nsnapshots, dim)``: {Y.shape}"
        )

    if U.ndim != 2:
        raise ValueError(
            f"inputs 'U' must be of shape ``(nsnapshots, dim)``: {U.shape}"
        )

    n, dx = X.shape
    _, du = U.shape
    _, dy = Y.shape
    if not (n == U.shape[0] == Y.shape[0]):
        raise ValueError(
            f"inputs 'X', 'U', and 'Y' must have matching snapshot counts: "
            f"{X.shape}, {U.shape}, and {Y.shape}"
        )

    if dx != dy:
        raise ValueError(
            f"'A' operator must be square, so 'X' and 'Y' must have the same "
            f"number of columns: {X.shape} and {Y.shape}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(X, U, Y)

    if eps is None:
        eps = max(n, dx + du, dy) * xp.finfo(X.dtype).eps

    if eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if method == "pinv":
        # Project out the control space with the (truncated) pseudo-inverse
        #   Xt = (I - P) X,  Yt = (I - P) Y,  P = U U^+,
        # using the SVD of U directly, i.e. without forming the (du, n) U^+.
        U_u, S_u, Vh_u = xp.linalg.svd(U, full_matrices=False)
        mask_u = eps * S_u[0] < S_u
        S_u_safe = xp.where(mask_u, S_u, xp.ones_like(S_u))
        S_u_inv = xp.where(mask_u, 1.0 / S_u_safe, xp.zeros_like(S_u))

        Qx = xp.conj(U_u).T @ X
        Qy = xp.conj(U_u).T @ Y
        Xt = X - U_u @ (mask_u[:, None] * Qx)
        Yt = Y - U_u @ (mask_u[:, None] * Qy)

        # C = Xt^T Yt + Yt^T Xt only needs a single matrix product as the
        # second term is the conjugate transpose of the first.
        G = xp.conj(Xt).T @ Yt
        C = G + xp.conj(G).T

        # Solve the Sylvester equation in the right singular basis of Xt, so
        # that A = V (Ctilde / (s_i^2 + s_j^2)) V^T.
        _, S, Vh = xp.linalg.svd(Xt, full_matrices=False)
        mask_x = eps * S[0] < S
        S = xp.where(mask_x, S, xp.zeros_like(S))

        CV = Vh @ C @ xp.conj(Vh).T
        denom = S[:, None] ** 2 + S[None, :] ** 2
        valid = (S[:, None] > 0) & (S[None, :] > 0)

        At = xp.where(valid, CV / xp.where(valid, denom, 1.0), xp.zeros_like(CV))
        A = xp.conj(Vh).T @ At @ Vh

        # Recover the control operator without forming the (n, dx) residual.
        B = (xp.conj(Vh_u).T * S_u_inv) @ (Qy - Qx @ A)
    elif method == "ridge":
        I_U = xp.eye(du, du, dtype=U.dtype, device=U.device)
        S_U = xp.conj(U).T @ U + eps * I_U

        Gx = xp.conj(U).T @ X
        Gy = xp.conj(U).T @ Y
        Zx = xp.linalg.solve(S_U, Gx)
        Zy = xp.linalg.solve(S_U, Gy)

        # Solve the Sylvester equation M A + A M = C with M = Xt^T Xt + eps I and
        # C = Xt^T Yt + Yt^T Xt, where Xt = (I - P) X and Yt = (I - P) Y are
        # the projected snapshots. To avoid forming the (n, dx) projections, we
        # eliminate them with the Woodbury identity
        #   Xt^T Xt = X^T X - Gx^T Zx - eps Zx^T Zx,
        #   Xt^T Yt = X^T Y - Gx^T Zy - eps Zx^T Zy,
        # where Gx = U^T X, Gy = U^T Y and Zx = (U^T U + eps I)^{-1} Gx. The
        # terms proportional to 'eps' account for the (regularized) projection
        # P not being idempotent. Since M is symmetric, diagonalizing it gives
        # A = Q (Ctilde / (lambda_i + lambda_j)) Q^T.
        I_X = xp.eye(dx, dx, dtype=X.dtype, device=X.device)
        M = (
            xp.conj(X).T @ X
            - xp.conj(Gx).T @ Zx
            - eps * (xp.conj(Zx).T @ Zx)
            + eps * I_X
        )
        G = xp.conj(X).T @ Y - xp.conj(Gx).T @ Zy - eps * (xp.conj(Zx).T @ Zy)
        C = G + xp.conj(G).T

        lam, Q = xp.linalg.eigh(M)
        At = (xp.conj(Q).T @ C @ Q) / (lam[:, None] + lam[None, :])
        A = Q @ At @ xp.conj(Q).T

        # Recover the control operator from the (small) normal equations.
        B = xp.linalg.solve(S_U, Gy - Gx @ A)
    else:
        raise ValueError(f"unknown method: {method!r}")

    return DenseDMDc(A=A, B=B)


# }}}


# {{{ dense extended DMDc


@register_dataclass
@dataclass(frozen=True)
class DenseExtendedDMDc(DMDcBase[ScalarTypeT]):
    """DMDc model in the space of nonlinear observables with external control."""

    C: Array2D[ScalarTypeT]
    """Decoder mapping the lifted space back to the physical space."""
    observables: tuple[Callable[[ArrayND[ScalarTypeT]], ArrayND[ScalarTypeT]], ...]
    """Sequence of maps lifting the physical state into the observable space."""

    @property
    def physical_dim(self) -> int:
        return self.C.shape[1]

    def encode(self, x: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert x.shape[-1] == self.physical_dim

        xp = array_api_compat.array_namespace(x, self.A)
        return xp.concat(
            [xp.reshape(g(x), (*x.shape[:-1], -1)) for g in self.observables], axis=-1
        )

    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim

        xp = array_api_compat.array_namespace(xhat, self.C)
        return xp.einsum("...j,ji->...i", xhat, self.C)

    def evolve(
        self,
        xhat: ArrayND[ScalarTypeT],
        u: ArrayND[ScalarTypeT],
    ) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim
        assert u.shape[-1] == self.control_dim

        xp = array_api_compat.array_namespace(xhat, u, self.A, self.B)
        Ax = xp.einsum("...j,ji->...i", xhat, self.A)
        Bu = xp.einsum("...j,ji->...i", u, self.B)

        return Ax + Bu


def build_dense_extended_dmdc(
    observables: Sequence[Callable[[ArrayND[ScalarTypeT]], ArrayND[ScalarTypeT]]],
    X: Array2D[ScalarTypeT],
    U: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    method: Literal["pinv", "ridge"] = "ridge",
    first_observable_is_state: bool = False,
    eps: float | None = None,
    xp: Any = None,
) -> DenseExtendedDMDc[ScalarTypeT]:
    r"""Construct an extended DMDc approximation in the space of *observables*.

    :arg observables: sequence of maps :math:`g(x)` lifting physical states.
    :arg X: state snapshot matrix of shape ``(nsnapshots, dim_x)``.
    :arg U: control input snapshot matrix of shape ``(nsnapshots, dim_u)``.
    :arg Y: output state snapshot matrix of shape ``(nsnapshots, dim_x)``.
    :arg method: regularized solver (``"ridge"`` or ``"pinv"``).
    :arg first_observable_is_state: if ``True``, use an exact identity decoder.
    :arg eps: tolerance for the regularized solver.
    """
    if X.ndim != 2:
        raise ValueError(
            f"inputs 'X' must be of shape ``(nsnapshots, dim)``: {X.shape}"
        )

    if Y is not None and Y.ndim != 2:
        raise ValueError(
            f"outputs 'Y' must be of shape ``(nsnapshots, dim)``: {Y.shape}"
        )

    if U.ndim != 2:
        raise ValueError(
            f"inputs 'U' must be of shape ``(nsnapshots, dim)``: {U.shape}"
        )

    if not observables:
        raise ValueError("no 'observables' provided (use at least the identity map)")

    def lift(x: Array2D[ScalarTypeT], *, xp: Any) -> Array2D[ScalarTypeT]:
        return xp.concat(
            [xp.reshape(g(x), (x.shape[0], -1)) for g in observables], axis=1
        )

    if Y is None:
        if xp is None:
            xp = array_api_compat.array_namespace(X, U)

        X_lift = lift(X, xp=xp)
        Y_lift = X_lift[1:, :]
        X_lift = X_lift[:-1, :]
        X = X[:-1]
        U = U[:-1]
    else:
        if xp is None:
            xp = array_api_compat.array_namespace(X, U, Y)

        X_lift = lift(X, xp=xp)
        Y_lift = lift(Y, xp=xp)

    d_lift = X_lift.shape[1]

    # NOTE: see build_dense_extended_dmd for math, design choices, etc.
    if first_observable_is_state:
        dmdc_model = build_dense_dmdc(X_lift, U, Y_lift, method=method, eps=eps, xp=xp)
        A = dmdc_model.A
        B = dmdc_model.B
        C = xp.eye(d_lift, X.shape[1], dtype=X.dtype, device=X.device)
    else:
        # Fit A, B, and decoder C in a single factorization pass
        Y_combo = xp.concat([Y_lift, X], axis=1)
        dmdc_combo = build_dense_dmdc(X_lift, U, Y_combo, method=method, eps=eps, xp=xp)
        W_A = dmdc_combo.A  # (d_lift, d_lift + d_x)
        W_B = dmdc_combo.B  # (d_u, d_lift + d_x)

        A = W_A[:, :d_lift]
        C = W_A[:, d_lift:]
        B = W_B[:, :d_lift]

    return DenseExtendedDMDc(A=A, B=B, C=C, observables=tuple(observables))


# }}}


# {{{ diagnostics


def relative_forecast_error(
    dmd: DMDcBase[ScalarTypeT],
    X: Array2D[ScalarTypeT],
    U: Array2D[ScalarTypeT],
    Xpred: Array2D[ScalarTypeT] | None = None,
    *,
    maxit: int | None = None,
    xp: Any = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute the per-step relative error of a forecast with control against *X*.

    If *Xpred* is not provided, a forecast is built with :meth:`DMDcBase.predict`
    using one control input per time step, so *U* must have exactly *maxit* rows.
    """
    max_maxit = X.shape[0] if Xpred is None else min(Xpred.shape[0], X.shape[0])
    if maxit is None:
        maxit = max_maxit - 1

    if not 0 < maxit < max_maxit:
        raise ValueError(f"'maxit' must be in (0, {max_maxit}): {maxit}")

    if Xpred is None and U.shape[0] != maxit:
        raise ValueError(
            f"'U' must have one control input per time step (maxit = {maxit}): "
            f"{U.shape}"
        )

    if xp is None:
        xp = (
            array_api_compat.array_namespace(X, U)
            if Xpred is None
            else array_api_compat.array_namespace(X, U, Xpred)
        )

    if Xpred is None:
        Xpred = dmd.predict(X[0], U, maxit, full=True)

    error = xp.linalg.norm(Xpred[: maxit + 1] - X[: maxit + 1], axis=-1)
    xnorm = xp.linalg.norm(X[: maxit + 1], axis=-1)

    xnorm = xp.linalg.norm(xnorm, axis=0, keepdims=True)
    xnorm = xp.where(xnorm < 100 * xp.finfo(X.dtype).eps, 1.0, xnorm)

    return error / xnorm


def fit_residual(
    dmd: DMDcBase[ScalarTypeT],
    X1: Array2D[ScalarTypeT],
    U: Array2D[ScalarTypeT],
    X2: Array2D[ScalarTypeT],
    *,
    xp: Any = None,
) -> Array0D[np.floating[Any]]:
    r"""Compute the relative one-step residual of the fitted DMDc model."""
    if xp is None:
        xp = array_api_compat.array_namespace(X1, U, X2)

    X_fit = dmd.decode(dmd.evolve(dmd.encode(X1), U))

    x2norm = xp.linalg.norm(X2)
    if abs(x2norm) < 100 * xp.finfo(X2.dtype).eps:
        x2norm = 1.0

    return xp.linalg.norm(X_fit - X2) / x2norm


# }}}
