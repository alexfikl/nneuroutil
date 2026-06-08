# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import array_api_compat
import numpy as np

from nneuroutil.helpers import module_logger, register_dataclass
from nneuroutil.typing import Array1D, Array2D, ArrayND

log = module_logger(__name__)


# {{{ DMD


@register_dataclass
@dataclass(frozen=True)
class DMD:
    Ahat: Array2D
    """Reduced-order model operator of shape :math:`(r, r)` with rank :attr:`rank`."""

    U: Array2D
    """Temporal modes as an array of shape :math:`(n - 1, r)`."""
    S: Array1D
    """Singular values as an array of shape :math:`(r,)`."""
    Vh: Array2D
    """Spatial modes as an array of shape :math:`(r, d)`."""

    @property
    def dtype(self) -> np.dtype:
        """The :class:`~numpy.dtype` of this operator."""
        return self.Ahat.dtype

    @property
    def ndim(self) -> int:
        """Number of array dimensions (here 2)."""
        return 2

    @property
    def shape(self) -> tuple[int, int]:
        """Tuple of array dimensions."""
        return self.Ahat.shape

    @property
    def reduced_size(self) -> int:
        """The rank (size) of the reduced order model."""
        return self.Ahat.shape[0]

    @property
    def full_size(self) -> int:
        """The size of the full model."""
        return self.Vh.shape[1]

    def eigendecomposition(self) -> tuple[Array1D, Array2D]:
        """Compute the eigendecomposition of :attr:`Ahat`."""
        xp = array_api_compat.array_namespace(self.Ahat)

        eigs, eigenvectors = xp.linalg.eig(self.Ahat)
        return eigs, eigenvectors

    def encode(self, x: ArrayND) -> ArrayND:
        """Project the full state *x* to the reduced coordinates."""
        assert x.shape[0] == self.full_size

        xp = array_api_compat.array_namespace(x, self.Vh)
        return xp.einsum("ij,j...->i...", self.Vh, x)

    def evolve(self, x: ArrayND) -> ArrayND:
        """Evolve the reduced-order model."""
        assert x.shape[0] == self.reduced_size

        xp = array_api_compat.array_namespace(x, self.Ahat)
        return xp.einsum("ij,j...->i...", self.Ahat, x)

    def decode(self, x: ArrayND) -> ArrayND:
        """Reconstruct the full state from the reduced state *x*."""
        assert x.shape[0] == self.reduced_size

        xp = array_api_compat.array_namespace(x, self.Vh)
        return xp.einsum("ij,i...->j...", xp.conj(self.Vh), x)

    def __matmul__(self, x: ArrayND) -> ArrayND:
        """Evolve the reduced-order model."""
        return self.evolve(x)

    def __call__(self, x: Array1D) -> Array1D:
        """Evolve the reduced-order model."""
        return self.evolve(x)


def reconstruct(dmd: DMD, x0: Array1D, steps: int) -> Array2D:
    """
    :arg x0: initial condition for the system. If this is the size of the
        reduced system, we assume that it represents the amplitudes of the DMD
        modes of the initial condition. Otherwise, the initial condition is
        projected onto the DMD modes.
    :arg steps: number of steps to compute.
    """
    xp = array_api_compat.array_namespace(x0, dmd.Ahat)

    # determine the DMD modes
    lambdas, vs = dmd.eigendecomposition()
    Phi = dmd.decode(vs)

    # project the initial condition on the modes by least squares
    if x0.shape == (dmd.reduced_size,):
        b = x0
    else:
        b, _, _, _ = xp.linalg.lstsq(Phi, x0)

    # compute all iterations of the operator
    n = xp.arange(steps, dtype=x0.dtype)
    Lambda = lambdas[None, :] ** n

    # apply the operator
    return (Lambda * b[None, :]) @ Phi.conj().T


# }}}


# {{{ total-least-squares


def total_least_squares(
    X: Array2D,
    Y: Array2D,
    *,
    rank: int | None = None,
    xp: Any = None,
) -> tuple[Array2D, Array2D]:
    nsnapshots, dim = X.shape
    if X.shape != Y.shape:
        raise ValueError(
            f"inputs 'X' and outputs 'Y' have different shapes: {X.shape} and {Y.shape}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(X, Y)

    if rank is not None and not 0 < rank < dim:
        raise ValueError(f"'rank' must be in (0, {dim}): {rank}")

    Z = xp.concat([X, Y], axis=1)
    assert Z.shape == (nsnapshots, 2 * dim)

    U, S, Vh = xp.linalg.svd(Z, full_matrices=False)
    S = xp.astype(S, X.dtype)

    if rank is not None:
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    X = U @ xp.diag(S) @ Vh[:, :dim]
    Y = U @ xp.diag(S) @ Vh[:, dim:]

    return X, Y


# }}}


# {{{ classic DMD


def build_dmd(
    X: Array2D,
    Y: Array2D,
    *,
    rank: int | None = None,
    xp: Any = None,
) -> DMD:
    """Construct a DMD approximation of the system with snapshots *X* and outputs *Y*.

    :arg X: system snapshots of shape ``(nsnapshots, ndim)``.
    :arg rank: if given, the desired fixed rank of the approximation.
    """
    _, dim = X.shape

    if rank is not None and not 0 < rank < dim:
        raise ValueError(f"'rank' must be in (0, {dim}): {rank}")

    if xp is None:
        xp = array_api_compat.array_namespace(X)

    U, S, Vh = xp.linalg.svd(X, full_matrices=False)
    S = xp.astype(S, X.dtype)

    if rank is not None:
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    # construct reduced order model
    Ahat = Vh @ xp.conj(Y).T @ U @ xp.diag(1 / S)
    assert Ahat.ndim == 2
    assert Ahat.shape[0] == Ahat.shape[1]

    return DMD(Ahat, U=U, S=S, Vh=Vh)


# }}}
