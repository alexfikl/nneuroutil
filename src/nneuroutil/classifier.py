# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import array_api_compat
import numpy as np

from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array1D, Array2D

log = module_logger(__name__)

# {{{ classify_linear_discriminant_analysis


@dataclass(frozen=True)
class LDAEstimator:
    G: Array2D[np.floating[Any]]

    centroids: Array1D[np.floating[Any]]
    proj_centroids: Array2D[np.floating[Any]]
    labels: Array1D[np.floating[Any]]

    def predict(
        self,
        x: Array1D[np.floating[Any]] | Array2D[np.floating[Any]],
        *,
        xp: Any = None,
    ) -> Array1D[np.floating[Any]]:
        if x.ndim == 1:
            return self.predict(x[None, :], xp=xp)[0]

        if xp is None:
            xp = array_api_compat.array_namespace(x, self.G)

        c = self.centroids
        cz = self.proj_centroids

        Z = (x - c) @ self.G
        d = xp.sum(Z**2, axis=1)[:, None] - 2 * Z @ cz.T + xp.sum(cz**2, axis=1)[None]  # ty: ignore[unresolved-attribute]

        return self.labels[xp.argmin(d, axis=1)]


def classify_linear_discriminant_analysis(
    features: tuple[Array2D[np.floating[Any]], ...],
    labels: tuple[Any, ...],
    *,
    eps: float | None = None,
    xp: Any = None,
) -> LDAEstimator:
    """Classify the given dataset using standard LDA (mainly based on
    Algorithm 1 from [Howland2004]_).

    .. [Howland2004] P. Howland, H. Park,
        *Generalizing Discriminant Analysis Using the Generalized Singular Value
        Decomposition*,
        IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 26,
        pp. 995--1006, 2004,
        `doi:10.1109/tpami.2004.46 <https://doi.org/10.1109/tpami.2004.46>`__.
    """
    if len(features) != len(labels):
        raise ValueError(
            f"'features' and 'labels' do not match: {len(features)} and {len(labels)}"
        )

    if len(features) <= 1:
        raise ValueError(
            "cannot perform classification on 'features' with 0 or 1 classes"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(*features)

    _, nfeatures = features[0].shape
    dtype = features[0].dtype
    device = features[0].device

    if any(f.ndim != 2 or f.shape[-1] != nfeatures for f in features[1:]):
        raise ValueError(
            f"'features' do not have the same number of features: '{nfeatures}'"
        )

    if eps is None:
        eps = 10.0 * xp.finfo(dtype).eps

    if eps <= 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    # compute class sizes: Equation (1) in [Howland2004]_
    ns = xp.asarray([len(x) for x in features], dtype=dtype, device=device)
    n = xp.sum(ns)
    k = len(labels)

    # compute centroids: c^{(i)} and c in [Howland2004]_
    cs = xp.stack([xp.mean(x, axis=0) for x in features])
    c_m = ns @ cs / n

    # compute H_B and H_W from 'features' and form K = [H_B, H_W]^T
    K = xp.vstack([
        xp.sqrt(ns)[:, None] * (cs - c_m),
        *(x - c_i for x, c_i in zip(features, cs, strict=True)),
    ])

    # compute the SVD
    P, R, Qh = xp.linalg.svd(K, full_matrices=False)

    rank = xp.sum(R[0] * (k + n) * eps < R)
    P, S, Qh = P[:, :rank], R[:rank], Qh[:rank]

    _, _, Wt = xp.linalg.svd(P[:k], full_matrices=False)
    G = Qh.T @ (Wt.T[:, : k - 1] / S[:, None])

    return LDAEstimator(
        G=G,
        centroids=c_m,
        proj_centroids=(cs - c_m) @ G,
        labels=xp.asarray(labels, dtype=dtype, device=device),
    )


# }}}
