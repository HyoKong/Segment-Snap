"""Connectivity rescoring: demote predictions whose masks are not one connected object.

    s' = s * max(f, 1e-6) ** gamma,   f = largest-connected-component fraction of the mask

The instance head places no connectivity constraint on a mask, so a prediction can be one solid
part plus a handful of stray points across the room. Those masks are usually spurious, and average
precision integrates over the whole ranking, so demoting them cleans the ranking without deleting
anything. Masks, axes and origins are untouched: this changes scores only.

gamma = 1 is the shipped setting. Measured on the validation split, over a base that already has
the largest-component cleanup applied before the box fit:

    gamma  0      1       2       3       4       6
    MAO-ST 0.3987 0.4149  0.4138  0.4121  0.4097  0.4071

so the optimum is interior and shallow — the gain is the existence of the rule, not the exponent.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _cc
from scipy.spatial import cKDTree

MAX_POINTS = 20000


def cc_fraction(P: np.ndarray, radius: float, rng=None, max_points: int = MAX_POINTS) -> float:
    """Fraction of a mask's points that lie in its largest connected component at `radius`.

    Masks above `max_points` are subsampled, because the pair query is quadratic in the worst
    case. `rng` defaults to a FRESH generator seeded 0, which makes the subsample a pure function
    of the input and therefore independent of the order instances are visited in. Passing a shared
    generator instead reproduces an older artifact whose subsample advanced with iteration order;
    see `--rescore-rng lineage` in scripts/infer_t1.py.
    """
    P = np.asarray(P, dtype=np.float64)
    if len(P) > max_points:
        rng = rng if rng is not None else np.random.default_rng(0)
        P = P[rng.choice(len(P), max_points, replace=False)]
    if len(P) < 2:
        return 1.0
    pairs = cKDTree(P).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return 1.0 / len(P)
    g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    _n, lab = _cc(g, directed=False)
    return float(np.bincount(lab).max()) / len(P)


def rescore(score: float, frac: float, gamma: float = 1.0) -> float:
    return float(score) * (max(frac, 1e-6) ** gamma)
