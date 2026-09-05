"""Oriented bounding boxes and the motion snap.

This is where the motion score lives: choosing the axis and origin from the part's own oriented
bounding box, with no learned motion head at all, lifts the motion multiplier from the ~60% a
regression head reaches to roughly 82%.

The one non-obvious requirement: the in-plane axes must come from a **minimum-area rectangle**
(rotating calipers over the convex hull), NOT from PCA. PCA's in-plane orientation follows the
point *distribution*, not the shape, and reproduces the annotation OBB far less often —
measured 62.8% vs 75.8% axis coverage on GT masks.
"""
from __future__ import annotations

import numpy as np

__all__ = ["min_area_obb", "convex_hull_2d", "snap_axis", "parallel_edges",
           "perp_foot", "point_to_line", "OBB"]


def point_to_line(p: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    """Perpendicular distance from p to the line (origin, axis) — the metric's own criterion."""
    a = axis / np.linalg.norm(axis)
    d = np.asarray(p, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return float(np.linalg.norm(d - d.dot(a) * a))


def perp_foot(p: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Projection of p onto the line — the origin placement the annotators used (the origin measurement: the GT
    origin is the OBB centroid projected onto the hinge line, median along-axis offset 0.000 m)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(p, dtype=np.float64) - o
    return o + d.dot(a) * a


def convex_hull_2d(P: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. numpy>=2 removed the 2-D form of np.cross, so the turn test is
    written out explicitly."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h: list = []
        for p in pts:
            while len(h) >= 2:
                a, b = h[-2], h[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 1e-12:
                    break
                h.pop()
            h.append(p)
        return h

    return np.array(half(P)[:-1] + half(P[::-1])[:-1])


class OBB:
    """axes: (3,3) unit rows sorted by descending extent; ext: (3,) extents; centroid: (3,)."""

    __slots__ = ("axes", "ext", "centroid")

    def __init__(self, axes, ext, centroid):
        self.axes, self.ext, self.centroid = axes, ext, centroid

    @property
    def normal(self) -> np.ndarray:
        """Thinnest axis = the plane normal = `dominantNormal` in the released annotations."""
        return self.axes[2]

    def __repr__(self) -> str:
        return f"OBB(ext={np.round(self.ext, 3).tolist()})"


def min_area_obb(P: np.ndarray, max_pts: int = 20000, rng=None) -> OBB:
    """PCA plane normal + in-plane minimum-area rectangle."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) > max_pts:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(len(P), max_pts, replace=False)]
    c = P.mean(0)
    X = P - c
    if len(P) < 3:
        return OBB(np.eye(3), np.zeros(3), c)

    _, V = np.linalg.eigh(X.T @ X / len(P))
    n = V[:, 0]                       # smallest variance direction
    e1 = V[:, 2] - n * (n @ V[:, 2])
    ne = np.linalg.norm(e1)
    if ne < 1e-12:
        e1 = np.eye(3)[np.argmin(np.abs(n))]
        e1 = e1 - n * (n @ e1)
        ne = np.linalg.norm(e1)
    e1 /= ne
    e2 = np.cross(n, e1)

    P2 = np.stack([X @ e1, X @ e2], 1)
    H = convex_hull_2d(P2)
    best_dir, best_area = np.array([1.0, 0.0]), np.inf
    if len(H) >= 3:
        for i in range(len(H)):
            e = H[(i + 1) % len(H)] - H[i]
            ln = np.linalg.norm(e)
            if ln < 1e-12:
                continue
            u = e / ln
            v = np.array([-u[1], u[0]])
            a, b = P2 @ u, P2 @ v
            area = (a.max() - a.min()) * (b.max() - b.min())
            if area < best_area:
                best_area = area
                best_dir = u if (a.max() - a.min()) >= (b.max() - b.min()) else v

    a1 = best_dir[0] * e1 + best_dir[1] * e2
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(n, a1)
    A = np.stack([a1, a2, n])
    ext = np.array([np.ptp(X @ a1), np.ptp(X @ a2), np.ptp(X @ n)])
    order = np.argsort(ext)[::-1]
    return OBB(A[order], ext[order], c)


def snap_axis(pred_axis: np.ndarray, obb: OBB) -> np.ndarray:
    """Snap a predicted direction to the nearest OBB axis, sign-agnostically.

    the axis measurement: the GT axis is within 15 deg of one of the part's own 3 OBB axes for 97.4% of rotations
    and 99.5% of translations. REACT3D reports the same refinement cutting orientation error
    6.72 deg -> 1.16 deg at scene scale.
    """
    a = np.asarray(pred_axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    cos = np.abs(obb.axes @ a)
    return obb.axes[int(np.argmax(cos))]


def parallel_edges(obb: OBB, axis: np.ndarray) -> np.ndarray:
    """The 4 OBB edge lines parallel to `axis`, returned as points on those lines (4,3)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = int(np.argmax(np.abs(obb.axes @ a)))
    j, l = [q for q in range(3) if q != k]
    return np.array([obb.centroid + obb.axes[j] * (sj * obb.ext[j] / 2)
                     + obb.axes[l] * (sl * obb.ext[l] / 2)
                     for sj in (-1, 1) for sl in (-1, 1)])
