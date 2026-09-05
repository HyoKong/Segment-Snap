"""Felzenszwalb–Huttenlocher superpoints on a bare point cloud (no mesh faces needed).

This is the option-(b) primitive in the superpoint sweep: the challenge cloud ships with `segments` all zeros and
`processed.zip` contains no mesh, so mesh-connectivity superpoints (option (a)) need a ScanNet++
download that is not on the critical path. A kNN graph over the 2 cm cloud needs nothing extra.

Same algorithm family as ScanNet's `segmentator` and `pointseg`: sort edges by weight, union with
the F&H predicate, then absorb components below `seg_min`.

**`seg_min` is the parameter that decides whether this is usable at all.** `pointseg` ships
`segMinVerts=20` and merges any smaller component into a neighbour across ANY edge, ignoring the
edge weight — with interactable instances at a median of 20 points, that deletes roughly half of
them before a network ever sees them (measured). Sweep it; do not accept the default.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class _DSU:
    __slots__ = ("p", "sz", "int_")

    def __init__(self, n: int):
        self.p = np.arange(n)
        self.sz = np.ones(n, np.int64)
        self.int_ = np.zeros(n, np.float64)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return int(x)

    def union(self, a: int, b: int, w: float) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]
        self.int_[a] = max(w, self.int_[a], self.int_[b])


def felzenszwalb_knn(coord: np.ndarray, normal: np.ndarray, color: np.ndarray | None = None,
                     k_thresh: float = 0.01, seg_min: int = 5, knn: int = 12,
                     color_weight: float = 0.2) -> np.ndarray:
    """Returns a (N,) int32 superpoint id per point."""
    n = len(coord)
    if n < 2:
        return np.zeros(n, np.int32)
    knn = min(knn, n - 1)
    _, nb = cKDTree(coord).query(coord, k=knn + 1)
    rows = np.repeat(np.arange(n), nb.shape[1])
    cols = nb.reshape(-1)
    m = rows < cols                       # each undirected edge once
    rows, cols = rows[m], cols[m]

    w = 1.0 - np.abs(np.einsum("ij,ij->i", normal[rows], normal[cols]))
    if color is not None and color_weight:
        dc = np.linalg.norm(color[rows].astype(np.float32) - color[cols].astype(np.float32), axis=1)
        w = w + color_weight * (dc / 441.673)   # /sqrt(3*255^2)

    o = np.argsort(w, kind="stable")
    rows, cols, w = rows[o], cols[o], w[o]

    dsu = _DSU(n)
    for a, b, wt in zip(rows, cols, w):
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        if wt <= min(dsu.int_[ra] + k_thresh / dsu.sz[ra],
                     dsu.int_[rb] + k_thresh / dsu.sz[rb]):
            dsu.union(ra, rb, float(wt))

    if seg_min > 1:
        for a, b in zip(rows, cols):
            ra, rb = dsu.find(a), dsu.find(b)
            if ra != rb and (dsu.sz[ra] < seg_min or dsu.sz[rb] < seg_min):
                dsu.union(ra, rb, 0.0)

    lab = np.array([dsu.find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(np.int32)


def ceiling_at_iou(superpoint: np.ndarray, gt_ids: np.ndarray, valid_classes=(1, 2),
                   iou_th: float = 0.5) -> dict:
    """Best mask achievable from whole superpoints: union of superpoints whose majority owner is
    that instance. This is the hard ceiling any superpoint-based decoder is bounded by."""
    enc = [int(v) for v in np.unique(gt_ids) if int(v) // 1000 in valid_classes]
    if not enc:
        return {}
    owner = {}
    for sp in np.unique(superpoint):
        idx = np.flatnonzero(superpoint == sp)
        g = gt_ids[idx]
        g = g[np.isin(g // 1000, valid_classes)]
        if len(g) > len(idx) / 2:                       # majority of the superpoint is foreground
            owner[int(sp)] = int(np.bincount(g).argmax())
    out = {}
    for e in enc:
        mine = [s for s, o in owner.items() if o == e]
        pred = np.flatnonzero(np.isin(superpoint, mine)) if mine else np.array([], np.int64)
        gm = set(np.flatnonzero(gt_ids == e).tolist())
        pm = set(pred.tolist())
        iou = len(gm & pm) / max(len(gm | pm), 1)
        out[e] = iou > iou_th
    return out
