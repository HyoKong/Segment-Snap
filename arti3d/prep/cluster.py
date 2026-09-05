"""Connected-component instance extraction from a point-level semantic probability field.

Arm A predicts a 3-class point label {0 background, 1 rotation-handle, 2 translation-handle} and
recovers instances geometrically rather than with a proposal head. The primitive is a radius graph
at r = 2.5 cm on the already-2 cm cloud, per class — measured as a 96.6% macro ceiling on
the 42-scene val set, which is what makes the no-superpoint Track 2 path viable at all.

`min_points = 3`, NOT 10: 17.9% of interactable instances have fewer than 10 points, and the measured
lesson is that every default threshold in this codebase is a quiet way to delete tiny handles.
"""
from __future__ import annotations

import numpy as np

__all__ = ["connected_components", "scene_instances", "FG_CLASSES"]

FG_CLASSES = (1, 2)          # 1 rotation-handle, 2 translation-handle; 0 is background


def connected_components(P: np.ndarray, radius: float) -> np.ndarray:
    """Component id per row of P, linking any two points within `radius`.

    Uses a sparse radius graph rather than `cKDTree.query_pairs`: an early-training model can label
    a large fraction of the cloud foreground, and the dense pair list for 100k+ points is a memory
    cliff exactly when the model is at its worst.
    """
    from scipy.sparse.csgraph import connected_components as _cc
    from sklearn.neighbors import radius_neighbors_graph
    if len(P) <= 1:
        return np.zeros(len(P), dtype=np.int64)
    g = radius_neighbors_graph(P, radius, mode="connectivity", include_self=True)
    _, lab = _cc(g, directed=False)
    return lab.astype(np.int64)


def scene_instances(coord: np.ndarray, prob: np.ndarray, radius: float = 0.025,
                    min_points: int = 3, fg_thresh: float = 0.0):
    """Per-point class probabilities -> scored instances.

    Returns (masks (N, K) bool, classes (K,) int64, scores (K,) float64). Score is the mean
    probability of the instance's own class over its own points; calibration is a separate,
    dev-fitted step deliberately not folded in here.

    `fg_thresh` demands a minimum foreground probability ON TOP of winning the argmax. It exists
    because Arm A over-predicts foreground by roughly 3x (measured: 701 predicted foreground points
    on a scene whose val-average GT foreground is ~231), and fat masks are precisely what caps
    per-instance IoU below the 0.5 matching threshold. Plain argmax only asks whether a handle class
    beats background, which on a 0.05%-foreground target is a very low bar — the CE weight of 0.1 on
    background deliberately makes it low. 0.0 reproduces plain argmax exactly.
    """
    assert prob.ndim == 2 and prob.shape[0] == len(coord), (coord.shape, prob.shape)
    labels = prob.argmax(1)
    if fg_thresh > 0:
        # demote to background any point whose winning foreground probability is unconvincing
        win = prob[np.arange(len(prob)), labels]
        labels = np.where((labels > 0) & (win < fg_thresh), 0, labels)
    masks, classes, scores = [], [], []
    for c in FG_CLASSES:
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        comp = connected_components(coord[idx], radius)
        for k in range(int(comp.max()) + 1):
            member = idx[comp == k]
            if member.size < min_points:
                continue
            m = np.zeros(len(coord), dtype=bool)
            m[member] = True
            masks.append(m)
            classes.append(c)
            scores.append(float(prob[member, c].mean()))
    if not masks:
        return (np.zeros((len(coord), 0), dtype=bool),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
    return (np.stack(masks, axis=1),
            np.asarray(classes, dtype=np.int64),
            np.asarray(scores, dtype=np.float64))
