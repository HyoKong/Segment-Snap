"""Ground-truth loading for the vendored USDNet evaluator.

The vendored evaluator has two sharp edges that this module exists to file down:

1. `util_3d.get_instances` looks up articulation GT by the **ENCODED** instance id
   (`sem_id * 1000 + pid + 1`), not by the raw movable pid that keys the HDF5 file.
   On a miss it does NOT raise — it silently substitutes `axis=[0,0,1], origin=[0,0,0]`
   (util_3d.py:188-189). A key-convention slip would therefore not crash, it would just
   quietly score against fabricated ground truth.  `load_gt` asserts full coverage.

2. `assign_instances_for_scan` iterates `for m in articulations_dict: ... m.item()`
   (evaluate_semantic_instance.py:601-603), so the dict keys must be numpy scalars.
   Plain Python ints raise AttributeError.

Layout consumed (from the organisers' processed.zip):
    <root>/articulate3d_challenge_{mov,inter}/
        {train,validation,test}/<sid>.npy                  (N,13) float32
        {train,validation}/<sid>_articulation.h5           keyed by str(pid)
        instance_gt/{train,validation}/<sid>.txt           sem*1000 + pid + 1
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import h5py
import numpy as np

#: Root of the organisers' processed release. Relative by default, so it resolves for anyone
#: running from the repository root, and overridable by environment for everyone else — every
#: entry point also exposes it as `--gt-root`.
DEFAULT_ROOT = os.environ.get("ARTI3D_GT_ROOT", "data/a3d/processed")
NPY_COLS = 13  # coord3 color3 normal3 sem inst segments inter

# npy column indices
C_COORD = slice(0, 3)
C_COLOR = slice(3, 6)
C_NORMAL = slice(6, 9)
C_SEM = 9
C_INST = 10
C_SEGMENTS = 11
C_INTER = 12

ROTATION, TRANSLATION = 1, 2


def track_root(track: str, root: str = DEFAULT_ROOT) -> str:
    assert track in ("mov", "inter"), track
    return os.path.join(root, f"articulate3d_challenge_{track}")


def list_scenes(track: str, split: str, root: str = DEFAULT_ROOT) -> list[str]:
    d = os.path.join(track_root(track, root), split)
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy"))


def load_points(track: str, split: str, sid: str, root: str = DEFAULT_ROOT) -> np.ndarray:
    a = np.load(os.path.join(track_root(track, root), split, f"{sid}.npy"))
    assert a.ndim == 2 and a.shape[1] == NPY_COLS, (sid, a.shape)
    return a


def encode_gt_id(sem_id: int, pid: int) -> int:
    """The evaluator's instance encoding (util_3d.Instance.raw_inst_id, inverted)."""
    return int(sem_id) * 1000 + int(pid) + 1


def load_scene_gt(track: str, split: str, sid: str, root: str = DEFAULT_ROOT) -> dict:
    """Returns gt_file path, the articulation dict in evaluator key convention, and inter labels."""
    tr = track_root(track, root)
    gt_file = os.path.join(tr, "instance_gt", split, f"{sid}.txt")
    gt_ids = np.loadtxt(gt_file, dtype=np.int64)

    pts = load_points(track, split, sid, root)
    assert len(gt_ids) == len(pts), (sid, len(gt_ids), len(pts))

    arts: Dict[np.int64, dict] = {}
    h5p = os.path.join(tr, split, f"{sid}_articulation.h5")
    if os.path.exists(h5p):
        with h5py.File(h5p, "r") as f:
            for k in f.keys():
                g = f[k]
                sem = int(np.asarray(g["sem_id"]).item())
                # numpy scalar key: the evaluator calls .item() on it
                arts[np.int64(encode_gt_id(sem, int(k)))] = {
                    "axis": np.asarray(g["axis"], dtype=np.float64),
                    "origin": np.asarray(g["origin"], dtype=np.float64),
                }

    # Coverage assert — see module docstring, edge #1.
    present = {int(v) for v in np.unique(gt_ids) if int(v) // 1000 in (ROTATION, TRANSLATION)}
    missing = present - {int(k) for k in arts}
    assert not missing, (
        f"{sid}: {len(missing)} GT instances have no articulation entry {sorted(missing)[:5]}; "
        "the evaluator would silently score them against axis=[0,0,1], origin=[0,0,0]"
    )

    return {
        "sid": sid,
        "gt_file": gt_file,
        "gt_ids": gt_ids,
        "n_points": len(gt_ids),
        "articulations_dict": arts,
        "interaction_labels": pts[:, C_INTER].astype(np.int64),
        "instance_ids": sorted(present),
    }


def load_gt(track: str, split: str, scenes: Optional[Iterable[str]] = None,
            root: str = DEFAULT_ROOT) -> dict:
    scenes = list(scenes) if scenes is not None else list_scenes(track, split, root)
    return {sid: load_scene_gt(track, split, sid, root) for sid in scenes}


#: Safety nudge applied to every emitted axis. See `gt_passthrough_preds`.
#: 1e-7 rad is the smallest value that removed all NaNs on val; 1e-6 leaves margin.
#: Induced angular error 5.7e-5 deg vs a 15 deg gate — 2.6e5x of headroom.
AXIS_EPSILON = 1e-6


def gt_passthrough_preds(gt: dict, jitter_axis: float = 0.0, jitter_origin: float = 0.0,
                         axis_scale: float = 1.0, seed: int = 0,
                         axis_epsilon: float = AXIS_EPSILON,
                         slide_origin: float = 0.0) -> dict:
    """GT re-emitted as predictions — the the passthrough check hard stop.

    NOTE (measured, 390 val instances): a passthrough with axis_epsilon=0 does NOT
    score 1.000. `match_criteria_MA_pred` computes
        deriavation = dot(a,b) / (norm(a) * norm(b));  arccos(abs(deriavation)) < 15deg
    with **no clipping**. For a bit-identical axis, sqrt(s)*sqrt(s) can differ from s by one
    ULP, so |cos| comes out as 1 + 2.2e-16 for **24.1% (94/390)** of val instances; arccos
    returns NaN, `NaN < x` is False, and an exactly-correct axis FAILS the gate.
    This is not a theoretical hazard: it costs ~24% of MA if the axes are left exact.

    `axis_epsilon` rotates every emitted axis by that many radians off exact parallelism, which
    forces |cos| < 1 strictly. Measured: eps=1e-7 -> 0 NaNs, max induced error 5.8e-6 deg.
    Set axis_epsilon=0 to reproduce the failure.

    The other perturbations exist for the evaluator self-tests (`arti3d.eval.selftest`):
      jitter_axis   degrees to rotate every axis by (about an axis-orthogonal direction)
      jitter_origin metres to shift every origin by, PERPENDICULAR to its own axis
                    (a parallel shift is free under the point-to-line metric)
      axis_scale    multiply the emitted axis by this (the evaluator's ORIGIN projection
                    divides by ||a|| once instead of twice, so a non-unit axis corrupts
                    MAO-ST while MA — which normalises correctly — still passes)
      slide_origin  metres to shift every origin ALONG its own axis. At unit norm this is FREE
                    (the criterion is point-to-LINE), which is why we place origins at the
                    perpendicular foot. It is also the only way to expose the axis-norm bug:
                    the residual is
                    d - s*(d.ahat)*ahat for ||a||=s, so a non-unit axis leaves (1-s)*(d.ahat)
                    behind — i.e. the corruption is proportional to the ALONG-axis component,
                    and a purely perpendicular offset cannot reveal it.
    """
    rng = np.random.default_rng(seed)
    preds = {}
    for sid, g in gt.items():
        ids = g["instance_ids"]
        n = g["n_points"]
        masks = np.zeros((n, len(ids)), dtype=np.float32)
        classes = np.zeros(len(ids), dtype=np.int64)
        scores = np.ones(len(ids), dtype=np.float32)
        origins = np.zeros((len(ids), 3), dtype=np.float64)
        axises = np.zeros((len(ids), 3), dtype=np.float64)
        for j, enc in enumerate(ids):
            masks[:, j] = (g["gt_ids"] == enc).astype(np.float32)
            classes[j] = enc // 1000
            a = g["articulations_dict"][np.int64(enc)]["axis"].copy()
            o = g["articulations_dict"][np.int64(enc)]["origin"].copy()
            a = a / np.linalg.norm(a)
            if jitter_axis:
                # rotate a by `jitter_axis` degrees about a direction orthogonal to it
                t = rng.normal(size=3)
                t -= t.dot(a) * a
                t /= np.linalg.norm(t)
                th = np.deg2rad(jitter_axis)
                a = a * np.cos(th) + t * np.sin(th)
            if jitter_origin:
                p = rng.normal(size=3)
                p -= p.dot(a) * a          # perpendicular component only
                p /= np.linalg.norm(p)
                o = o + p * jitter_origin
            if slide_origin:
                o = o + a * slide_origin      # free at unit norm; the axis-norm probe
            if axis_epsilon:
                # Deterministic nudge off exact parallelism (no RNG) so the evaluator's
                # unclipped arccos stays finite and artifacts are byte-reproducible.
                e = np.zeros(3); e[int(np.argmin(np.abs(a)))] = 1.0
                t = e - a * float(a @ e); t /= np.linalg.norm(t)
                a = a * np.cos(axis_epsilon) + t * np.sin(axis_epsilon)
                a /= np.linalg.norm(a)
            axises[j] = a * axis_scale
            origins[j] = o
        preds[sid] = {
            "pred_masks": masks,
            "pred_classes": classes,
            "pred_scores": scores,
            "pred_origins": origins,
            "pred_axises": axises,
        }
    return preds
