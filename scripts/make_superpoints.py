"""Generate kNN Felzenszwalb-Huttenlocher superpoints for every scene.

    PYTHONPATH=src python scripts/make_superpoints.py \
        --roots data/pointcept_mov data/pointcept_lite --workers 8

Parameters chosen by sweep at **kThresh 0.005, segMinVerts 5**:
movable macro ceiling **98.4%** (rotation 100.0%, translation 96.8%), 14,154 superpoints/scene,
median superpoint size 12.0. The nearby setting (0.01, 5) was rejected on the PER-CLASS bar, not
the macro — its translation ceiling is 91.6%, under the 94% floor, and translation is half the
score. That is the whole reason the gate condition was stated per class.

Writes `<root>/<split>/<scene>/superpoint.npy`, int32 (N,), contiguous ids from 0. The same array
goes into every root given, because superpoints are a property of the geometry and both roots hold
the same cloud in the same row order — recomputing per root would burn CPU to produce identical
files, and worse, could produce *non*-identical ones if the parameters ever drifted between calls.

Determinism: `felzenszwalb_knn` is a deterministic function of (coord, normal, color) and the two
parameters. The same cloud gives the same labelling on every run, which matters because the
superpoints are baked into checkpoints via the mask targets — a regenerated, subtly different
superpoint set would silently invalidate a trained model (the same hazard as a permuted row order, one level up).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "src")
from arti3d.eval.gt import list_scenes  # noqa: E402
from arti3d.prep.superpoint import felzenszwalb_knn  # noqa: E402

# The the superpoint sweep decision. Changing these invalidates every checkpoint trained on the old ones.
K_THRESH, SEG_MIN = 0.005, 5
#: The released code evaluates on validation; the held-out test split is not supported.
SPLITS = ("train", "validation")


def one(args) -> tuple:
    split, sid, roots, k_thresh, seg_min = args
    src = os.path.join(roots[0], split, sid)
    coord = np.load(f"{src}/coord.npy").astype(np.float64)
    normal = np.load(f"{src}/normal.npy").astype(np.float64)
    color = np.load(f"{src}/color.npy").astype(np.float64)

    sp = felzenszwalb_knn(coord, normal, color, k_thresh=k_thresh, seg_min=seg_min)
    sp = np.ascontiguousarray(sp, dtype=np.int32)
    assert len(sp) == len(coord), (sid, len(sp), len(coord))
    assert sp.min() == 0 and sp.max() == len(np.unique(sp)) - 1, (
        f"{sid}: superpoint ids are not contiguous from 0")

    for r in roots:
        d = os.path.join(r, split, sid)
        if not os.path.isdir(d):
            continue
        # Row order must match this root's own cloud, or the labelling is attached to the wrong
        # points. Cheap to check, catastrophic to miss.
        c = np.load(f"{d}/coord.npy")
        assert len(c) == len(coord) and np.array_equal(c, np.load(f"{src}/coord.npy")), (
            f"{sid}: {r} coord.npy differs from {roots[0]} — refusing to write superpoints")
        np.save(f"{d}/superpoint.npy", sp)

    sizes = np.bincount(sp)
    return sid, len(coord), int(sp.max()) + 1, float(np.median(sizes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/pointcept_mov"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--k-thresh", type=float, default=K_THRESH)
    ap.add_argument("--seg-min", type=int, default=SEG_MIN)
    ap.add_argument("--out", default="superpoints_report.md")
    a = ap.parse_args()

    print(f"the superpoint sweep params: kThresh={a.k_thresh}, segMinVerts={a.seg_min}")
    print(f"roots: {a.roots}")
    rows, t0 = {}, time.time()
    for split in a.splits:
        scenes = list_scenes("mov", split)
        jobs = [(split, s, a.roots, a.k_thresh, a.seg_min) for s in scenes]
        nsp, med, npts = [], [], []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (sid, n, k, m) in enumerate(ex.map(one, jobs, chunksize=1)):
                nsp.append(k); med.append(m); npts.append(n)
                if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                    print(f"  {split} {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
        rows[split] = dict(scenes=len(scenes), points=int(np.sum(npts)),
                           sp_per_scene=float(np.mean(nsp)), median_size=float(np.mean(med)),
                           total_sp=int(np.sum(nsp)))
        r = rows[split]
        print(f"{split:11s} {r['scenes']:3d} scenes  {r['sp_per_scene']:.0f} sp/scene  "
              f"median size {r['median_size']:.1f}")

    with open(a.out, "w") as fh:
        fh.write("# Superpoint generation\n\n")
        fh.write(f"`scripts/make_superpoints.py`, kThresh={a.k_thresh}, segMinVerts={a.seg_min} "
                 f"(chosen by sweep), roots {a.roots}.\n\n")
        fh.write("| split | scenes | points | superpoints/scene | median size | total superpoints |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for s, r in rows.items():
            fh.write(f"| {s} | {r['scenes']} | {r['points']:,} | {r['sp_per_scene']:.0f} | "
                     f"{r['median_size']:.1f} | {r['total_sp']:,} |\n")
        fh.write(f"\nWall clock {time.time()-t0:.0f}s at {a.workers} workers. The sweep measured "
                 f"14,154 sp/scene and median size 12.0 on the 42 val scenes; the validation row "
                 f"above is the consistency check on that.\n")
    with open(os.path.join(a.roots[0], "superpoint_params.json"), "w") as fh:
        json.dump({"k_thresh": a.k_thresh, "seg_min": a.seg_min, "splits": rows}, fh, indent=1)
    print(f"\n-> {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
