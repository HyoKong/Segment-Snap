"""Minimal conversion for the interactable-handle track (semantic segmentation; no superpoints, no motion targets).

    PYTHONPATH=src python scripts/to_pointcept_lite.py --out data/pointcept_lite [--workers 24]

the semantic model predicts 3 classes over points: {0 background, 1 rotation-handle, 2 translation-handle},
instances via kNN connected components at inference, parent motion type inherited from the movable
part. VERIFIED : the `inter` track's own `sem_gt` column (col 9) is ALREADY exactly that
label — it equals the parent movable's motion type (11/11 instances checked on 0a76e06478) — so no
cross-referencing against the mov track is needed. The mov and inter npys are also byte-identical in
columns 0:9, so geometry can come from either.

Emits, per scene:
    <out>/<split>/<scene>/coord.npy    float32 (N,3)
    <out>/<split>/<scene>/color.npy    uint8   (N,3)
    <out>/<split>/<scene>/normal.npy   float32 (N,3)
    <out>/<split>/<scene>/segment.npy  int16   (N,)   {0,1,2}; not applicable: the held-out test split is not supported
    <out>/<split>/<scene>/inter_gt.npy int32   (N,)   parent movable pid, for CC grouping at eval

INVARIANT (asserted): row i here is row i of the challenge cloud. Nothing is reordered, nothing is
subsampled. Everything downstream depends on this.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "src")
from arti3d.eval.gt import list_scenes, load_points, C_INTER  # noqa: E402

#: The released code evaluates on validation; the held-out test split is not supported.
SPLITS = ("train", "validation")


def convert(args) -> tuple:
    """Returns (sid, n_points, n_fg, n_rot, quirks) — quirks are RECORDED, never fatal."""
    split, sid, out = args
    a = load_points("inter", split, sid)          # (N,13) float32
    quirks = []

    # Assert on EVERY scene, not a sample.
    m = load_points("mov", split, sid)
    if not np.array_equal(a[:, 0:9], m[:, 0:9]):
        quirks.append("mov/inter geometry cols 0:9 differ")
    u = np.unique(a[:, 9]).astype(int).tolist()
    if not set(u) <= {0, 1, 2}:
        quirks.append(f"inter sem_gt outside {{0,1,2}}: {u}")
    # inter sem_gt must equal the PARENT movable's motion type, per instance
    inter_pid = a[:, C_INTER].astype(np.int64)
    sem_i = a[:, 9].astype(np.int64)
    inst_m = m[:, 10].astype(np.int64)
    sem_m = m[:, 9].astype(np.int64)
    for pid in np.unique(inter_pid[sem_i > 0]):
        pm = inst_m == pid
        if not pm.any():
            quirks.append(f"pid {pid}: no movable points")
            continue
        child = np.bincount(sem_i[(inter_pid == pid) & (sem_i > 0)]).argmax()
        parent = np.bincount(sem_m[pm]).argmax()
        if child != parent:
            quirks.append(f"pid {pid}: child class {child} != parent {parent}")
    d = os.path.join(out, split, sid)
    os.makedirs(d, exist_ok=True)
    n = len(a)

    np.save(f"{d}/coord.npy", np.ascontiguousarray(a[:, 0:3], dtype=np.float32))
    np.save(f"{d}/color.npy", np.ascontiguousarray(a[:, 3:6], dtype=np.uint8))
    np.save(f"{d}/normal.npy", np.ascontiguousarray(a[:, 6:9], dtype=np.float32))

    seg = a[:, 9].astype(np.int16)                # {0,1,2}, already the semantic label
    inter = a[:, C_INTER].astype(np.int32)
    np.save(f"{d}/segment.npy", seg)
    np.save(f"{d}/inter_gt.npy", inter)

    # invariant: unchanged row order and count
    assert len(np.load(f"{d}/coord.npy")) == n
    fg = int((seg > 0).sum())
    return sid, n, fg, int((seg == 1).sum()), quirks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pointcept_lite")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    a = ap.parse_args()

    stats = {}
    all_quirks: dict = {}
    for split in a.splits:
        scenes = list_scenes("inter", split)
        jobs = [(split, s, a.out) for s in scenes]
        tot_n = tot_fg = tot_rot = 0
        with ProcessPoolExecutor(a.workers) as ex:
            for sid, n, fg, rot, quirks in ex.map(convert, jobs, chunksize=2):
                tot_n += n; tot_fg += fg; tot_rot += rot
                if quirks:
                    all_quirks[sid] = quirks
        stats[split] = dict(scenes=len(scenes), points=tot_n, foreground=tot_fg,
                            rotation=tot_rot, translation=tot_fg - tot_rot,
                            fg_fraction=tot_fg / max(tot_n, 1))
        s = stats[split]
        print(f"{split:11s} {s['scenes']:3d} scenes  {s['points']:>11,} pts  "
              f"fg {s['foreground']:>7,} ({100*s['fg_fraction']:.4f}%)  "
              f"rot {s['rotation']:,} / tra {s['translation']:,}")

    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump({"stats": stats, "quirks": all_quirks}, f, indent=1)
    if all_quirks:
        print(f"\n!! {len(all_quirks)} scene(s) with data quirks (recorded, not fatal):")
        for sid, q in list(all_quirks.items())[:10]:
            print(f"   {sid}: {q[:3]}")
    else:
        print("\nglobal asserts clean on every scene: mov/inter cols 0:9 identical, "
              "inter sem_gt in {0,1,2} and == parent movable motion type")
    print(f"\nwrote {a.out}")
    print("NOTE: foreground is ~0.05-0.08% of points. The loss MUST be class-balanced, and the "
          "coarse-to-fine GT dilation (expand_dict/, which ships inside processed.zip) is the "
          "difference between AP50 18.3 and 0.0 on tiny parts in SceneFun3D's own ablation.")


if __name__ == "__main__":
    main()
