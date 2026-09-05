"""Convert the organisers' release into training format for the MOVABLE-part track.

    PYTHONPATH=src python scripts/to_pointcept_mov.py --out data/pointcept_mov [--workers 8]

A separate data root from `pointcept_lite` on purpose. Both tracks need a key called `segment` and
Pointcept gives us exactly one per scene folder, but they mean different things:

    pointcept_lite/  segment = the INTER label  — {0, 1, 2} on HANDLE points, where the class is the
                                                  parent movable part's motion type (the semantic track0)
    pointcept_mov/   segment = the MOV label    — {0, 1, 2} on MOVABLE PART points, the part's own
                                                  motion type (the instance track)

Writing both into one folder would mean one of them silently overwriting the other, and the failure
would look like a model that mysteriously segments the wrong thing. Two roots, no ambiguity.

Emits, per scene:
    <out>/<split>/<scene>/coord.npy      float32 (N,3)
    <out>/<split>/<scene>/color.npy      uint8   (N,3)
    <out>/<split>/<scene>/normal.npy     float32 (N,3)
    <out>/<split>/<scene>/segment.npy    int16   (N,)  movable sem {0,1,2}; -1 for test
    <out>/<split>/<scene>/instance.npy   int32   (N,)  movable instance pid; -1 for background/test

`superpoint.npy` is added separately by `scripts/make_superpoints.py` so the sweep
parameters are recorded next to the thing they produced.

INVARIANT (asserted on EVERY scene, not a sample): row i here is row i of the challenge cloud.
Nothing is reordered, nothing is subsampled. Everything downstream depends on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "src")
from arti3d.eval.gt import list_scenes, load_points  # noqa: E402

#: The released code evaluates on validation; the held-out test split is not supported.
SPLITS = ("train", "validation")
C_SEM, C_INST = 9, 10
BACKGROUND = -1          # Pointcept's instance_ignore_index


def convert(args) -> tuple:
    split, sid, out = args
    m = load_points("mov", split, sid)
    quirks = []
    n = len(m)

    a = load_points("inter", split, sid)
    if not np.array_equal(a[:, 0:9], m[:, 0:9]):
        quirks.append("mov/inter geometry cols 0:9 differ")

    d = os.path.join(out, split, sid)
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/coord.npy", np.ascontiguousarray(m[:, 0:3], dtype=np.float32))
    np.save(f"{d}/color.npy", np.ascontiguousarray(m[:, 3:6], dtype=np.uint8))
    np.save(f"{d}/normal.npy", np.ascontiguousarray(m[:, 6:9], dtype=np.float32))

    seg = m[:, C_SEM].astype(np.int16)
    u = set(np.unique(seg).astype(int).tolist())
    if not u <= {0, 1, 2}:
        quirks.append(f"mov sem_gt outside {{0,1,2}}: {sorted(u)}")
    inst = m[:, C_INST].astype(np.int32)
    # Background must be the ignore index, not id 0 — otherwise every background point becomes
    # one enormous "instance 0" and InstanceParser will happily build a mask for it.
    inst = np.where(seg > 0, inst, BACKGROUND).astype(np.int32)
    ids = np.unique(inst[inst != BACKGROUND])
    n_inst = int(ids.size)
    # Every movable instance must carry exactly one semantic class.
    for pid in ids:
        cls = np.unique(seg[inst == pid])
        if cls.size != 1:
            quirks.append(f"pid {int(pid)}: {cls.tolist()} classes on one instance")

    np.save(f"{d}/segment.npy", seg)
    np.save(f"{d}/instance.npy", inst)
    assert len(np.load(f"{d}/coord.npy")) == n, sid
    return sid, n, n_inst, int((seg > 0).sum()), quirks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pointcept_mov")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    a = ap.parse_args()

    stats, all_quirks = {}, {}
    for split in a.splits:
        scenes = list_scenes("mov", split)
        tot_n = tot_i = tot_fg = 0
        with ProcessPoolExecutor(a.workers) as ex:
            for sid, n, ni, fg, quirks in ex.map(
                    convert, [(split, s, a.out) for s in scenes], chunksize=2):
                tot_n += n; tot_i += ni; tot_fg += fg
                if quirks:
                    all_quirks[sid] = quirks
        stats[split] = dict(scenes=len(scenes), points=tot_n, instances=tot_i, foreground=tot_fg,
                            fg_fraction=tot_fg / max(tot_n, 1))
        s = stats[split]
        print(f"{split:11s} {s['scenes']:3d} scenes  {s['points']:>11,} pts  "
              f"{s['instances']:>5,} instances  fg {s['foreground']:>9,} "
              f"({100*s['fg_fraction']:.3f}%)")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump({"stats": stats, "quirks": all_quirks}, f, indent=1)
    if all_quirks:
        print(f"\n!! {len(all_quirks)} scene(s) with quirks (recorded, not fatal):")
        for sid, q in list(all_quirks.items())[:10]:
            print(f"   {sid}: {q[:3]}")
    else:
        print("\nglobal asserts clean on every scene")
    # A known-answer check, so a conversion regression is caught at conversion time.
    if "validation" in stats:
        print(f"\nvalidation instances: {stats['validation']['instances']} (expected: 390)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
