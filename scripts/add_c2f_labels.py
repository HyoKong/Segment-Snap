"""Attach expand_dict to the converted scenes as expand.npz (idx/dist/sem/inst).

    PYTHONPATH=src python scripts/add_c2f_labels.py --out data/pointcept_lite

Reads the release's own expand_dict/*.pkl (it ships with the data) and writes a
compact npz next to each scene so ArtiC2FDataset can dilate without unpickling per sample.
"""
import argparse, os, pickle, sys
import numpy as np
sys.path.insert(0, "src")
from arti3d.eval.gt import list_scenes, track_root

KEYS = {"idx": "expand_idx_records", "dist": "expand_distances",
        "sem": "expand_sem_records", "inst": "expand_inst_records"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pointcept_lite")
    ap.add_argument("--track", default="inter")
    a = ap.parse_args()
    ed = os.path.join(track_root(a.track), "expand_dict")
    n = miss = 0
    for split in ("train", "validation"):
        for sid in list_scenes(a.track, split):
            p = os.path.join(ed, f"{sid}.pkl")
            d = os.path.join(a.out, split, sid)
            if not os.path.exists(p):
                miss += 1
                continue
            with open(p, "rb") as f:
                e = pickle.load(f)
            arrs = {k: np.asarray(e[v]) for k, v in KEYS.items() if v in e}
            if "dist" not in arrs:      # some releases omit distances; treat all as radius-0 members
                arrs["dist"] = np.zeros(len(arrs["idx"]), np.float32)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "expand.npz"), **arrs)
            n += 1
    print(f"wrote expand.npz for {n} scenes ({miss} without an expand_dict entry)")
    if n:
        z = np.load(os.path.join(a.out, "train", list_scenes(a.track, "train")[0], "expand.npz"))
        print("  sample:", {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
        print(f"  dist range: {z['dist'].min():.4f} .. {z['dist'].max():.4f} m")
        for r in (0.10, 0.04, 0.0):
            print(f"  points added at r={r:.2f}: {(z['dist'] <= r).sum() if r > 0 else 0}")

if __name__ == "__main__":
    main()
