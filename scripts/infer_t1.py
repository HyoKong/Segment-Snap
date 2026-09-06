"""Track 1 inference: movable parts, then motion by geometric decoding. No learned motion head.

    PYTHONPATH=.:third_party/volt python scripts/infer_t1.py \
        --run <run dir> --ckpt epoch_14 --split validation \
        --handles <arm-A prob dir> --out runs/t1_val

Five stages, in order:

  1. SEGMENTATION. SPFormer queries over superpoints on a Volt-B backbone emit part masks with a
     class in {rotation, translation}. Selection is per-query argmax rather than a global top-k
     over the (query x class) grid — see arti3d.models.spformer_argmax.
  2. CLEANUP BEFORE THE FIT. The geometry is fitted to the mask's largest connected component at
     `--snap-largest-cc` metres, not to the whole mask. The SUBMITTED MASK IS NOT MODIFIED: a few
     stray points across the room inflate the bounding box and relocate every quantity derived
     from it, and this is what makes the derived axis and origin stable. Worth +0.037 on the
     ranking column; about a quarter of instances are affected.
  3. AXIS, and the classes differ: a translation slides along the part's own fitted plane normal
     (per-part), a rotation hinges about a FIXED VERTICAL DIRECTION — a dataset prior that hinges
     are vertical, the same vector for every instance, passing ~93% of matched rotations.
  4. ORIGIN (rotations only; the metric ignores translation origins). The hinge is the box edge,
     of the four parallel to the axis, FARTHEST from the part's handle, and the origin is the
     perpendicular foot of the box centroid on that edge. **The handle position comes from the
     Track 2 model** (`--handles`), which is the cross-track coupling: it is worth about x2.9 on
     the ranking column against the no-handle centroid rule. With no handle within
     `--handle-max-dist`, the centroid is used.
  5. CONNECTIVITY RESCORING. s' = s * f^gamma with f the mask's largest-component fraction; see
     arti3d.geom.rescore.

Inference runs in fp32. The instance head thresholds three times, and half precision flips
borderline instances — measured, it emptied a third of the scenes once.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _cc
from scipy.spatial import cKDTree

from arti3d.geom.rescore import cc_fraction, rescore
from arti3d.geom.snap import ROTATION, snap_instance
from arti3d.io import load_weights, md5, resolve_ckpt
from arti3d.prep.cluster import scene_instances


def largest_cc(P: np.ndarray, radius: float, min_points: int) -> np.ndarray | None:
    """Boolean selector for the largest connected component of `P` at `radius`, or None."""
    if len(P) < 2:
        return None
    pairs = cKDTree(P).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return None
    g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    _n, lab = _cc(g, directed=False)
    sel = lab == int(np.argmax(np.bincount(lab)))
    return sel if max(min_points, 2) <= sel.sum() < len(P) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory (holds config.py, model/)")
    ap.add_argument("--ckpt", default="epoch_14", help="best | last | N | epoch_N | path")
    ap.add_argument("--split", default="validation", choices=("train", "validation"))
    ap.add_argument("--data-root", default=None,
                    help="dataset root, overriding the training config's own value. The configs "
                         "carry a path relative to the repository root because they are also the "
                         "training record; this lets the release run from anywhere.")
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", required=True, help="output directory for preds pkl + metrics")

    ap.add_argument("--handles", default=None,
                    help="Track-2 probability directory (<sid>_prob.npy). Without it, rotation "
                         "origins fall back to the box centroid and the ranking column drops ~x2.9")
    ap.add_argument("--handle-radius", type=float, default=0.025)
    ap.add_argument("--handle-min-points", type=int, default=3)
    ap.add_argument("--handle-max-dist", type=float, default=0.5,
                    help="a handle farther than this from the part is not that part's handle")

    ap.add_argument("--snap-largest-cc", type=float, default=0.05, metavar="RADIUS",
                    help="cleanup radius for the pre-fit largest component; 0 disables it")
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--axis-rule", default="canonical",
                    choices=("canonical", "most_vertical", "longest"))
    ap.add_argument("--topk-rule", default="argmax", choices=("argmax", "cap"),
                    help="'argmax' is the shipped rule; 'cap' is upstream's global top-k")

    ap.add_argument("--gamma", type=float, default=1.0, help="connectivity rescoring exponent")
    ap.add_argument("--cc-radius", type=float, default=0.05)
    ap.add_argument("--rescore-rng", default="per-instance", choices=("per-instance", "lineage"),
                    help="'per-instance' (default) gives every mask a fresh generator, so the "
                         "subsample of a >20k-point mask does not depend on iteration order. "
                         "'lineage' shares one generator across the run and exists ONLY to "
                         "reproduce the  reference artifact bit-for-bit.")
    ap.add_argument("--no-eval", action="store_true", help="write predictions without scoring")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt-root", default=None,
                    help="root of the organisers' processed release (default: $ARTI3D_GT_ROOT, "
                         "else data/a3d/processed)")
    a = ap.parse_args()
    if a.gt_root:
        os.environ["ARTI3D_GT_ROOT"] = a.gt_root
        import arti3d.eval.gt as _gt
        _gt.DEFAULT_ROOT = a.gt_root

    #: PRECISION POLICY, pinned rather than inherited. Inference is fp32 and TF32 is OFF on both
    #: matmul and cuDNN paths: the instance head thresholds three times, and a reduced-precision
    #: accumulation flips instances that sit near a threshold. Leaving these at the framework
    #: default makes the result depend on the torch version and the card.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    from pointcept.datasets import build_dataset, point_collate_fn
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    import arti3d.datasets.insseg      # noqa: F401  (registers the dataset type)
    import arti3d.datasets.transforms  # noqa: F401  (registers deterministic voxelisation)
    from arti3d.models.spformer_argmax import patch_spformer_argmax

    if a.topk_rule == "argmax":
        patch_spformer_argmax()
        print("selection: per-query argmax")

    cfg = Config.fromfile(os.path.join(a.run, "config.py"))
    data_root = a.data_root or cfg.data_root
    dcfg = dict(cfg.data.val)
    dcfg["split"] = a.split
    dcfg["data_root"] = data_root
    ds = build_dataset(dcfg)
    print(f"data root {data_root}")

    ckpt_path = resolve_ckpt(a.run, a.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg.model).to(a.device).eval()
    load_weights(model, ckpt, a.weights)
    print(f"checkpoint {ckpt_path}  epoch {ckpt.get('epoch')}  weights={a.weights}  "
          f"md5 {md5(ckpt_path)}")

    #: `nudge` accepts a generator but ignores it (the rotation off exact parallelism is
    #: deterministic); kept for signature compatibility.
    rng = np.random.default_rng(0)
    #: The shared generator for `--rescore-rng lineage`. Unused in the default mode.
    lineage_rng = np.random.default_rng(0)

    n = min(a.limit, len(ds)) if a.limit else len(ds)
    preds, npoints, n_inst, n_cleaned, n_handled, t0 = {}, {}, 0, 0, 0, time.time()

    for i in range(n):
        sample = ds[i]
        batch = point_collate_fn([sample])
        batch = {k: (v.to(a.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.inference_mode():
            out = model(batch)                       # fp32: no autocast, deliberately
        sid = sample["name"]

        #: SPFormer.prediction already returns numpy sorted by score descending.
        masks = np.asarray(out["pred_masks"]).astype(bool)      # (K, N)
        scores = np.asarray(out["pred_scores"]).astype(np.float64)
        classes = np.asarray(out["pred_classes"]).astype(np.int64)
        coord = np.load(os.path.join(data_root, a.split, sid, "coord.npy")).astype(np.float64)

        # --- handle centroids from the Track 2 probability field ------------------------------
        hcent = []
        if a.handles:
            hp = np.load(os.path.join(a.handles, f"{sid}_prob.npy")).astype(np.float32)
            assert len(hp) == len(coord), (sid, len(hp), len(coord))
            hm, _c, _s = scene_instances(coord, hp, radius=a.handle_radius,
                                         min_points=a.handle_min_points)
            hcent = [coord[hm[:, j]].mean(0) for j in range(hm.shape[1])]

        keep, ax, og = [], [], []
        for k in range(masks.shape[0]):
            idx = np.flatnonzero(masks[k])
            if idx.size < a.min_points:
                continue
            P = coord[idx]

            # stage 2 — clean the point set the box is fitted to; the mask itself is untouched
            if a.snap_largest_cc > 0:
                sel = largest_cc(P, a.snap_largest_cc, a.min_points)
                if sel is not None:
                    P = P[sel]
                    n_cleaned += 1

            # stage 4 — the part's own handle, if one is close enough
            handle, rule = None, "centroid"
            if hcent:
                d = [float(np.min(np.linalg.norm(P - c, axis=1))) for c in hcent]
                j = int(np.argmin(d))
                if d[j] < a.handle_max_dist:
                    handle, rule, n_handled = hcent[j], "handle_far", n_handled + 1

            axis, origin = snap_instance(P, int(classes[k]), handle, rng,
                                         axis_rule=a.axis_rule, origin_rule=rule)
            keep.append(k)
            ax.append(axis)
            og.append(origin)

        preds[sid] = {
            "pred_masks": masks[keep].T.astype(bool) if keep
                          else np.zeros((len(coord), 0), dtype=bool),
            "pred_classes": classes[keep] if keep else np.zeros(0, np.int64),
            "pred_scores": scores[keep] if keep else np.zeros(0, np.float64),
            "pred_axises": np.stack(ax) if ax else np.zeros((0, 3)),
            "pred_origins": np.stack(og) if og else np.zeros((0, 3)),
        }
        npoints[sid] = len(coord)
        n_inst += len(keep)
        print(f"  [{i+1}/{n}] {sid}: {masks.shape[0]} raw -> {len(keep)} kept")

    print(f"\nscenes {len(preds)} · instances {n_inst} · {n_cleaned} boxes cleaned · "
          f"{n_handled} origins from a handle · {time.time()-t0:.0f}s")

    # --- stage 5, as a SEPARATE PASS IN SORTED SCENE ORDER -----------------------------------
    # Deliberately not folded into the loop above. The dataset does not yield scenes in sorted
    # order, and under `--rescore-rng lineage` the shared generator's draw sequence depends on the
    # order large masks are visited in. A separate sorted pass mirrors the original two-stage
    # pipeline (infer, write, then rescore the written artifact) and makes the lineage well
    # defined. Under the default per-instance mode the order is irrelevant and this costs nothing.
    n_sub = 0
    for sid in sorted(preds):
        d = preds[sid]
        if not d["pred_scores"].size:
            continue
        coord = np.load(os.path.join(data_root, a.split, sid, "coord.npy")).astype(np.float64)
        M, sc = d["pred_masks"], d["pred_scores"]
        out_sc = np.empty(len(sc), np.float64)
        for k in range(M.shape[1]):
            P = coord[M[:, k]]
            if len(P) > 20000:
                n_sub += 1
            f = cc_fraction(P, a.cc_radius,
                            rng=lineage_rng if a.rescore_rng == "lineage" else None)
            out_sc[k] = rescore(sc[k], f, a.gamma)
        d["pred_scores"] = out_sc
    print(f"  connectivity rescoring: gamma={a.gamma} r={a.cc_radius} "
          f"rng={a.rescore_rng} · {n_sub} masks above the 20k subsample bound")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"t1_{a.split}_preds.pkl"), "wb") as fh:
        pickle.dump(preds, fh, protocol=4)

    res = None
    if not a.no_eval:
        from arti3d.eval.gt import load_gt
        from arti3d.eval.run_eval import evaluate_preds, fmt
        gt = load_gt("mov", a.split, sorted(preds))
        res = evaluate_preds(preds, gt, track="mov")
        print("  " + fmt(res))
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump({"config": vars(a), "checkpoint_md5": md5(ckpt_path),
                   "n_scenes": len(preds), "n_instances": n_inst,
                   "metrics": {k: v for k, v in (res or {}).items() if k != "_log"}}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
