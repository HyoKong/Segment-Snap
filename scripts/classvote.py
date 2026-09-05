"""Class vote: let Track 1's predicted PARTS correct the class label of Track 2's child detections.

    PYTHONPATH=. python scripts/classvote.py \
        --t2 runs/t2_val/t2_validation_preds.pkl --t1 runs/t1_val/t1_validation_preds.pkl \
        --out runs/t2_val_voted

A handle's class is its parent part's motion type, so a handle sitting inside a predicted DRAWER is
a translation handle whatever the semantic net called it. Track 1 predicts the parts; this transfers
their class down to the handles they contain. That is the second half of the cross-track coupling:
Track 2 supplies Track 1's handles for the origin rule, and Track 1 supplies Track 2's class prior.

THE RULE, one rule with one threshold, applied to CHILD detections only:

  1. among Track-1 parts containing >= 90% of the child's points, take the highest-scoring one; if
     its score is >= --part-score and its class differs, adopt that class;
  2. otherwise, if the best-containing INCUMBENT instance holds >= 90% of the child and its class
     differs, adopt that class.

**Masks and scores are not touched** — only the class label changes. So mask matching is unaffected
and the only thing that moves is which class each prediction is matched against, which is exactly
the quantity the rule claims to fix. Incumbents are never relabelled: they are the ranking's head,
and the evidence for flipping them was never there.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

PART_MIN_SCORE = 0.1     # Track-1 parts below this never vote
CONTAIN = 0.9            # a part must hold this fraction of the child to speak for it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2", required=True, help="Track 2 predictions pkl (with the union applied)")
    ap.add_argument("--t1", required=True, help="Track 1 predictions pkl (the parts)")
    ap.add_argument("--part-score", type=float, default=0.3,
                    help="minimum Track-1 part score for its class to be adopted")
    ap.add_argument("--split", default="validation", choices=("train", "validation"))
    ap.add_argument("--track", default="inter", choices=("inter", "mov"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--gt-root", default=None,
                    help="root of the organisers' processed release (default: $ARTI3D_GT_ROOT, "
                         "else data/a3d/processed)")
    a = ap.parse_args()
    if a.gt_root:
        os.environ["ARTI3D_GT_ROOT"] = a.gt_root
        import arti3d.eval.gt as _gt
        _gt.DEFAULT_ROOT = a.gt_root

    with open(a.t2, "rb") as fh:
        preds = pickle.load(fh)
    with open(a.t1, "rb") as fh:
        parts = pickle.load(fh)

    n_flip = n_child = 0
    for sid, p in preds.items():
        m = p["pred_masks"]
        cls = np.asarray(p["pred_classes"]).astype(int).copy()
        is_child = np.asarray(p.get("is_child", np.zeros(len(cls), bool)), bool)
        size = m.sum(0)
        n_child += int(is_child.sum())
        if not is_child.any():
            continue

        q = parts[sid]
        psc = np.asarray(q["pred_scores"], np.float64)
        pcls = np.asarray(q["pred_classes"]).astype(int)
        keep = np.flatnonzero(psc >= PART_MIN_SCORE)
        P = q["pred_masks"][:, keep].astype(np.float32)
        assert P.shape[0] == m.shape[0], (sid, P.shape, m.shape)

        M = m.astype(np.float32)
        cont_part = (M.T @ P) / np.maximum(size[:, None], 1)          # child x part
        inc = np.flatnonzero(~is_child)
        cont_inc = ((M.T @ m[:, inc].astype(np.float32)) / np.maximum(size[:, None], 1)
                    if len(inc) else np.zeros((m.shape[1], 0), np.float32))

        for k in np.flatnonzero(is_child):
            if size[k] < 1:
                continue
            new = 0
            hit = np.flatnonzero(cont_part[k] >= CONTAIN)
            if len(hit):
                j = hit[int(np.argmax(psc[keep][hit]))]
                if psc[keep][j] >= a.part_score:
                    new = int(pcls[keep][j])
            if new in (0, cls[k]) and len(inc):
                j = int(np.argmax(cont_inc[k]))
                if cont_inc[k, j] >= CONTAIN:
                    new = int(cls[inc[j]])
            if new not in (0, cls[k]):
                cls[k] = new
                n_flip += 1
        p["pred_classes"] = cls

    print(f"class vote: {n_flip} of {n_child} child detections relabelled "
          f"(part score >= {a.part_score}, containment >= {CONTAIN})")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"t2_{a.split}_voted_preds.pkl"), "wb") as fh:
        pickle.dump(preds, fh, protocol=4)

    res = None
    if not a.no_eval:
        from arti3d.eval.gt import load_gt
        from arti3d.eval.run_eval import evaluate_preds
        gt = load_gt(a.track, a.split, sorted(preds))
        res = evaluate_preds(preds, gt, track=a.track)
        print(f"  AP50 = {res['AP50']!r}")
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump({"config": vars(a), "n_flipped": n_flip, "n_child": n_child,
                   "metrics": {k: v for k, v in (res or {}).items() if k != "_log"}}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
