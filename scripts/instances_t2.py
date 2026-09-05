"""Track 2 instances: connected components over the semantic probability field, scored by mean
probability, with the S2 child head's detections appended strictly below.

    PYTHONPATH=. python scripts/instances_t2.py \
        --probs runs/armA/infer_validation --split validation \
        --union-child runs/s2/child_validation.pkl --out runs/t2_val

There are no queries and no superpoints on this track. Interactable handles are 16-21 points; a
query decoder loses badly at that size and superpoint pooling destroys about a fifth of the handle
targets outright. So the model is a point-level 3-class semantic net, and instances come from
connected components of its foreground argmax at `--radius` (2.5 cm: 2.0 shatters handles, 3.0
merges neighbours). Each instance's class is the majority vote of its points and its score is the
MEAN per-point probability — which is why the inference stage saves probabilities, not the argmax.

THE UNION. The joint model's child head emits a second, independent set of detections. They are
appended STRICTLY BELOW every incumbent instance rather than merged into the ranking: the child's
score is a product of two sigmoids while the incumbent's is a mean probability, so the two are not
on a common scale and only their relative position matters. Average precision integrates the whole
ranking, so tail detections that convert missed handles are free as long as the head is
undisturbed. The gain is flat for every scale in [0.05, 0.5] and collapses at 1.0 — a plateau, not
a fitted parameter — and the strictly-below property is ASSERTED on the assembled predictions
rather than assumed, because "appended below" is otherwise a statement of intent.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from arti3d.eval.gt import list_scenes, load_points
from arti3d.prep.cluster import scene_instances

LITE_ROOT = "data/pointcept_lite"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="dir of <sid>_prob.npy from infer_t2_sem.py")
    ap.add_argument("--split", default="validation", choices=("train", "validation"))
    ap.add_argument("--track", default="inter", choices=("inter", "mov"))
    ap.add_argument("--data-root", default=LITE_ROOT,
                    help="point-cloud root holding <split>/<sid>/coord.npy")
    ap.add_argument("--radius", type=float, default=0.025)
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--union-child", default=None,
                    help="child predictions pkl from infer_s2_child.py; omitted = no union")
    ap.add_argument("--union-scale", type=float, default=0.05,
                    help="child scores are multiplied by this before appending. Any value in "
                         "[0.05, 0.5] gives the same result; what matters is 'strictly below'.")
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

    scenes = list_scenes(a.track, a.split)
    have = {f[:-len("_prob.npy")] for f in os.listdir(a.probs) if f.endswith("_prob.npy")}
    missing = sorted(set(scenes) - have)
    assert not missing, f"{len(missing)} scenes have no probabilities: {missing[:5]}"

    preds, n_inst = {}, 0
    for sid in scenes:
        prob = np.load(os.path.join(a.probs, f"{sid}_prob.npy")).astype(np.float32)
        coord = np.load(os.path.join(a.data_root, a.split, sid, "coord.npy"))
        #: The submission is indexed against the challenge cloud's ROW ORDER. A silent permutation
        #: here produces a perfectly well-formed artifact that scores zero, so it is asserted
        #: rather than trusted.
        src = load_points(a.track, a.split, sid)
        assert len(src) == len(coord) == len(prob), (sid, len(src), len(coord), len(prob))
        assert np.array_equal(src[:, :3], coord), (
            f"{sid}: {a.data_root} coord.npy is not byte-identical to the challenge cloud")

        m, c, s = scene_instances(coord, prob, a.radius, a.min_points)
        preds[sid] = {"pred_masks": m, "pred_classes": c, "pred_scores": s,
                      "is_child": np.zeros(len(c), dtype=bool)}
        n_inst += len(c)

    print(f"scenes {len(preds)} · instances {n_inst} "
          f"(mean {n_inst / max(len(preds), 1):.1f}/scene) · r={a.radius} min_pts={a.min_points}")

    if a.union_child:
        with open(a.union_child, "rb") as fh:
            child = pickle.load(fh)
        child = child["preds"] if isinstance(child, dict) and "preds" in child else child
        n_add = 0
        for sid in list(preds):
            cp = child.get(sid)
            if cp is None or cp["pred_masks"].shape[1] == 0:
                continue
            ip = preds[sid]
            assert cp["pred_masks"].shape[0] == ip["pred_masks"].shape[0], (
                f"{sid}: child has {cp['pred_masks'].shape[0]} rows, incumbent "
                f"{ip['pred_masks'].shape[0]} — not the same point cloud")
            preds[sid] = {
                "pred_masks": np.concatenate([ip["pred_masks"], cp["pred_masks"]], axis=1),
                "pred_classes": np.concatenate([ip["pred_classes"], cp["pred_classes"]]),
                "pred_scores": np.concatenate([
                    ip["pred_scores"].astype(np.float64),
                    cp["pred_scores"].astype(np.float64) * a.union_scale]),
                #: which columns came from the child head — the class-vote only relabels those
                "is_child": np.concatenate([ip["is_child"],
                                            np.ones(cp["pred_masks"].shape[1], dtype=bool)])}
            n_add += cp["pred_masks"].shape[1]
        # the property the operation is named for, checked on the assembled predictions
        for sid, p in preds.items():
            k = p["pred_masks"].shape[1]
            cp = child.get(sid)
            if cp is None or k == 0:
                continue
            n_c = cp["pred_masks"].shape[1]
            if n_c == 0 or k == n_c:
                continue
            assert p["pred_scores"][:k - n_c].min() > p["pred_scores"][k - n_c:].max(), (
                f"{sid}: union scale {a.union_scale} does not put the child strictly below the "
                "incumbent, so the measured plateau does not apply")
        n_inst += n_add
        print(f"union: +{n_add} child instances at scale {a.union_scale} -> {n_inst} total")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"t2_{a.split}_preds.pkl"), "wb") as fh:
        pickle.dump(preds, fh, protocol=4)

    res = None
    if not a.no_eval:
        from arti3d.eval.gt import load_gt
        from arti3d.eval.run_eval import evaluate_preds
        gt = load_gt(a.track, a.split, sorted(preds))
        res = evaluate_preds(preds, gt, track=a.track)
        print(f"  AP50 = {res['AP50']!r}")
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump({"config": vars(a), "n_scenes": len(preds), "n_instances": n_inst,
                   "metrics": {k: v for k, v in (res or {}).items() if k != "_log"}}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
