"""Evaluator self-tests: does our scoring reproduce the official metric?

    python -m arti3d.eval.selftest [--scenes N] [--track mov]

(i)   GT-as-prediction scores 1.000 on every AP column   <- hard stop
(ii)  perturbation tests pin the thresholds we believe the metric uses:
        axis  +14 deg passes / +16 deg fails       (15 deg gate, sign-agnostic)
        origin 0.24 m perp passes / 0.26 m fails   (0.25 m point-to-LINE gate)
        translation instances ignore origin entirely
        a non-unit axis (norm 0.476, the baseline's median) corrupts MAO-ST while
        MA still passes  -- the evaluator divides by ||a|| once, not twice
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .gt import load_gt, gt_passthrough_preds, ROTATION, TRANSLATION
from .run_eval import evaluate_preds, fmt

TOL = 1e-6
_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        _fail += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=0, help="0 = all val scenes")
    ap.add_argument("--track", default="mov")
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()

    gt = load_gt(a.track, a.split)
    if a.scenes:
        gt = {k: gt[k] for k in sorted(gt)[: a.scenes]}
    n_inst = sum(len(g["instance_ids"]) for g in gt.values())
    n_rot = sum(1 for g in gt.values() for i in g["instance_ids"] if i // 1000 == ROTATION)
    print(f"scenes={len(gt)}  instances={n_inst}  (rotation={n_rot} translation={n_inst - n_rot})\n")

    # ---- (i) passthrough -------------------------------------------------
    print("(i) GT passthrough — the passthrough check hard stop")
    base = evaluate_preds(gt_passthrough_preds(gt), gt, a.track)
    print(f"      {fmt(base)}")
    for k in ("AP50", "MA_ap50", "MO_ap50", "MAO_ap50", "MAO_ST_ap50"):
        check(f"{k} == 1.000", abs(base[k] - 1.0) < TOL, f"got {base[k]:.6f}")

    # ---- (ii) axis threshold --------------------------------------------
    print("\n(ii-a) axis gate at 15 deg")
    for deg, want_pass in ((14.0, True), (16.0, False)):
        r = evaluate_preds(gt_passthrough_preds(gt, jitter_axis=deg), gt, a.track)
        ok = (r["MA_ap50"] > 0.99) if want_pass else (r["MA_ap50"] < 0.01)
        check(f"axis +{deg:.0f} deg -> MA {'passes' if want_pass else 'fails'}",
              ok, f"MA={r['MA_ap50']:.4f} AP50={r['AP50']:.4f}")
        check(f"axis +{deg:.0f} deg leaves AP50 untouched", abs(r["AP50"] - 1.0) < TOL,
              f"AP50={r['AP50']:.6f}")

    # ---- origin threshold ------------------------------------------------
    print("\n(ii-b) origin gate at 0.25 m, point-to-line")
    for m, want_pass in ((0.24, True), (0.26, False)):
        r = evaluate_preds(gt_passthrough_preds(gt, jitter_origin=m), gt, a.track)
        # translation is origin-exempt, so MO can never drop below the translation share
        ok = (r["MO_ap50"] > 0.99) if want_pass else (r["MO_ap50"] < 0.99)
        check(f"origin {m} m perp -> MO {'passes' if want_pass else 'drops'}",
              ok, f"MO={r['MO_ap50']:.4f} MAO_ST={r['MAO_ST_ap50']:.4f}")

    # ---- translation origin exemption ------------------------------------
    print("\n(ii-c) translation instances ignore origin")
    r = evaluate_preds(gt_passthrough_preds(gt, jitter_origin=5.0), gt, a.track)
    check("origin 5 m perp: MO_ap50 ~ 0.5 (translation class unaffected, macro over 2 classes)",
          0.40 < r["MO_ap50"] < 0.60, f"MO={r['MO_ap50']:.4f}")
    check("origin 5 m perp: MA_ap50 still 1.000 (axis independent of origin)",
          abs(r["MA_ap50"] - 1.0) < TOL, f"MA={r['MA_ap50']:.4f}")

    # ---- non-unit axis ----------------------------------------------------
    print("\n(ii-d) a non-unit axis silently corrupts the origin residual")
    # A purely perpendicular offset CANNOT expose it: the residual is d - s*(d.ahat)*ahat,
    # so the corruption is proportional to the ALONG-axis component. Slide along the axis —
    # free at unit norm (point-to-LINE), fatal at norm 0.476.
    r_unit = evaluate_preds(gt_passthrough_preds(gt, slide_origin=2.0), gt, a.track)
    check("slide 2 m ALONG the axis at unit norm is free (point-to-line)",
          abs(r_unit["MAO_ST_ap50"] - 1.0) < TOL, f"MAO_ST={r_unit['MAO_ST_ap50']:.4f}")
    r = evaluate_preds(gt_passthrough_preds(gt, slide_origin=2.0, axis_scale=0.476), gt, a.track)
    check("norm 0.476: MA still 1.000 (the AXIS test divides by the product of norms)",
          abs(r["MA_ap50"] - 1.0) < TOL, f"MA={r['MA_ap50']:.4f}")
    check("norm 0.476 + same 2 m slide: MAO_ST craters (ORIGIN projection divides once)",
          r["MAO_ST_ap50"] < r_unit["MAO_ST_ap50"] - TOL,
          f"MAO_ST={r['MAO_ST_ap50']:.4f} vs {r_unit['MAO_ST_ap50']:.4f} at unit norm")

    # ---- unclipped arccos --------------------------------------------------
    print("\n(ii-e) unclipped arccos — an EXACTLY correct axis fails the gate")
    r0 = evaluate_preds(gt_passthrough_preds(gt, axis_epsilon=0.0), gt, a.track)
    check("axis bit-identical to GT: MA drops below 1.000 (arccos(1+1ULP) -> NaN)",
          r0["MA_ap50"] < 1.0 - TOL, f"MA={r0['MA_ap50']:.4f} vs {base['MA_ap50']:.4f} with the nudge")
    check("the nudge recovers it fully",
          abs(base["MA_ap50"] - 1.0) < TOL, f"MA={base['MA_ap50']:.4f}")
    check("AP50 is unaffected either way (segmentation is identical)",
          abs(r0["AP50"] - 1.0) < TOL, f"AP50={r0['AP50']:.4f}")

    print(f"\n{'ALL CHECKS PASSED' if _fail == 0 else str(_fail) + ' CHECK(S) FAILED'}")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
