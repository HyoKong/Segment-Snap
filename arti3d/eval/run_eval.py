"""Thin wrapper over the vendored USDNet evaluator.

`evaluate()` mutates module-level globals (`opt`, `CLASS_LABELS`, `VALID_CLASS_IDS`) and returns
a bare tuple whose arity depends on the flags. This wrapper pins the dataset to "articulate3d",
names the outputs, and re-asserts the settings the metric depends on afterwards — notably
`min_region_sizes == 1`, which a prior `evaluate(dataset="stpls3d")` call in the same process
would have silently changed to 10 (evaluate_semantic_instance.py:754) and which would then drop
every handle-sized instance.
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
from typing import Optional

import numpy as np

from . import evaluate_semantic_instance as ESI

# The ranking column on the live leaderboard is AP50_axis_origin == MAO-ST.
MOV_KEYS = ["AP50", "MA_ap50", "MO_ap50", "MAO_ap50", "MAO_ST_ap50",
            "I_ap50", "I_vector_ap50", "I_out_ap50", "I_GT_ap50", "I_out_GT_ap50"]


def evaluate_preds(preds: dict, gt: dict, track: str = "mov",
                   eval_articulation: Optional[bool] = None,
                   eval_hierarchy_inter: bool = False,
                   verbose: bool = False) -> dict:
    """preds: {sid: {pred_masks (N,K), pred_classes, pred_scores[, pred_origins, pred_axises]}}
    gt: output of gt.load_gt (same scene keys)."""
    if eval_articulation is None:
        eval_articulation = (track == "mov")

    missing = set(preds) - set(gt)
    assert not missing, f"predictions for scenes with no GT: {sorted(missing)[:5]}"

    # every scene's GT lives in the same instance_gt/<split> dir
    gt_dirs = {os.path.dirname(g["gt_file"]) for g in gt.values()}
    assert len(gt_dirs) == 1, f"GT files span multiple dirs: {gt_dirs}"
    gt_path = gt_dirs.pop()

    for sid, p in preds.items():
        n_gt = gt[sid]["n_points"]
        assert p["pred_masks"].shape[0] == n_gt, (
            f"{sid}: pred_masks has {p['pred_masks'].shape[0]} rows, GT has {n_gt} points. "
            "Masks must be (N_points, N_inst); Pointcept emits the transpose."
        )

    gt_articulations = {
        sid: {"articulations_dict": gt[sid]["articulations_dict"],
              "interaction_labels": gt[sid]["interaction_labels"]}
        for sid in preds
    }

    buf = io.StringIO()
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
        with ctx:
            out = ESI.evaluate(
                preds, gt_path, tmp.name, dataset="articulate3d",
                eval_articulation=eval_articulation,
                gt_articulations=gt_articulations,
                eval_hierarchy_inter=eval_hierarchy_inter,
            )

    # The metric's own invariants, re-checked after the call mutated the globals.
    assert ESI.opt["min_region_sizes"][0] == 1, (
        f"min_region_sizes is {ESI.opt['min_region_sizes'][0]}, expected 1 — a prior "
        "evaluate(dataset='stpls3d') in this process would set it to 10 and drop small instances")
    assert tuple(ESI.CLASS_LABELS) == ("rotation", "translation"), ESI.CLASS_LABELS
    assert list(ESI.VALID_CLASS_IDS) == [1, 2], ESI.VALID_CLASS_IDS

    if isinstance(out, tuple):
        res = dict(zip(MOV_KEYS, out))
    else:
        res = {"AP50": out}
    res["_log"] = buf.getvalue()
    return res


def fmt(res: dict) -> str:
    cols = [k for k in MOV_KEYS if k in res]
    return "  ".join(f"{k}={res[k]:.4f}" for k in cols)
