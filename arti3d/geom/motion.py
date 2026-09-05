"""Augmentation safety for pipelines that carry motion targets.

A part's axis and hinge origin are geometric quantities attached to its points, so any augmentation
that is not a rigid motion invalidates them. Two are banned outright wherever motion targets exist:

  * an elastic distortion is not rigid, so no rotation carries an axis through it;
  * a flip is orientation-reversing — an axis maps as `R a` but a moment maps as `-det(R) R m`, and
    getting that sign wrong is a silent failure, so rather than implement the reflection algebra and
    depend on it, flips are refused.

The released models train with `motion_targets=False` and decode motion geometrically at inference
(`arti3d/geom/snap.py`), so nothing here is on the released training path. It is kept because the
dataset enforces the constraint at construction rather than trusting a config comment, and because
the constraint is a real property of anyone who turns motion targets on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .obb import OBB, min_area_obb, perp_foot

__all__ = ["canonical_obb", "axis_cls_encode", "axis_cls_decode",
           "edge_cls_encode", "edge_cls_decode", "instance_targets",
           "FORBIDDEN_TRANSFORMS", "assert_motion_safe_pipeline"]

ROTATION, TRANSLATION = 1, 2

#: ElasticDistortion is non-rigid, so no rotation carries the axis through it.
#: a flip is orientation-reversing: an axis maps as R a but a MOMENT maps as -det(R) R m,
#: and the sign is exactly what an earlier draft of PLAN.md got wrong. Rather than implement the
#: reflection algebra and rely on it, both are banned wherever motion targets exist.
FORBIDDEN_TRANSFORMS = ("RandomFlip", "ElasticDistortion")


def assert_motion_safe_pipeline(transforms, where: str = "pipeline") -> None:
    """Refuse to run a motion-target codepath alongside a flip or an elastic distortion.

    The training configs carry this constraint as a comment, but a comment
    does not survive a copy-paste into the the Track-1 instance model config — and the failure mode is silent: training
    completes, val looks plausible, and every submitted axis is wrong in a way no validator catches
    (V3 only checks that the axis is unit-norm, and a reflected axis is). This turns it into a
    startup crash.
    """
    names = []
    for t in transforms or ():
        n = t.get("type") if isinstance(t, dict) else getattr(t, "type", type(t).__name__)
        if n:
            names.append(str(n))
    bad = [n for n in names if n in FORBIDDEN_TRANSFORMS]
    assert not bad, (
        f"{where}: {bad} present while motion targets are in use. ElasticDistortion is "
        "non-rigid, so no rotation carries an axis through it. a flip is orientation-reversing "
        f"and the moment sign flips with det(R). Remove {bad} or drop the motion targets. "
        f"(pipeline was: {names})")
