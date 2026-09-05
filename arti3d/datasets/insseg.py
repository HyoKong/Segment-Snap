"""Dataset for movable-part instance segmentation over superpoints.

`DefaultDataset.VALID_ASSETS` does not include `superpoint` (defaults.py:37-45), so a scene folder
containing `superpoint.npy` would have it **silently skipped** and SPFormer would then fail on a
missing key — or worse, a future refactor would quietly fall back to point resolution. This adds it,
which is the whole reason the class exists.

It also enforces the motion-target constraint at construction rather than trusting a config comment:
The instance model itself has no motion targets, but the joint model inherits this class, and an
unsafe augmentation is a silent failure — the targets are quietly wrong rather than absent, so the
flip trains fine, evaluates plausibly, and reflects every submitted axis past a validator that only
checks unit norm.
"""
from __future__ import annotations

import os

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset

from ..geom.motion import assert_motion_safe_pipeline


@DATASETS.register_module()
class ArtiInsSegDataset(DefaultDataset):
    """DefaultDataset + `superpoint`, + the the augmentation-safety pipeline guard."""

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "superpoint",
        "segment",
        "instance",
    ]

    def __init__(self, motion_targets: bool = False, require_assets=(), **kwargs):
        # Guard BEFORE super().__init__ composes the transforms, so the crash names the offending
        # transform rather than dying somewhere inside Compose.
        if motion_targets:
            assert_motion_safe_pipeline(kwargs.get("transform"),
                                        f"{type(self).__name__}(split={kwargs.get('split')})")
        self.motion_targets = motion_targets
        super().__init__(**kwargs)
        self._require_assets(require_assets)

    def _require_assets(self, required) -> None:
        """Fail at construction if any scene is missing an asset the run depends on.

        `DefaultDataset.get_data` loads whatever `.npy` files happen to be in the folder and
        silently skips the rest, then substitutes an all -1 array for a missing `segment` or
        `instance` (defaults.py:126-138). So pointing this config at the WRONG DATA ROOT — the
        interactable one, whose `segment` means something else entirely and which has no
        `instance.npy` or `superpoint.npy` at all — produces a run that trains happily against
        all-background instance targets and reports a plausible-looking loss. There is no later
        stage that catches it. Hence: check every scene, at construction, before a GPU is touched.
        """
        if not required:
            return
        missing = {}
        for path in self.data_list:
            have = {f[:-4] for f in os.listdir(path) if f.endswith(".npy")}
            gap = [a for a in required if a not in have]
            if gap:
                missing[os.path.basename(path)] = gap
        # Name the assets that are ACTUALLY absent, not the full requirement list — an error that
        # reads "missing [everything]" when only superpoint is missing sends the reader to the wrong
        # place, and this one fires at launch time when nobody is inclined to read carefully.
        absent = sorted({a for gap in missing.values() for a in gap})
        assert not missing, (
            f"{type(self).__name__}(split={self.split}, data_root={self.data_root}): "
            f"{len(missing)} of {len(self.data_list)} scenes are missing {absent} "
            f"(required: {sorted(required)}) — e.g. {dict(list(missing.items())[:3])}. "
            "If `superpoint` is the gap, scripts/make_superpoints.py has not finished. "
            "If `instance` or `segment` is, check data_root: the interactable root has neither "
            "instance nor superpoint, and its `segment` is the handle label, not the "
            "movable-part label.")
