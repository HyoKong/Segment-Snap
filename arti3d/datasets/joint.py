"""S2 joint dataset — the Track-1 instance model's movable sample, plus per-point CHILD (interactable) targets.

S2 needs, in ONE sample: movable instances (parent branch, superpoint SPFormer) and interactable
instances (child branch, point level). No existing root carries both — `pointcept_mov` has movable
`instance.npy` + `superpoint.npy`, `pointcept_lite` has the interactable labels — and the design notes
therefore specified a converter that merges them into a third root.

**NO CONVERTER: THE TWO ROOTS ARE READ DIRECTLY.**
established that the two roots are byte-identical
row-for-row — 42/42 validation and 195/195 train scenes, coords equal at `atol=0 rtol=0`, and both
equal to the CHALLENGE cloud, which is the arbiter the submission is indexed against. Given that,
materialising a third root would copy 3.5 GB to gain nothing and would introduce the one failure a
derived root always eventually has: silently drifting from its sources after one of them is
regenerated. This class reads the child labels from the parallel root at load time instead. It also
removes the converter from the critical path, which the design notes cheap to build.

WHAT THE CHILD LABELS ACTUALLY ARE, because the two roots differ in what they carry about this.
`pointcept_lite` has **no `instance.npy`**. The per-point interactable instance id is
`inter_gt.npy`, and it is a RAW id, while the challenge evaluator encodes `sem * 1000 + id + 1`.
Verified byte-exact on all 42 validation scenes, 0 mismatching:

    challenge gt_ids == where(inter_gt > 0, segment * 1000 + inter_gt + 1, 1)

so the child targets are fully derivable from lite's own files and this class needs no dependency on
the challenge GT directory at training time. Raw ids are emitted; encoding is a scoring concern.

THE ROW-ALIGNMENT GUARD IS NOT DECORATION. If the roots ever diverge — one regenerated, a different
dedup, a changed voxel pass — every point would receive its movable label from one location and its
handle label from another. Nothing downstream would notice: both arrays are length N, both are int,
`require_assets` finds every key, and the loss falls smoothly against systematically scrambled child
targets. That is  with no observable symptom, so the length check runs on EVERY sample (it is
free) and a full coordinate comparison runs on the first `verify_coords` samples per worker, which
catches a regenerated root on the first batch rather than after a night of training.

THIS APPLIES DOUBLE HERE. `DefaultDataset.get_data` substitutes all-`-1` arrays for a missing
`segment`/`instance` rather than failing, which is how a run trains happily against empty targets.
The child keys are not in `VALID_ASSETS` and would not even be loaded, so `_require_child` checks
every scene at construction, before a GPU is touched.
"""
from __future__ import annotations

import os

import numpy as np

from pointcept.datasets.builder import DATASETS

from .insseg import ArtiInsSegDataset

CHILD_ASSETS = ("inter_gt", "segment")
#: Rows sampled per scene for the construction-time coordinate fingerprint. 512 strided rows out of
#: ~300 k detects any regeneration or reordering in practice while costing ~no I/O next to the
#: header-only shape check.
FINGERPRINT_ROWS = 512


@DATASETS.register_module()
class ArtiJointDataset(ArtiInsSegDataset):
    """`ArtiInsSegDataset` + `child_instance` / `child_segment` from a parallel root."""

    def __init__(self, child_root: str, verify_coords: int = 8, **kwargs):
        self.child_root = child_root
        self.verify_coords = int(verify_coords)
        self._verified = 0
        super().__init__(**kwargs)
        self._require_child()

    # ------------------------------------------------------------------ construction-time checks
    def _child_dir(self, path: str) -> str:
        """`<data_root>/<split>/<sid>` -> `<child_root>/<split>/<sid>`.

        Derived from the resolved scene path, never rebuilt from `self.split`: under a json split
        file (tv_train229.json) the scenes come from several directories at once and `self.split` is
        a filename, not a path component. That is exactly the defect fixed in c2f.py, and it is not
        being reintroduced here.
        """
        sid = os.path.basename(path)
        split_dir = os.path.basename(os.path.dirname(path))
        return os.path.join(self.child_root, split_dir, sid)

    def _require_child(self) -> None:
        missing = {}
        for path in self.data_list:
            d = self._child_dir(path)
            gap = [a for a in CHILD_ASSETS if not os.path.exists(os.path.join(d, f"{a}.npy"))]
            if gap:
                missing[os.path.basename(path)] = gap
        assert not missing, (
            f"{type(self).__name__}: {len(missing)} of {len(self.data_list)} scenes lack child "
            f"assets under child_root={self.child_root!r} — e.g. {dict(list(missing.items())[:3])}. "
            "Expected `inter_gt.npy` and `segment.npy` per scene (the interactable root). Note "
            "`instance.npy` does NOT exist there; if you are looking for it, see this module's "
            "docstring — the per-point interactable id is inter_gt.npy.")
        self._require_alignment()

    def _require_alignment(self) -> None:
        """Enforce row alignment at CONSTRUCTION, per scene, before a GPU is touched.

        The roots are row-aligned today. This keeps it proven after someone
        regenerates one of them: a load-time merge is safe exactly as long as the assumption it
        rests on is checked at load time, which is the same argument `require_assets` makes about
        missing files. Without the converter there is no build step where a mismatch would surface,
        so this IS that step.

        Cheap on purpose — a full comparison of 237 scenes' clouds on every dataset construction
        would tax every run to guard against a rare event. `mmap_mode='r'` reads the header only, so
        the shape check costs no I/O, and the fingerprint reads ~`FINGERPRINT_ROWS` strided rows.
        A regenerated or re-ordered root changes those rows with overwhelming probability; a
        deliberate adversary could evade it, but nothing here is adversarial.
        """
        bad_n, bad_fp = [], []
        for path in self.data_list:
            cp = os.path.join(self._child_dir(path), "coord.npy")
            if not os.path.exists(cp):        # some roots ship no coord; shape check then falls to
                cp = os.path.join(self._child_dir(path), "inter_gt.npy")   # the label length itself
            a = np.load(os.path.join(path, "coord.npy"), mmap_mode="r")
            b = np.load(cp, mmap_mode="r")
            if len(a) != len(b):
                bad_n.append((os.path.basename(path), len(a), len(b)))
                continue
            if b.ndim == 2 and b.shape[1] == a.shape[1]:
                step = max(1, len(a) // FINGERPRINT_ROWS)
                if not np.array_equal(np.asarray(a[::step]), np.asarray(b[::step])):
                    bad_fp.append(os.path.basename(path))
        assert not bad_n, (
            f"{type(self).__name__}: {len(bad_n)} scenes have different point counts in "
            f"{self.data_root} and {self.child_root} — e.g. {bad_n[:3]}. The roots have diverged "
            "and merging them by row index would pair each point's movable label with a different "
            "point's handle label. The two dataset roots must share a row order.")
        assert not bad_fp, (
            f"{type(self).__name__}: {len(bad_fp)} scenes match in length but differ in a strided "
            f"coordinate fingerprint — e.g. {bad_fp[:3]}. That is a row PERMUTATION between "
            f"{self.data_root} and {self.child_root}, which no length check would ever catch "
            "(same length is not same rows)."
            "the two dataset roots must share a row order.")

    # ------------------------------------------------------------------------------- data loading
    def get_data(self, idx):
        data = super().get_data(idx)
        path = self.data_list[idx % len(self.data_list)]
        d = self._child_dir(path)
        child_inst = np.load(os.path.join(d, "inter_gt.npy")).reshape(-1).astype(np.int32)
        child_seg = np.load(os.path.join(d, "segment.npy")).reshape(-1).astype(np.int32)

        n = len(data["coord"])
        assert len(child_inst) == len(child_seg) == n, (
            f"{os.path.basename(path)}: child labels have {len(child_inst)}/{len(child_seg)} rows "
            f"against {n} points in {self.data_root}. The two roots have diverged; merging them by "
            "row index would pair each point's movable label with a different point's handle label. "
            "The two dataset roots must share a row order.")
        if self._verified < self.verify_coords:
            self._verified += 1
            ref = np.load(os.path.join(d, "coord.npy"))
            assert np.array_equal(ref, data["coord"]), (
                f"{os.path.basename(path)}: coordinates differ between {self.data_root} and "
                f"{self.child_root} despite matching length — this is a row PERMUTATION, the "
                "silent failure the row-alignment gate exists to prevent.")

        data["child_instance"] = child_inst
        data["child_segment"] = child_seg

        # DECLARE THE CHILD KEYS AS INDEXABLE *HERE*, BEFORE ANY TRANSFORM RUNS. Found the hard way: a transform that runs first silently drops them.
        #
        # `index_operator` (transform.py:25-37) installs a DEFAULT `index_valid_keys` — coord,
        # color, normal, superpoint, strength, segment, instance — the first time any transform
        # selects rows, and the the Track-1 instance model pipeline's first three transforms (SphereCrop, RandomDropout,
        # SphereCrop) all select rows. The config's explicit `Update` comes much later, just before
        # GridSample. So relying on the config alone leaves the child arrays UNINDEXED through the
        # early crops and indexed from then on.
        #
        # The result is not a length error, which is what makes it dangerous: the final GridSample
        # index is derived from `coord` and applied to whatever length `child_instance` happens to
        # be, so the output length MATCHES and the values come from the wrong rows. The control
        # measured 12,605 differing elements at identical length (202,716 both) — a length check
        # passes this every time.
        #
        # Setting it on the sample makes the class correct on its own rather than dependent on a
        # config getting it right. The config's own `Update` must ALSO list the child keys, because
        # that Update replaces this list wholesale (its real job is dropping `superpoint`).
        data["index_valid_keys"] = ["coord", "color", "normal", "superpoint", "strength",
                                    "segment", "instance", "child_instance", "child_segment"]
        return data
