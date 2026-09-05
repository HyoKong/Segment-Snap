"""Coarse-to-fine GT dilation for Pointcept (Arm A).

Pointcept has no c2f support; this adds it. USDNet inherits the idea from SceneFun3D, whose own
ablation on tiny functional parts is AP50 **18.3** at r0=0.1, **9.8** at r0=0.05 and **0.0** with no
dilation — on a target whose foreground is the same 0.05–0.1% order as ours.

`expand_dict/<scene>.pkl` (shipped inside the organisers' processed.zip) holds, per scene:
    expand_idx_records   (K,) point indices near some instance
    expand_dist_records  (K,) distance from that instance
    expand_sem_records   (K,) semantic label to assign
    expand_inst_records  (K,) instance id to assign
Dilating to radius r means: additionally label `expand_idx[dist <= r]` with `expand_sem[...]`.

**The schedule must END on raw labels.** At 2 cm resolution a 0.04 m dilation is a ~2-voxel ring
around a ~20-point handle; a model that only ever sees dilated targets learns systematically fat
masks, and at IoU 0.5 on tiny instances that bias is exactly what drops matches below threshold.
The evaluator scores undilated GT, so the last stage must too. Default schedule:

    first 50% of training : r = 0.10   (learnable signal at 0.08% foreground)
    next  30%             : r = 0.04
    final 20%             : r = 0.00   (raw labels — free, needs no expand_dict lookup)

Radius is carried on the dataset CLASS (not the instance) because Pointcept clones datasets into
dataloader workers; a hook sets it once per epoch in the parent and workers inherit it at fork.
"""
from __future__ import annotations

import os
from bisect import bisect_right

import numpy as np


from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset
from pointcept.datasets.transform import TRANSFORMS
from pointcept.engines.hooks import HOOKS, HookBase


#: (start FRACTION of training, dilation radius in metres). Fractions, not absolute epochs,
#: deliberately: the trainer runs `max_epoch = cfg.eval_epoch` outer epochs and folds
#: `cfg.epoch // cfg.eval_epoch` dataset passes into each, so with epoch=200 and eval_epoch=20 the
#: epoch counter only ever reaches 19. An absolute schedule keyed at 100/160 would never fire and
#: the run would end at maximum dilation — precisely the failure this schedule exists to prevent.
DEFAULT_SCHEDULE = ((0.0, 0.10), (0.5, 0.04), (0.8, 0.00))


def radius_for_epoch(epoch: int, max_epoch: int, schedule=DEFAULT_SCHEDULE) -> float:
    frac = epoch / max(max_epoch, 1)
    starts = [s for s, _ in schedule]
    return schedule[max(0, bisect_right(starts, frac) - 1)][1]


@DATASETS.register_module()
class ArtiC2FDataset(DefaultDataset):
    """DefaultDataset + epoch-dependent GT dilation applied to `segment`.

    `dilate` SAYS WHETHER THIS DATASET IS THE ONE BEING TRAINED ON, and it exists because the
    original proxy for that question — `self.split == "train"` — is wrong the moment a split stops
    being the literal string "train". The train+val runs use a json split file
    (a json split file), for which that test evaluates False: the dilation would be silently switched
    off for the whole run, producing an un-curriculumed model wearing a curriculum config. The
    resulting model looks entirely normal — the ablation without the curriculum lands inside the
    seed noise floor — so no downstream check would fire.

    Default `None` keeps the behaviour of a plain directory split. A split that is NOT a plain
    directory name must state `dilate` explicitly: guessing is what makes this failure possible,
    and a construction-time crash is cheap.
    """

    current_radius: float = 0.0        # class attribute; set by ArtiC2FHook

    def __init__(self, dilate: bool | None = None, size_weight=False, sw_radius=0.025,
                 sw_target=100.0, sw_power=1.0, sw_clip=(1.0, 12.0), **kwargs):
        self._size_weight = bool(size_weight)
        self._sw_radius, self._sw_target = float(sw_radius), float(sw_target)
        self._sw_power, self._sw_clip = float(sw_power), tuple(sw_clip)
        super().__init__(**kwargs)
        if dilate is None:
            split = kwargs.get("split", "train")
            simple = isinstance(split, str) and os.path.isdir(
                os.path.join(self.data_root, split))
            assert simple, (
                f"{type(self).__name__}(split={split!r}): `dilate` must be stated explicitly for a "
                "merged or file-based split. The legacy default infers it from `split == 'train'`, "
                "which silently disables c2f dilation here and would train a noc2f model under a "
                "c2f run-id.")
            dilate = split == "train"
        self._dilate = bool(dilate)

    def _add_size_weight(self, data):
        """Per-point weight ~ (target / component_size)**power, computed from the GT labels.

        The component is the SAME object Arm A clusters at inference (`connected_components` at the
        shipped radius), so "instance" means one thing at train time and test time. Background keeps
        weight 1.0 — this experiment is about small-vs-large FOREGROUND, not foreground-vs-background,
        which is a different question with its own name.
        """
        from arti3d.prep.cluster import connected_components
        seg = data["segment"]
        coord = data["coord"]
        w = np.ones(len(seg), dtype=np.float32)
        for c in (1, 2):
            idx = np.flatnonzero(seg == c)
            if idx.size == 0:
                continue
            comp = connected_components(coord[idx], self._sw_radius)
            sizes = np.bincount(comp)
            per_point = sizes[comp].astype(np.float32)
            w[idx] = np.clip((self._sw_target / np.maximum(per_point, 1.0)) ** self._sw_power,
                             self._sw_clip[0], self._sw_clip[1])
        data["size_weight"] = w
        return data

    def get_data(self, idx):
        data = super().get_data(idx)
        if self._size_weight:
            data = self._add_size_weight(data)
        r = type(self).current_radius
        # Dilate the TRAINING dataset only. `test_mode` is False for the val split too, so gating
        # on it would have dilated validation labels and made every reported mIoU incomparable to
        # the evaluator, which scores raw GT.
        if r <= 0 or not self._dilate:
            return data
        # Address the scene by the path the loader actually resolved, NOT by rebuilding
        # `data_root/split/name`. Under a json split the scenes come from several directories at
        # once ("train/x" and "validation/y"), so there is no single `split` component to rebuild
        # with — the old form pointed at `data_root/tv_train229.json/<name>/expand.npz`, which never
        # exists, and the `os.path.exists` guard below would have turned that into a silent no-op.
        p = os.path.join(self.data_list[idx % len(self.data_list)], "expand.npz")
        if not os.path.exists(p):
            return data
        z = np.load(p)
        keep = z["dist"] <= r
        if not keep.any():
            return data
        idxs = z["idx"][keep]
        seg = data["segment"].copy()
        valid = idxs < len(seg)         # guard: expand_dict indexes the full cloud
        seg[idxs[valid]] = z["sem"][keep][valid].astype(seg.dtype)
        data["segment"] = seg
        return data


@HOOKS.register_module()
class ArtiC2FHook(HookBase):
    """Sets ArtiC2FDataset.current_radius once per epoch, and logs each transition."""

    def __init__(self, schedule=DEFAULT_SCHEDULE):
        self.schedule = tuple(tuple(s) for s in schedule)
        assert all(0.0 <= f <= 1.0 for f, _ in self.schedule), (
            f"c2f schedule must be FRACTIONS of training in [0,1], got {self.schedule}")
        assert self.schedule[-1][1] == 0.0, (
            "the c2f schedule MUST end at radius 0.0 — a model that never trains on raw labels "
            "learns systematically fat masks, and the evaluator scores undilated GT")
        self._last = None

    def before_epoch(self):
        r = radius_for_epoch(self.trainer.epoch, self.trainer.max_epoch, self.schedule)
        ArtiC2FDataset.current_radius = r
        if r != self._last:
            self.trainer.logger.info(
                f"[c2f] epoch {self.trainer.epoch}: GT dilation radius -> {r} m"
                + (" (RAW labels — final stage)" if r == 0.0 else ""))
            self._last = r
