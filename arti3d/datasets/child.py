"""`ChildInstanceParser` — the child branch's equivalent of Pointcept's `InstanceParser`.

WHY THIS HAS TO EXIST. `InstanceParser` (transform.py:1294) handles the key `instance` and nothing
else: it maps instances of ignored classes to -1, renumbers the survivors to a contiguous
`0..K-1`, and builds centroid/bbox tensors. The child branch carries `child_instance`, which the
stock transform never touches, and the raw values there are the interactable ids straight out of
`inter_gt.npy` — arbitrary, non-contiguous, and with **0 meaning background** rather than -1.

Handing those to a criterion is a silent failure of the usual kind: a Hungarian matcher over
`unique(child_instance)` would happily treat background as instance number zero and match a query to
it, so the model would learn to predict "everything that is not a handle" as its first instance.
Nothing about that crashes and the loss falls perfectly smoothly.

CONVENTIONS, chosen to match the parent branch exactly so the two halves of the criterion can share
code and so anyone reading the model does not have to hold two schemes in their head:

    background / ignored  ->  -1        (parent: `instance_ignore_index=-1`)
    real instances        ->  0..K-1    contiguous, ordered by first appearance of the sorted id

`min_voxels` DEFAULTS TO 0, DELIBERATELY. the measured rule for the child head is `npoint_thr = 0`, and the
same reasoning applies with more force to the targets than to the predictions: 17.9% of interactable
instances are under 10 points, and dropping a GT instance does not make it stop counting — the
evaluator still scores it against us, so a dropped target is a guaranteed miss traded for a slightly
cleaner loss. The parameter exists only so the effect can be measured, never as a default.

`child_segment` is remapped alongside for the ignore convention only ({0 background} -> -1); its
class values {1 rotation-handle, 2 translation-handle} are left in the CHALLENGE encoding, because
 is emphatic that a blind +1/-1 anywhere in this pipeline is how predictions get silently dropped.

Runs AFTER GridSample, like `InstanceParser` does — the ids must be renumbered over the voxels that
actually survive, or the contiguous range would include instances that no longer have any points.
"""
from __future__ import annotations

import numpy as np

from pointcept.datasets.transform import TRANSFORMS

IGNORE = -1


@TRANSFORMS.register_module()
class ChildInstanceParser(object):
    """Raw `inter_gt` ids -> contiguous child instance ids with a -1 ignore convention."""

    def __init__(self, min_voxels: int = 0, instance_ignore_index: int = IGNORE,
                 background_ids=(0,)):
        self.min_voxels = int(min_voxels)
        self.ignore = int(instance_ignore_index)
        self.background_ids = tuple(background_ids)

    def __call__(self, data_dict):
        assert "child_instance" in data_dict, (
            "ChildInstanceParser found no `child_instance`. Either the dataset is not "
            "ArtiJointDataset, or the key was dropped before this transform — check that the "
            "config's `Update(index_valid_keys=...)` lists it.")
        inst = np.asarray(data_dict["child_instance"]).reshape(-1).astype(np.int64).copy()

        fg = ~np.isin(inst, self.background_ids)
        if self.min_voxels > 1 and fg.any():
            ids, counts = np.unique(inst[fg], return_counts=True)
            too_small = ids[counts < self.min_voxels]
            if len(too_small):
                fg &= ~np.isin(inst, too_small)

        out = np.full(inst.shape, self.ignore, dtype=np.int64)
        if fg.any():
            _uniq, inverse = np.unique(inst[fg], return_inverse=True)
            out[fg] = inverse
        data_dict["child_instance"] = out

        if "child_segment" in data_dict:
            seg = np.asarray(data_dict["child_segment"]).reshape(-1).astype(np.int64).copy()
            # Ignore where there is no child instance, so segment and instance agree on what is
            # foreground. Class VALUES stay in the challenge encoding {1, 2} — see .
            seg[out == self.ignore] = self.ignore
            data_dict["child_segment"] = seg
        return data_dict
