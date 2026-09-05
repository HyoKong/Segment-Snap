"""Deterministic voxelisation, for pipelines that need `inverse` and cannot use test mode.

Validation/test pipelines must be **deterministic** (replace the `np.random.randint`
voxel representative; document the mechanism)". This is that replacement.

WHY IT IS NEEDED AT ALL, given `mode='test'` is already deterministic
---------------------------------------------------------------------
`GridSample` has two modes and they return different SHAPES:

  * `mode='test'` returns a LIST of fragments covering every original point. Deterministic (it
    enumerates `i in range(count.max())`, no RNG). This is what semseg inference uses, and it is
    what `scripts/armA_infer.py` runs.
  * `mode='train'` returns ONE subsampled cloud plus an `inverse` map back to the original points.
    Its voxel representative is chosen by `np.random.randint` (`transform.py:914`) — the defect.

Instance segmentation cannot use the fragment form: SPFormer's masks live at **superpoint**
resolution and are expanded through the voxel cloud (`spformer.py:185`), and `InsSegTester` needs
one cloud plus `origin_*` and `inverse` to map back. So the insseg val/test pipeline is stuck with
the `mode='train'` shape — and therefore with the RNG, unless it is replaced. Hence this transform:
the `mode='train'` shape with the representative chosen deterministically.

MECHANISM
----------------------------------------
Upstream picks, within each voxel's run of the key-sorted index array, an offset
`np.random.randint(0, count.max(), count.size) % count`. We take offset **0** — the first member of
each run under a **stable** sort of the hash key. Stability is what makes it well-defined: with an
unstable sort the "first member" of a tied run depends on the sort's internal pivoting, which is
reproducible for identical input but not something to rely on. `np.argsort(kind='stable')` makes the
representative the surviving point with the lowest original index, which is a property of the data
alone.

Consequence to be aware of when comparing numbers: this changes WHICH point represents each voxel
relative to a random draw, so val metrics from this transform are not bit-comparable with metrics
produced by the stock random one. It removes variance; it does not reproduce any particular sample.
"""
from __future__ import annotations

import numpy as np

# TRANSFORMS lives in `datasets.transform`, not `datasets.builder` (which only holds DATASETS).
from pointcept.datasets.transform import TRANSFORMS, GridSample, index_operator

__all__ = ["GridSampleDeterministic"]


@TRANSFORMS.register_module()
class GridSampleDeterministic(GridSample):
    """`GridSample(mode='train')` with the voxel representative chosen deterministically."""

    def __init__(self, **kwargs):
        kwargs.pop("mode", None)
        super().__init__(mode="train", **kwargs)

    def __call__(self, data_dict):
        assert "coord" in data_dict
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)

        key = self.hash(grid_coord)
        idx_sort = np.argsort(key, kind="stable")      # stable: see MECHANISM above
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        # offset 0 within each run, instead of np.random.randint(...) % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
        idx_unique = idx_sort[idx_select]

        data_dict = index_operator(data_dict, idx_unique)
        if self.return_inverse:
            data_dict["inverse"] = np.zeros_like(inverse)
            data_dict["inverse"][idx_sort] = inverse
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[idx_unique]
            if "grid_coord" not in data_dict["index_valid_keys"]:
                data_dict["index_valid_keys"].append("grid_coord")
        if self.return_min_coord:
            data_dict["min_coord"] = min_coord.reshape([1, 3])
        if self.return_displacement:
            displacement = scaled_coord - grid_coord - 0.5
            if self.project_displacement:
                displacement = np.sum(displacement * data_dict["normal"], axis=-1, keepdims=True)
            data_dict["displacement"] = displacement[idx_unique]
            if "displacement" not in data_dict["index_valid_keys"]:
                data_dict["index_valid_keys"].append("displacement")
        return data_dict
