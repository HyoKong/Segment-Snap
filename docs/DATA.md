# Data preparation

[Overview](../README.md) · [Method](METHOD.md) · [Validation results](RESULTS_VAL.md) · [Checkpoints](../checkpoints/README.md)

The pipeline uses two aligned views of each scene: movable-part inputs and dense-handle inputs.
The part and dense predictors read their respective views, while the joint part-handle predictor
uses both. Historical names such as `t1`, `t2`, and `s2` remain for script and checkpoint
compatibility; the guides describe their roles in one coupled interaction-understanding pipeline.

## Prerequisite: the processed Articulate3D release

Download and unpack the organisers' processed [Articulate3D release](https://huggingface.co/datasets/INSAIT-Institute/Articulate3D). These commands expect its root at `data/a3d/processed`; alternatively, set `ARTI3D_GT_ROOT` to its location before running preparation or evaluation. The release supplies the `mov` and `inter` clouds, articulation HDF5 files, decoded `instance_gt/`, and `expand_dict/` records.

The released preparation defaults and validation reproduction use 195 training scenes and 42 validation scenes. The held-out test split is not required by this release's evaluator workflow.

Each source cloud is an `(N, 13)` `float32` array. Columns are, in order:

| Columns | Meaning |
|---|---|
| `0:3` | XYZ coordinates, in metres |
| `3:6` | RGB colour |
| `6:9` | surface normal |
| `9` | semantic motion/interaction class: background `0`, rotation `1`, translation `2` |
| `10` | movable-part instance identifier |
| `11` | organiser segment identifier |
| `12` | interactable parent-part identifier |

The `mov` and `inter` clouds have matching geometry, colour, and normal values in columns `0:9`; their label columns serve different annotations. Point order, coordinate frame, and metre units are contracts: generated arrays and predictions must retain source row `i` at row `i`. The conversion scripts check row counts, and superpoint generation refuses roots whose coordinates differ. Do not reorder, voxelise, or independently transform a cloud before using its predictions with this release.

## Prepare the derived roots

From the repository root, after the environment setup in [the overview](../README.md#quick-start), run:

```bash
export PYTHONPATH="$PWD:$PWD/third_party/volt"
export ARTI3D_GT_ROOT=data/a3d/processed

python scripts/to_pointcept_mov.py --out data/pointcept_mov
python scripts/to_pointcept_lite.py --out data/pointcept_lite
python scripts/make_superpoints.py \
  --roots data/pointcept_mov data/pointcept_lite
python scripts/add_c2f_labels.py --out data/pointcept_lite
```

The two conversion scripts must finish before `make_superpoints.py`; `add_c2f_labels.py` then copies the organiser's `inter/expand_dict` records into the lite root.

`to_pointcept_mov.py` and `to_pointcept_lite.py` accept `--out`, `--workers`, and `--splits`; their defaults create both `train` and `validation`. They read the source root through `ARTI3D_GT_ROOT`, not a `--gt-root` flag. `make_superpoints.py` accepts `--roots`, `--workers`, `--splits`, `--k-thresh`, and `--seg-min`; use its defaults (`0.005`, `5`) with released checkpoints. `add_c2f_labels.py` accepts `--out` and `--track`.

For released-checkpoint inference, run the two conversions and superpoint generation. `expand.npz` is not read by inference; it is needed only to retrain the curriculum-based dense-handle configuration (the `noc2f` configuration does not use it). Retraining also needs the training split, while validation reproduction only needs validation data and the evaluator's processed release.

## Derived layout

```text
data/
  a3d/processed/
    articulate3d_challenge_{mov,inter}/
      {train,validation}/<scene>.npy
      {train,validation}/<scene>_articulation.h5
      instance_gt/{train,validation}/<scene>.txt
      expand_dict/<scene>.pkl
  pointcept_mov/<split>/<scene>/
    coord.npy color.npy normal.npy segment.npy instance.npy superpoint.npy
  pointcept_lite/<split>/<scene>/
    coord.npy color.npy normal.npy segment.npy inter_gt.npy [expand.npz] [superpoint.npy]
```

`pointcept_mov/segment.npy` labels movable parts and `instance.npy` carries their raw identifiers. `pointcept_lite/segment.npy` labels handles by their parent motion class; `inter_gt.npy` carries the parent movable identifier. Keep these roots separate: both use the filename `segment.npy`, but it has different meanings. The joint predictor reads the movable root together with the lite root.

## Common problems

- `FileNotFoundError` under `data/a3d/processed`: set `ARTI3D_GT_ROOT` to the unpacked processed-release root, not to a track directory.
- A missing `superpoint.npy` means `make_superpoints.py` has not completed. Regenerate it with the default parameters and the same converted roots.
- Do not substitute the lite root for the movable root. It lacks movable instances and its `segment.npy` is a handle label.
- A missing `expand.npz` blocks only curriculum-based dense-model retraining. Run `add_c2f_labels.py` after the lite conversion and ensure the processed release includes `expand_dict/`.
- If evaluation reports point-count or ground-truth coverage failures, rebuild from the unmodified organiser clouds; those checks usually indicate a row-order or root mismatch.
