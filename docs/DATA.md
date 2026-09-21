# Data

Everything runs on the organisers' released point clouds of
[Articulate3D](https://huggingface.co/datasets/INSAIT-Institute/Articulate3D) (ScanNet++ scenes
with part, motion and handle annotations), in the processed form the challenge distributes:
per-scene `.npy` clouds with 13 columns (coordinates, colour, normal, semantic and instance labels,
segment ids, interaction labels), articulation HDF5 files, decoded instance ground truth and the
`expand_dict/` dilation records. Nothing here needs the held-out test split, and no script accepts
one: the released code evaluates on the 42-scene validation split only.

## What you need

| root | contents | used by |
|---|---|---|
| `data/a3d/processed/` | the organisers' processed release: `articulate3d_challenge_{mov,inter}/{train,validation}/<sid>.npy`, per-scene articulation HDF5, `instance_gt/`, `expand_dict/` | the evaluator, the preparation scripts |
| `data/pointcept_mov/` | training-format clouds for Track 1 and the joint model: `coord / color / normal / segment / instance / superpoint` per scene | `infer_t1.py`, `infer_s2_child.py` |
| `data/pointcept_lite/` | the same clouds for Track 2, plus `expand.npz` for the label curriculum | `infer_t2_sem.py`, `instances_t2.py`, `classvote.py` |

Every entry point takes `--data-root` and `--gt-root`, so the roots can live anywhere;
`reproduce_val.sh` reads `DATA_ROOT`, `LITE_ROOT` and `ARTI3D_GT_ROOT` from the environment, and the
evaluator's default root is `$ARTI3D_GT_ROOT`, else `data/a3d/processed`.

## Preparation

Four scripts, each reading the organisers' release from `$ARTI3D_GT_ROOT` and writing the default
root above (`--out` moves it; `--splits train validation` and `--workers N` narrow the work):

```bash
export PYTHONPATH="$PWD:$PWD/third_party/volt"
python scripts/to_pointcept_mov.py                      # -> data/pointcept_mov
python scripts/to_pointcept_lite.py                     # -> data/pointcept_lite
python scripts/make_superpoints.py --roots data/pointcept_mov data/pointcept_lite   # adds superpoint.npy
python scripts/add_c2f_labels.py                        # adds expand.npz under data/pointcept_lite
```

Inference from the released checkpoints needs the first three (superpoints are the Track-1 model's
attention units; both roots receive the same array because both hold the same cloud). The
coarse-to-fine labels are needed only to **retrain** the Track-2 dense model; the `noc2f` variant
config does not use them at all.

## Three contracts

**Row order.** Predictions are indexed against the challenge cloud's own point order, so a silent
permutation anywhere in preparation produces a well-formed prediction file that scores zero. The
conversion asserts on every scene that row *i* of the output is row *i* of the source cloud, and
`instances_t2.py` re-asserts it per scene against the source `.npy` rather than trusting the files
on disk.

**Superpoints.** `make_superpoints.py` runs a Felzenszwalb–Huttenlocher segmentation over a
12-nearest-neighbour graph at `k_thresh` 0.005 and `seg_min` 5 — not the library defaults: the
setting was swept for the achievable ceiling on movable parts (98.4 % macro on validation
ground-truth masks; the nearby 0.01 setting lost 5 points on translations), since a superpoint that
straddles a part boundary caps what any decoder above it can recover. The function is deterministic
in the cloud and the two parameters, which matters because the superpoints are baked into the
Track-1 checkpoint through its mask targets: a regenerated, subtly different partition would
silently invalidate the model.

**Ground truth encoding.** The evaluator encodes an instance as `semantic × 1000 + id + 1`; the
training data keeps raw ids. Encoding is a scoring concern and happens only inside `arti3d/eval/`.
