# Data

Everything here uses the organisers' released point clouds. Nothing in this repository requires the
held-out test split, and no script accepts one — the released code evaluates on validation only.

## What you need

| root | contents | used by |
|---|---|---|
| `data/a3d/processed/` | the organisers' release: `articulate3d_challenge_{mov,inter}/{train,validation}/<sid>.npy`, per-scene articulation HDF5, `instance_gt/` | the evaluator |
| `data/pointcept_mov/` | training-format clouds for Track 1: `coord/color/normal/segment/instance/superpoint` per scene | Track 1 |
| `data/pointcept_lite/` | the same clouds for Track 2, plus `expand.npz` for the label curriculum | Track 2 |

Every entry point takes `--data-root` and `--gt-root`, so the roots can live anywhere;
`reproduce_val.sh` reads `DATA_ROOT`, `LITE_ROOT` and `ARTI3D_GT_ROOT` from the environment.

## Preparation

```
python scripts/to_pointcept_mov.py                                 # -> data/pointcept_mov
python scripts/to_pointcept_lite.py                                # -> data/pointcept_lite
python scripts/make_superpoints.py                                 # adds superpoint.npy
python scripts/add_c2f_labels.py                                   # adds expand.npz
```

The source is read from `$ARTI3D_GT_ROOT` (default `data/a3d/processed`), so these take no source
argument. Each writes to the default output above; `--out` moves it, `--roots` selects which roots
`make_superpoints.py` walks, and `--workers N` and `--splits train validation` narrow the work:

```
python scripts/to_pointcept_mov.py  --out /path/to/pointcept_mov  --splits validation --workers 8
python scripts/make_superpoints.py  --roots /path/to/pointcept_mov --splits validation
```

**Row order is the contract.** Predictions are indexed against the challenge cloud's own row order,
so a silent permutation anywhere in preparation produces a perfectly well-formed submission that
scores zero. The conversion preserves it and `instances_t2.py` re-asserts it per scene against the
source `.npy` rather than trusting it.

**Superpoints** (`make_superpoints.py`) are a Felzenszwalb–Huttenlocher segmentation of a kNN graph.
The parameters are not the library defaults: they were swept for the achievable ceiling on movable
parts, since a superpoint that straddles a part boundary caps what any query decoder above it can
recover, no matter how well it is trained.

**The coarse-to-fine labels** (`add_c2f_labels.py`) write `expand.npz`, the dilated handle targets the
Track-2 curriculum anneals over training. They are needed to RETRAIN Track 2; inference from a
released checkpoint does not read them. The `noc2f` ablation config does not use them at all.
