# Articulate3D: movable parts and their motion, without a motion head

Reference implementation for two tasks on indoor 3D scans:

* **Track 1** — segment the movable parts of a scene and predict each part's motion axis and hinge
  origin.
* **Track 2** — segment the interactable handles.

The central claim is that **articulation does not need to be regressed**. Given a part's mask, its
axis and hinge origin can be chosen — discretely — from the part's own oriented bounding box, and
that choice beats a learned motion head. There is no motion head in this repository.

The second claim is that **the two tracks should feed each other**. Track 2's handles decide where
Track 1 puts a hinge; Track 1's parts decide what class Track 2 gives a handle. Removing the first
coupling costs a factor of **2.98** on Track 1's ranking metric.

---

## Results on validation

Produced by `scripts/reproduce_val.sh` from the three released checkpoints on one RTX 5070 Ti. The
checkpoints are **train-only** — they never saw a validation scene. Full precision, and the
provenance of every number, in [`docs/RESULTS_VAL.md`](docs/RESULTS_VAL.md).

### Track 1

| | AP50 | AP50_axis | AP50_origin | **AP50_axis_origin** |
|---|---|---|---|---|
| released configuration | 0.47930 | 0.43747 | 0.43115 | **0.40984** |
| ablation: no handles, origin = box centroid | 0.47930 | 0.43747 | 0.14373 | 0.13739 |

**×2.98 from the cross-track coupling.** AP50 and AP50_axis are identical to the last digit in both
rows — handles enter only through the origin, so nothing else can move.

### Track 2

| stage | AP50 |
|---|---|
| single semantic model | 0.24635 |
| + child-head union (the *S2* joint model) | 0.29649 |
| + class vote from Track 1's parts | **0.30991** |

### Which Track-2 checkpoint we release, and why that choice is weak

All five train-only members, each through the full chain:

| member | single | + union | + class vote |
|---|---|---|---|
| `armA_r1` | 0.24419 | 0.28964 | 0.30372 |
| **`armA_long` — released** | **0.24635** | **0.29649** | **0.30991** |
| `armA_noc2f` | 0.24299 | 0.29260 | 0.30599 |
| `armA_seed2` | 0.23977 | 0.28378 | 0.29696 |
| `armA_bgw03` | 0.24527 | 0.29466 | 0.30932 |

> The released member is the argmax of the full validation pipeline. The top two members are
> separated by 0.00058, while two runs of the same configuration differing only in training seed are
> separated by 0.00675 — 11.6× larger — so this selection is not resolved by the evidence, and any of the top three
> members would be a defensible release.


---

## Two things a reader should know before trusting the numbers

**How the child is associated with its parent matters, and the release gets it right.** The child
head emits one probability map per *query*, while the instance head emits a reordered, filtered
subset. Fetching the child by the parent's *position* in the output rather than by the query index
it came from leaves only 7.7 % of children inside their own parent, against 54.6 % when the query
index is tracked through top-k and NMS, and drives class agreement with the matched ground-truth
handle down to 54.8 % — chance, for two classes — against 63.1 %. On validation the correct
association is worth **+0.0124 AP50** at the union stage.

**Reproducibility across hardware.** Against the original research code **on the same GPU**, this
implementation is bit-identical in masks, axes and origins, and its per-instance scores differ by no
more than two runs of the same code differ from each other (max 2×10⁻⁵, from non-deterministic GPU
reductions in the instance head) — which changes no metric column. Against a reference artifact
produced on an **RTX 5090**, Track 1 reproduces the ranking column to **4×10⁻⁶**: nine of 6285 masks
flip by whole superpoints at the per-superpoint threshold. Inference is fp32 with TF32 disabled.

---

## Setup

```bash
git clone --recursive https://github.com/<user>/<repo>
cd <repo>
# if you forgot --recursive:
git submodule update --init --recursive
bash setup.sh
```

The backbone lives in `third_party/volt`, a submodule pinned to the `articulate3d` branch of our
Volt fork. Our changes to Volt are **four commits** on top of upstream `df41b45`, and they are
exactly these:

| commit | what it does |
|---|---|
| `models: make optional model families import-optional` | the model registry loads without every compiled extension (spconv, pointops, pointgroup_ops); missing families warn and are skipped |
| `volt: flash-attn fallback and RoPE frequency scaling` | falls back to `scaled_dot_product_attention` where flash-attn has no build, and adds rotary-frequency scaling so a model pretrained at one voxel grid transfers to another |
| `train: skip OOM and degenerate batches instead of aborting the run` | one unlucky crop no longer kills a multi-day run |
| `test: decouple test-loader workers and pin_memory from batch size` | test-time data loading was tied to a batch size of 1, leaving the GPU idle through each scene's CPU voxelisation |

Data preparation: [`docs/DATA.md`](docs/DATA.md). Checkpoints and their md5s:
[`checkpoints/README.md`](checkpoints/README.md).

## Reproducing

```bash
DATA_ROOT=... LITE_ROOT=... ARTI3D_GT_ROOT=... bash scripts/reproduce_val.sh
```

Six stages in the only order the dependency graph allows — Track 2's instances are Track 1's
handles, and Track 1's parts are Track 2's class prior. The script prints both tracks' tables and the
Track-1 ablation.

Training: [`docs/METHOD.md`](docs/METHOD.md) describes each model; `configs/arti3d/` holds the
configurations exactly as they were trained.

## Evaluation

`arti3d/eval/` vendors the organisers' evaluator **byte-identically**, so any divergence from
official scoring is visible in a diff. `python -m arti3d.eval.selftest` runs known-answer tests
against it, including two sharp edges worth knowing about: an unclipped `arccos` that returns NaN for
a *bit-exact* axis, and an origin projection that divides by the axis norm once rather than twice, so
a non-unit axis silently corrupts the origin gate while passing the axis gate.

---

## Relation to our challenge entry

Our competition entry, built on this method, placed **first on both tracks of the Articulate3D
challenge test set**. It included additional engineering that is not part of this release, and the
numbers reported here are validation results of the released configuration.

## License

MIT. The vendored evaluator retains its upstream attribution; the Volt submodule is MIT
(© Kadir Yilmaz).
