# Segment–Snap

**Geometric and semantic coupling for interaction understanding in 3D scenes** — reference
implementation and train-only checkpoints for two coupled tasks on indoor scans (Articulate3D):

* **Track 1 — movable parts and their motion.** Segment each movable part and predict its motion
  axis and hinge origin.
* **Track 2 — interactable handles.** Segment the handles through which the parts are operated.

Learned predictors find the broad part surfaces and the small handles. Everything else is geometry
and one-way transfer between the two tasks:

1. **Motion is decoded, not regressed.** A part's motion is chosen from its own oriented bounding
   box: a vertical prior for rotation axes, the fitted plane normal for translation axes, and a
   hinge line selected among the box edges by the part's predicted handle. There is no learned
   motion head in this repository. In this setting the training-free decoder is effective, and
   matched learned alternatives on the same frozen features did not improve on it (measured in the
   paper; that comparison is not part of this release).
2. **Handles → parts.** Track 2's predicted handles decide where Track 1 places a hinge. With masks
   and axes held fixed, this raises the motion-gated AP from 0.13739 to 0.40984: **+0.272 absolute**,
   a factor of 2.98 over a weak no-handle baseline.
3. **Parts → handles.** The joint part-and-handle model supplies additional handle proposals,
   appended strictly below the dense detections (**+0.050 AP50**), and predicted parts then correct
   those proposals' motion classes (**+0.013 AP50**). Each transfer is applied once; there is no
   feedback loop.

Method: [`docs/METHOD.md`](docs/METHOD.md). Every number with its provenance:
[`docs/RESULTS_VAL.md`](docs/RESULTS_VAL.md). Data: [`docs/DATA.md`](docs/DATA.md).
Checkpoints: [`checkpoints/README.md`](checkpoints/README.md).

---

## Results on validation

42 validation scenes, train-only checkpoints (they never saw a validation scene), one RTX 5070 Ti,
fp32. `scripts/reproduce_val.sh` prints every number in both tables.

### Track 1 — movable parts and motion

| configuration | AP50 | AP50_axis | AP50_origin | **AP50_axis_origin** (ranking column) |
|---|---:|---:|---:|---:|
| **released decode** (origins from the predicted handles) | 0.47930 | 0.43747 | 0.43115 | **0.40984** |
| control: origins at the box centroid (no handles) | 0.47930 | 0.43747 | 0.14373 | 0.13739 |

AP50 and the axis-gated column are identical to the last digit: handles enter the pipeline only
through the hinge origin. The +0.27245 on the ranking column has a paired scene-jackknife 95 %
interval of [+0.2186, +0.3263]. All of it is on rotations (+0.5449 on the rotation class); the
metric's origin test never binds on a translation, so that class is unchanged.

### Track 2 — interactable handles

| stage | AP50 |
|---|---:|
| dense semantic model | 0.24635 |
| + handle proposals from the joint model, appended below | 0.29649 |
| + class vote from Track 1's parts | **0.30991** |

Full precision, the per-class split, the component ladder, the controls, the released model among
its training family, and cross-hardware reproducibility: [`docs/RESULTS_VAL.md`](docs/RESULTS_VAL.md).

---

## Setup

```bash
git clone --recursive https://github.com/HyoKong/Segment-Snap.git
cd Segment-Snap
```

The backbone lives in `third_party/volt`, a submodule pinned to the `articulate3d` branch of our
[Volt fork](https://github.com/HyoKong/Volt) (`bash setup.sh` fetches it if the clone was not
recursive). Our four commits on top of upstream Volt `df41b45`:

| commit | what it does |
|---|---|
| `models: make optional model families import-optional` | the model registry loads without every compiled extension (pointops, pointgroup_ops); missing families warn and are skipped |
| `volt: flash-attn fallback and RoPE frequency scaling` | falls back to `scaled_dot_product_attention` where flash-attn has no build; rotary-frequency scaling lets a model pretrained at one voxel grid transfer to another |
| `train: skip OOM and degenerate batches instead of aborting the run` | one unlucky crop no longer kills a multi-day run |
| `test: decouple test-loader workers and pin_memory from batch size` | test-time loading was tied to a batch size of 1, leaving the GPU idle through each scene's CPU voxelisation |

**Environment.** Python 3.12. The numbers above were produced with `torch 2.8.0+cu128`; the
commands below install that exact stack (with plain `pip`, drop the leading `uv`).

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
# one resolution pass, so nothing later upgrades torch behind your back
uv pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch==2.8.0 torchvision==0.23.0 numpy scipy h5py addict \
    timm einops open3d plyfile scikit-learn pandas plotly termcolor yapf tqdm pyyaml peft
# must match the torch build exactly; they come from the PyG wheel index, not PyPI
uv pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
# imported by the backbone's model registry; never used in computation
uv pip install spconv-cu126
export PYTHONPATH="$PWD:$PWD/third_party/volt"
```

`requirements.txt` lists our package's own imports (torch, numpy, scipy, h5py, addict). The second
group above is imported by the backbone's model registry and dataset code (Pointcept), so it is
required at import time even though the released models use none of it. Volt's compiled extensions
and flash-attn are optional: without them the registry skips the model families this project does
not use, and attention falls back to `scaled_dot_product_attention` (numerically equivalent, slower).

## Reproducing the tables

```bash
# the three checkpoints, from Hugging Face imsuperkong/Segment-Snap, md5-verified (4.9 GB)
python scripts/download_checkpoints.py --dest checkpoints

# data preparation: docs/DATA.md (four scripts over the organisers' processed release)

DATA_ROOT=data/pointcept_mov LITE_ROOT=data/pointcept_lite ARTI3D_GT_ROOT=data/a3d/processed \
    bash scripts/reproduce_val.sh
```

The script runs six stages in the only order the dependency graph allows — Track 2's handle
instances are Track 1's hinge cue, and Track 1's parts are Track 2's class prior — and prints both
tables plus the no-handle control. The evaluator is the organisers' own, vendored byte-identically
under `arti3d/eval/`; its known-answer tests run with

```bash
ARTI3D_GT_ROOT=data/a3d/processed python -m arti3d.eval.selftest
```

## Repository layout

```
arti3d/            the package: geom/ (box, axis, hinge, rescoring) · prep/ (superpoints, components)
                   datasets/ · models/ (SPFormer selection, query tracking, the joint model) · eval/ (vendored evaluator)
scripts/           the six inference stages, reproduce_val.sh, data preparation, checkpoint download
configs/arti3d/    training configurations, exactly as trained
checkpoints/       the three released config.py files; the weights download here
docs/              METHOD.md · RESULTS_VAL.md · DATA.md
third_party/volt   the Volt backbone (submodule)
```

Training uses the configurations in `configs/arti3d/` through Volt's trainer; the recipe is in
[`docs/METHOD.md`](docs/METHOD.md#training).

## Relation to our challenge entry

Our competition entry, built on this method, placed first on both tracks of the Articulate3D
challenge test set. It included additional engineering that is not part of this release, and every
number in this repository is a validation result of the released configuration.

## Citation

```bibtex
@article{kong2026segmentsnap,
  title  = {Geometric and Semantic Coupling for Interaction Understanding in 3D Scenes},
  author = {Kong, Hanyang and Yang, Xingyi},
  year   = {2026},
  note   = {preprint in preparation}
}
```

## License

MIT (see `LICENSE`). `arti3d/eval/evaluate_semantic_instance.py`, `util.py` and `util_3d.py` are
vendored byte-identically from the organisers' USDNet benchmark (MIT) and keep their own headers;
the Volt submodule is MIT (© Kadir Yilmaz).
