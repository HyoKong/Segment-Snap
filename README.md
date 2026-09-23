# Segment–Snap

**Geometric and Semantic Coupling for Interaction Understanding in 3D Scenes**

[Hanyang Kong](https://hyokong.github.io/)¹ · [Xingyi Yang](https://adamdad.github.io/)²†<br>
¹ National University of Singapore · ² The Hong Kong Polytechnic University<br>
† Corresponding author: [Xingyi Yang](mailto:xingyi.yang@polyu.edu.hk)

<p align="center">
  <a href="https://arxiv.org/abs/2609.25247"><img src="docs/assets/button-paper.svg" width="104" height="40" alt="Paper on arXiv"></a>
  <a href="https://hyokong.github.io/segment-snap-page/"><img src="docs/assets/button-project.svg" width="152" height="40" alt="Project page"></a>
  <a href="https://huggingface.co/imsuperkong/Segment-Snap"><img src="docs/assets/button-huggingface.svg" width="166" height="40" alt="Hugging Face checkpoints"></a>
</p>

<p align="center">
  <a href="docs/assets/teaser.svg"><img src="docs/assets/teaser.svg" width="1100" alt="Segment–Snap couples movable parts, motion, and handles: dense handles guide hinge placement; joint part queries add handle proposals; standalone parts provide class context."></a>
</p>

**Handles guide motion. Parts refine handles.** One directed pass each way, with no iterative
feedback. Select the figure to inspect the full-resolution version.

[Method](docs/METHOD.md) · [Validation results](docs/RESULTS_VAL.md) · [Data preparation](docs/DATA.md) · [Checkpoint guide](checkpoints/README.md)

Segment–Snap recovers **movable parts, their motion, and the handles used to operate them** from a
3D indoor scan. A door's surface constrains its possible motion, but its handle helps identify
which side is hinged. In the other direction, a part provides context for finding and classifying
small handles. Our method makes these complementary geometric and semantic relationships explicit.

This repository contains the reference implementation, training configurations, and pretrained
checkpoints for the public Articulate3D validation experiments.

## The idea

Three independently trained predictors see the same scene and serve different roles:

| Predictor | What it produces | How it is used |
|---|---|---|
| Movable-part predictor | Part masks and rotation/translation classes | Final part segmentation, geometric motion decoding, and context for handle labels |
| Dense handle predictor | Pointwise handle probabilities | Initial handle instances, hinge-location cues, and fallback label evidence |
| Joint part-handle predictor | Its own parent parts and per-query handle masks | Additional handle proposals; its parent parts do not replace the standalone part output |

**Handles guide geometry.** A training-free decoder fits a box to each part's reliable support.
It uses a vertical-axis prior for rotations and the fitted surface normal for translations.
A nearby predicted handle selects a candidate hinge line on the opposite side of a rotating part.

**Parts support handle understanding.** The joint predictor contributes complementary handle
proposals. The standalone part predictions then help correct these proposals' motion classes,
with overlapping dense detections as fallback evidence. Dense detections themselves stay unchanged.

Each transfer is applied once: the final handle set is **not** fed back into motion decoding.
Only the geometric decoder is training-free; all three predictors are learned. See the
[method guide](docs/METHOD.md) for the dataflow, assumptions, and implementation.

## Validation highlights

Results use the released checkpoints on **42 public validation scenes**. AP values below are
percentages; gains are percentage points (pp), calculated before rounding. The models were
optimized on the 195 training scenes, while validation informed model, checkpoint, and
hyperparameter selection.

### Handle-guided part motion

| Origin rule | Part AP50 | + Axis | + Origin | + Axis & origin |
|---|---:|---:|---:|---:|
| Part centroid, without handles | 47.93 | 43.75 | 14.37 | 13.74 |
| **Predicted-handle guidance** | 47.93 | 43.75 | **43.11** | **40.98** |

Changing only hinge origins gives **+27.25 pp** in motion-gated AP50. Masks, classes, scores, and
axes are fixed in this comparison. The gain is rotation-specific: +54.49 pp on rotations and
zero on translations, whose origins are not evaluated.

### Part-informed handle detection

| Handle output | AP50 | Gain over preceding row |
|---|---:|---:|
| Dense detections | 24.63 | — |
| + Joint predictor's proposals | 29.65 | +5.01 |
| + Contextual label correction | **30.99** | +1.34 |

The proposal gain is supported by paired scene analysis and training-seed checks. The final
label-correction gain includes both part context and dense-handle fallback; its size varies with
the joint predictor's training seed. These results support proposal complementarity, not an
isolated claim that conditioning alone causes the gain.

The [results guide](docs/RESULTS_VAL.md) defines every metric, explains the controls and uncertainty,
and shows which experiments the release reproduces.

### Challenge result

Our competition entry placed **first in both evaluated outputs** of the Articulate3D challenge:
48.28 motion-gated part AP50 and 34.46 handle AP50, as recorded in the September 6, 2026
[leaderboard snapshot](https://art3d-challenge.mooo.com/web/challenges/challenge-page/1/leaderboard/).
That entry used additional engineering; these hidden-test scores are not the expected output of
the released validation recipe.

## Quick start

Run the commands below from the repository root. Validation inference needs a CUDA-capable GPU,
the processed dataset, and all three released checkpoints.

### 1. Set up the environment

```bash
git clone --recursive https://github.com/HyoKong/Segment-Snap.git
cd Segment-Snap

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt \
  timm einops open3d plyfile scikit-learn pandas plotly \
  termcolor yapf tqdm pyyaml peft huggingface_hub spconv-cu126
python -m pip install torch_scatter torch_cluster \
  --no-index --only-binary=:all: \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

export PYTHONPATH="$PWD:$PWD/third_party/volt"
```

The reference environment uses Python 3.12 and PyTorch 2.8.0 with CUDA 12.8. Match the PyG wheels
to your Python, PyTorch, and CUDA build if you choose another environment. Restore `PYTHONPATH`
when opening a new shell.

`requirements.txt` lists the project's minimal imports, **not the complete Volt/Pointcept
environment**. Likewise, `setup.sh` initializes the submodule and installs that minimal list;
it does not replace the full setup above. The current Pointcept registry and trainer import
`spconv` even though the Volt backbone does not use sparse convolutions in its forward pass.
Volt's optional `flash-attn` dependency has an in-tree PyTorch attention fallback. The other
vendored model families do not need to be installed or trained for this release.

### 2. Prepare the data

Follow the [data guide](docs/DATA.md) to obtain the organizers' processed release and create the
aligned part and handle inputs. The default layout is:

```text
data/
├── a3d/processed/     # Original processed scenes and evaluation annotations
├── pointcept_mov/    # Part inputs, including superpoints
└── pointcept_lite/   # Dense handle inputs
```

The three predictors must use the same point ordering and coordinate frame. Keep the original
processed data available: evaluation and point-order checks read it directly.

### 3. Download the checkpoints

```bash
python scripts/download_checkpoints.py --dest checkpoints
python scripts/download_checkpoints.py --dest checkpoints --verify-only
```

The three weights total approximately 4.9 GB. The [checkpoint guide](checkpoints/README.md)
documents their roles, configurations, checksums, and the separate backbone initialization needed
only for retraining.

### 4. Run the validation pipeline

```bash
DATA_ROOT=data/pointcept_mov \
LITE_ROOT=data/pointcept_lite \
ARTI3D_GT_ROOT=data/a3d/processed \
OUT=runs/reproduce_val \
  bash scripts/reproduce_val.sh
```

This runs the six-stage inference pipeline and a no-handle motion control. Predictions and
`metrics.json` files are saved under `runs/reproduce_val/`; the terminal prints a summary in
**[0, 1] AP units**, rather than the percentages used above. See
[pipeline outputs](docs/METHOD.md#running-the-pipeline) for individual stages and file formats.
Choose a different `OUT` directory to keep previous results.

The vendored evaluator also provides known-answer checks:

```bash
ARTI3D_GT_ROOT=data/a3d/processed python -m arti3d.eval.selftest
```

## Repository guide

| Location | Purpose |
|---|---|
| [`arti3d/`](arti3d/) | Predictors, geometric decoding, data preparation, and evaluation |
| [`scripts/`](scripts/) | Data conversion, checkpoint download, and inference entrypoints |
| [`configs/arti3d/`](configs/arti3d/) | Training recipes; see [training instructions](docs/METHOD.md#training) |
| [`checkpoints/`](checkpoints/) | Released configurations and checkpoint download instructions |
| [`docs/`](docs/) | Method, data, and validation guides |
| [`third_party/volt/`](third_party/volt/) | Pinned Volt/Pointcept backbone dependency |

## Citation

If this work is useful to your research, please cite our [arXiv preprint](https://arxiv.org/abs/2609.25247).
The citation is also available as [CITATION.bib](CITATION.bib).

```bibtex
@article{kong2026segmentsnap,
  title   = {Geometric and Semantic Coupling for
             Interaction Understanding in 3D Scenes},
  author  = {Kong, Hanyang and Yang, Xingyi},
  journal = {arXiv preprint arXiv:2609.25247},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.25247}
}
```

## Acknowledgments and license

Segment–Snap builds on [Volt](third_party/volt/README.md), Pointcept, and SPFormer, and uses
Articulate3D's annotations and the USDNet evaluation implementation. Our contribution is the
geometric and semantic coupling, not the underlying backbone or the benchmark.

See [LICENSE](LICENSE) for this repository's MIT license. Vendored code retains its upstream
attribution and license notices; the dataset is obtained separately under its providers' terms.
