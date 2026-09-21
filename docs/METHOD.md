# Method

Two tasks over the same indoor scans: **movable parts with their motion** (Track 1) and
**interactable handles** (Track 2). Learned predictors find the broad part surfaces and the small
handles; the motion of a part is then *decoded* from its geometry and its handle rather than
regressed by a learned head; and the two tasks feed each other exactly once each, in one direction
at a time. The three trained networks share an architecture family but not weights.

## Information flow

```
                 ┌─ Track-2 dense model ─► per-point probabilities ─► connected components ─► dense handle instances ─┐
                 │                                   │                                                                │
  scene ─────────┤                                   └─► handle centroids ─► hinge origins ─┐                         │ union: proposals
                 │                                                                          │                         │ appended below
                 ├─ Track-1 part model ─► part masks + classes ─► geometric motion decode ──┴─► PARTS + MOTION        │
                 │                                                        │  parts vote on the proposals' classes     ▼
                 └─ joint model (parts + child head) ─► handle proposals ─┼──────────────────────────────────────► class vote ─► HANDLES
                                                                          └───────────────────────────────────────────┘
```

`scripts/reproduce_val.sh` runs the six stages in the only order that satisfies this graph:

| stage | script | reads | writes |
|---|---|---|---|
| 1 | `infer_t2_sem.py` | the Track-2 checkpoint | per-point handle probabilities, one `<sid>_prob.npy` per scene |
| 2 | `instances_t2.py` | stage 1 | dense handle instances — **also Track 1's hinge cue** |
| 3 | `infer_t1.py --handles <stage 1>` | the Track-1 checkpoint, stage 1 | part masks, classes, scores, axes, origins |
| 4 | `infer_s2_child.py` | the joint checkpoint | handle proposals, one set per predicted part |
| 5 | `instances_t2.py --union-child <stage 4>` | stages 1 and 4 | the dense instances with the proposals appended below |
| 6 | `classvote.py --t1 <stage 3>` | stages 3 and 5 | the proposals relabelled by the parts that contain them |

The handles Track 1 consumes come from stage 2, before the union and the vote exist, so nothing
flows back: one pass in each direction, no cycle.

---

## Track 1 — parts, then motion by geometric decoding

### 1. Part segmentation

A Volt-B backbone (pretrained on ScanNet++) with an SPFormer decoder: 200 learned queries attending
over superpoints, two classes {rotation, translation}. Superpoints are a Felzenszwalb–Huttenlocher
segmentation of a 12-nearest-neighbour graph (`k_thresh` 0.005, `seg_min` 5), chosen by a sweep
for the achievable ceiling on movable parts rather than left at the library default: a superpoint
that straddles a part boundary caps what any decoder above it can recover.

**Selection is per-query argmax, not a global top-k.** Upstream flattens the (query × class) grid
and takes one top-k over it, so a confident query can occupy two output slots with both class
hypotheses over the same mask while a weaker query gets none. Taking each query's best class once
emits 6285 instances on validation against 5556 under the stock rule, and is worth +0.0054 AP50
(`arti3d/models/spformer_argmax.py`; `--topk-rule cap` restores upstream's rule).

### 2. Support before the fit

The box is fitted to the mask's **largest connected component at 5 cm**, not to the whole mask —
and **the submitted mask is not modified**. Instance masks carry no connectivity constraint, so a
handful of stray points across the room inflates the box and relocates every quantity derived from
it. Alone the cleanup does nothing (−0.0011 on the ranking column); in place, once something chooses
among the box's edges, it is worth +0.0295 (`docs/RESULTS_VAL.md`, the component ladder).

### 3. The box

PCA gives the plane normal; the in-plane axes come from the **minimum-area rectangle** over the
convex hull of the support points (rotating calipers), not from PCA — PCA's in-plane orientation
follows the point distribution rather than the shape and reproduces the annotation box far less
often (62.8 % vs 75.8 % axis coverage on ground-truth masks). The centre is the support-point mean.

### 4. The axis — a prior for rotations, geometry for translations

* A **translation** slides along the part's own **fitted plane normal**: a per-part quantity
  (2100 distinct values over the 2808 predicted translations).
* A **rotation** hinges about **a fixed vertical direction (world Z)**: a dataset-level prior, the
  same vector for all 3477 predicted rotations. On validation it passes the metric's 15° gate on
  93.6 % of matched rotations; the plane normal passes on 94.2 % of matched translations.

`--axis-rule most_vertical` selects the box axis nearest vertical per part instead. It disagrees
with the constant by more than the 15° gate on 17.9 % of predicted rotations, scores +0.002 higher
on the released model and the sign of that difference flips across training seeds, so the two rules
are not resolved; the prior's total cost is 12 ground-truth rotations whose axis is more than 15°
from vertical and which fail the axis gate for that reason.

### 5. The hinge — the handle picks a side

Four candidate lines run parallel to the axis through the box corners. Geometrically they are
**two physical sides of the part, each duplicated across the plate thickness**: within a pair the
lines are 0.08 m apart at the median, under the metric's 0.25 m origin tolerance, while the pairs
are 0.63 m apart. The rule chooses the line **farthest from the part's predicted handle** and places
the origin at the perpendicular foot of the box centroid on it — a representative point on the
hinge line, which is what the evaluator's point-to-line criterion measures. The handle is the
nearest dense handle instance (stage 2) within 0.5 m of the part; with none that close, the origin
falls back to the box centroid (the control in `reproduce_val.sh`, stage 3b, applies the fallback
to every part). Translations get the centroid: the metric ignores their origin.

On validation the four candidates contain a passing origin for 94.4 % of matched rotations and the
rule picks it on 83.4 %; the rule takes the right *side* on 88.9 % of matched rotations, and which of
the pair's two lines it takes is decided by plate thickness, which the metric does not score.

### 6. Connectivity rescoring

`score × f^γ` with `f` the mask's largest-component fraction at 5 cm and γ = 1. Fragmented masks are
usually spurious, and average precision integrates the whole ranking, so demoting them cleans it
without deleting anything. Masks, axes and origins are untouched. Worth +0.016 on the ranking
column; the optimum in γ is interior and shallow (`arti3d/geom/rescore.py`).

### Constants (Track 1)

| what | value | flag in `scripts/infer_t1.py` |
|---|---|---|
| instance selection | per-query argmax | `--topk-rule argmax` |
| smallest emitted part | 4 points | `--min-points 4` |
| cleanup radius before the fit | 0.05 m | `--snap-largest-cc 0.05` (0 disables) |
| axis rule | world Z for rotations, plane normal for translations | `--axis-rule canonical` |
| handle instances | components at 0.025 m, ≥ 3 points (as in Track 2) | `--handle-radius 0.025 --handle-min-points 3` |
| handle gate | nearest handle within 0.5 m | `--handle-max-dist 0.5` |
| origin rule | line farthest from the handle; centroid without a handle | (fixed) |
| rescoring | γ = 1, components at 0.05 m, masks over 20 000 points subsampled with a fixed seed | `--gamma 1.0 --cc-radius 0.05` |

The two hand-chosen constants sit at a measured optimum or plateau: the handle gate at 0.25 / 0.5 /
1.0 m gives 0.40855 / 0.40984 / 0.40783, the cleanup radius at 0.025 / 0.05 / 0.1 m gives
0.40841 / 0.40984 / 0.40984 (`docs/RESULTS_VAL.md`).

---

## Track 2 — handles, then two more sources of evidence

### 1. The dense model

Point-level 3-class semantic segmentation on a Volt-B backbone: {background, rotation-handle,
translation-handle}, where a handle's class is its parent part's motion type. **No queries and no
superpoints on this track, deliberately**: handles are 16–21 points, a query decoder loses badly at
that size, and superpoint pooling empties about a fifth of the handle targets outright. Training
uses a coarse-to-fine label curriculum (targets dilated to 0.10 m for the first half of training,
0.04 m until 80 %, raw labels for the last 20 %, so that the model is never evaluated on targets
fatter than the ones it last saw). The curriculum is part of the released recipe and not a claim:
in a same-seed comparison at the shipped length it moved the dense model by +0.008, inside
scene-sampling noise.

### 2. Instances

Connected components at **2.5 cm** of the foreground argmax (2.0 cm shatters handles, 3.0 cm merges
neighbours), class by majority vote, score the **mean per-point probability** — which is why stage 1
saves probabilities rather than labels. Components under **3 points** are dropped. Three is a
coverage choice, not an AP optimum: 17.9 % of interactable instances have fewer than 10 points, and
a floor of 10 scores +0.028 higher on validation by discarding 105 mostly-false detections along with
seven handles that had no other cover (`docs/RESULTS_VAL.md`). The floor acts on the dense instances
only.

### 3. Handle proposals from the joint model, appended below

The joint model (`S2` in the code, after the second training stage) has Track 1's backbone and
decoder plus a per-point **child head**: for each part query it predicts which points are that
part's handle. This is a second source of handle detections produced by a different mechanism, and
the two sources fail in different places. Fixed settings, each with its reason:

* child probability threshold 0.30; inference in fp32, because a thresholded head flips instances
  under half precision;
* a proposal's score is its **parent query's score** (measured over the whole part), not the
  child's own mean probability (a product of sigmoids over a handful of points);
* proposals are not clipped to the parent mask — the proposal exists to be tighter than the part;
* no minimum size beyond one point;
* **association by query index**: `child_prob` is emitted per query while the instance head emits
  a reordered, filtered subset, so a proposal is fetched by the query index its parent came from,
  tracked through top-k and NMS (`arti3d/models/spformer_qtrack.py`). Fetched by output position
  instead, only 7.7 % of proposals lie inside their own part (54.6 % by query index) and their class
  agreement with the matched handle falls to chance.

The proposals are **appended strictly below every dense instance of the scene** at score
`0.05 × parent score` rather than merged into the ranking: a product of sigmoids and a mean
probability are not on a common scale, so only their relative position is meaningful. Average
precision integrates the whole ranking, so tail detections that recover missed handles add area
while the head of the ranking is undisturbed. The strictly-below property is asserted on the
assembled predictions; the gain is flat for every scale in [0.05, 0.5] and the script refuses 1.0,
where a proposal could outrank a dense instance.

### 4. The class vote — Track 1's parts correct the proposals' labels

A handle's class is its parent part's motion type, so a handle inside a predicted drawer is a
translation handle whatever the dense model called it. For each proposal:

1. among Track-1 parts (score ≥ 0.1) containing ≥ 90 % of its points, take the highest-scoring; if
   that part's score is ≥ 0.3 and its class differs, adopt the part's class;
2. otherwise, if the dense instance that best contains it holds ≥ 90 % of its points and its class
   differs, adopt that class.

**Only proposals are relabelled; dense instances never are.** Masks and scores are untouched, so
the only thing that moves is which class each proposal is matched against — exactly the quantity
the rule claims to fix. On the released outputs the vote relabels 885 of 4129 proposals.

### Constants (Track 2)

| what | value | flag |
|---|---|---|
| component radius | 0.025 m | `instances_t2.py --radius 0.025` |
| smallest dense instance | 3 points | `instances_t2.py --min-points 3` |
| child probability threshold | 0.30 | `infer_s2_child.py --thr 0.30` |
| proposal association | by query index | `infer_s2_child.py --association query` |
| append scale | 0.05 × parent score | `instances_t2.py --union-scale 0.05` |
| vote: containment / part score | 0.9 / 0.3 (parts under 0.1 never vote) | `classvote.py --part-score 0.3` |

---

## The coupling, as implemented

* **Handles → parts** (stage 2 → 3): the dense model's handle instances choose the hinge line. With
  masks and axes fixed, origins at the handle-chosen line score 0.40984 on the ranking column
  against 0.13739 at the box centroid: +0.272, or ×2.98 over the no-handle baseline. The absolute
  figure is the transferable one — on ground-truth masks the same handles are worth +0.290 while
  the ratio falls to ×1.53, so the ratio mostly measures how weak the no-handle baseline is.
* **Parts → handles** (stage 3 → 6, with the proposals from stage 4): +0.050 AP50 from the
  appended proposals and +0.013 from the vote. The proposals' contribution comes from the
  association, not from conditioning the child head on the part (a matched unconditioned branch
  measured −0.003, sign undetermined); of the vote's +0.013, the share carried by the parts alone
  is +0.010.
* **Independent in effect.** Switching handles off and on while switching the vote off and on
  moves Track 1 only with the handles and Track 2 only with the vote; the interaction is exactly
  zero (the vote computed from the no-handle arm is bit-identical).
* **No second pass.** Feeding the final handle set back into stage 3 was measured at +0.007 on the
  ranking column, inside scene-sampling noise; the release keeps one pass each way.

---

## Training

The Volt-B backbone weights (`weights/volt-base-scannetpp.pth`, from the upstream Volt release)
initialise all three models; the scenes are ScanNet++ scenes, so the pretraining distribution is
the evaluation distribution. Each model trains on the 195 training scenes for 400 epochs with AdamW
(lr 3·10⁻⁴, one-cycle schedule, batch 2, mixed precision, EMA weights) through Volt's trainer, from
the configuration in `configs/arti3d/`; the three shipped `checkpoints/*/config.py` are those
configurations with the seed the run actually drew.

| model | config | loss | released checkpoint |
|---|---|---|---|
| Track-1 part model | `insseg-spformer-volt-B-s1a-long.py` | Hungarian-matched classification + mask BCE + dice, weight decay 0.1 | `epoch_14` — the 14th of 20 evaluation checkpoints (70 % of the schedule), selected post hoc on the ranking column (`checkpoints/README.md`) |
| Track-2 dense model | `semseg-volt-B-armA-long.py` | cross-entropy (class weights 0.1 / 1 / 1) + Lovász, weight decay 0.05, with the curriculum | `model_best` (the trainer's mIoU selection) |
| joint model | `insseg-s2-joint-volt-B.py` | the Track-1 losses plus the child head's (class weights 1 / 1 / 2) | `model_last` |

`configs/arti3d/` also holds three variants of the dense model that were trained for the disclosure
table in `docs/RESULTS_VAL.md` (no curriculum; background weight 0.3; a second seed). They train no
released checkpoint.

## Scope

This repository is the method, trained on the training split and evaluated on validation. Our
competition entry, built on it, placed first on both tracks of the Articulate3D challenge test set
and included additional engineering that is not part of this release.
