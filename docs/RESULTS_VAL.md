# Validation results

[Overview](../README.md) · [Method](METHOD.md) · [Data](DATA.md) · [Checkpoints](../checkpoints/README.md)

The experiments ask three questions: do handles resolve hinge ambiguity, do part-associated
proposals recover additional handles, and does part context improve their motion labels?
The first two effects are the strongest evidence for coupling; the smaller label-correction gain
is reported with its training variability.

## Evaluation protocol

The reference configuration uses three released checkpoints, optimized on **195 training scenes**
and evaluated on **42 public Articulate3D validation scenes**. Validation informed model,
checkpoint, and hyperparameter selection. These are development-set results, not an untouched
estimate of generalization. Reference inference used an RTX 5070 Ti, full precision, and TF32
disabled.

All AP scores in this guide are **percentages**. Differences are **percentage points (pp)**,
computed before rounding the displayed endpoints. The evaluator and saved JSON files use
fractions in [0, 1]. Each AP is macro-averaged over rotation and translation.

| Metric | What a match must satisfy | JSON key |
|---|---|---|
| Part or handle AP50 | Instance-mask IoU greater than 0.5 | `AP50` |
| Part AP50 + axis | Mask criterion and sign-invariant axis error below 15° | `MA_ap50` |
| Part AP50 + origin | Mask criterion and the rotational origin-distance test | `MO_ap50` |
| Motion-gated part AP50 | Mask, axis, and the joint rotational origin-distance tests | `MAO_ST_ap50` |

For the joint metric, the displacement between representative origins is projected perpendicular
to each of the predicted and annotated axes; both distances must be below 0.25 m. The origin-only
diagnostic uses the annotated axis. **Translation origins are not evaluated by these metrics.**
The vendored evaluator also returns `MAO_ap50`, an alternative Euclidean-origin implementation;
it is not interchangeable with the `MAO_ST_ap50` used for the motion results here.

### Reading uncertainty

An interval `[L, U]` below is an approximate 95% confidence interval for the stated **AP difference**
in pp. It is computed by a paired leave-one-scene-out jackknife: omit each scene, recompute both
arms on the same remaining scenes, and use the full-set difference ± 1.96 jackknife standard
errors. It is not the minimum and maximum omitted-scene scores.

These intervals hold model predictions fixed and describe scene-sampling uncertainty. Training
variation is assessed separately by changing a predictor's training seed. A positive interval
does not establish that a gain will have the same size after retraining.

## Handles disambiguate hinge placement

The control fixes part masks, classes, scores, and axes, changing only the origin rule.
Both configurations are produced by the release pipeline.

| Origin rule | Part AP50 | + Axis | + Origin | + Axis & origin |
|---|---:|---:|---:|---:|
| Support centroid, no handle cue | 47.93 | 43.75 | 14.37 | 13.74 |
| **Predicted-handle guidance** | 47.93 | 43.75 | **43.11** | **40.98** |

Handle-guided selection gives **+27.25 pp** in motion-gated AP50, with a paired interval of
**[+21.86, +32.63] pp**. Since mask and axis scores are unchanged, this isolates hinge placement
rather than improved segmentation. The corresponding class breakdown is:

| Motion-gated AP50 | Rotation | Translation | Macro |
|---|---:|---:|---:|
| Centroid origins | 1.88 | 25.59 | 13.74 |
| Handle-guided origins | 56.37 | 25.59 | 40.98 |
| **Change (pp)** | **+54.49** | **0.00** | **+27.25** |

Only rotations benefit from this intervention because the metric omits the translation-origin
test. The per-rotation gain has its own paired interval, [+43.71, +65.27] pp; the translation
difference is identically zero in every omitted-scene comparison.

## Parts complement dense handle detections

The dense predictor supplies the initial handles. The joint predictor adds child proposals, and
contextual correction then changes only the new proposals' classes.

| Handle output | AP50 | Change from preceding row (pp) | Instances |
|---|---:|---:|---:|
| Dense detections | 24.63 | — | 342 |
| + Joint predictor's proposals | 29.65 | +5.01 | 4,471 |
| + Contextual label correction | **30.99** | +1.34 | 4,471 |

The **4,129 additional proposals** yield +5.01 pp, with a paired interval of
**[+1.47, +8.56] pp**. In the measured matching analysis they preserve the dense detections'
matching prefix and recover 53 additional ground-truth handles. Spatially permuted proposals and
random size-matched masks return the dense-only score, supporting genuine localization rather
than a benefit from merely appending more detections.

This control establishes the value of the **implemented proposal source**. It does not isolate
parent conditioning itself: the matched conditioning/readout study does not establish that
conditioning alone is responsible for the gain. Query-index association is used throughout the
released validation pipeline.

### Which source supplies the corrected labels?

Every row below uses the same uncorrected union, holding masks and scores fixed. The correction
sources are alternatives, not successive stages.

| Class-context source | Handle AP50 | Gain over the uncorrected union (pp) |
|---|---:|---:|
| None | 29.65 | — |
| Standalone parts only | 30.63 | +0.98 |
| Dense detections only | 30.66 | +1.01 |
| **Parts with dense-detection fallback** | **30.99** | **+1.34** |

The full rule relabels 885 child proposals and leaves dense detections unchanged. Its +1.34-pp
reference gain has a paired interval of **[+0.37, +2.31] pp**. The parts-only and dense-only gains
overlap and must not be added. The released script implements the full rule; the source-isolation
experiments belong to the paper's additional analysis.

| Handle class | Before correction | After correction | Change (pp) |
|---|---:|---:|---:|
| Rotation | 48.56 | 48.54 | −0.02 |
| Translation | 10.74 | 13.45 | +2.71 |

The net benefit is concentrated on translation handles. The small rotation-handle change is
unresolved under paired scene analysis; the rule should not be described as improving every class.

## Stability of the coupling gains

The following ranges change one predictor's training seed while keeping the other predictors
fixed. They describe **paired gains within each model draw**, not the range of absolute AP.

| Mechanism | Reference gain (pp) | Gain across training draws (pp) | Predictor varied |
|---|---:|---:|---|
| Handle-guided hinges | +27.25 | +25.27 to +27.38 | Part model, 4 draws |
| Appended child proposals | +5.01 | +3.94 to +5.01 | Joint model, 3 draws |
| Contextual label correction | +1.34 | +0.19 to +1.34 | Joint model, 3 draws |

Handle guidance and proposal augmentation remain beneficial across these checks. Label correction
is positive in the listed draws, but its repeated gains can be much smaller than the reference
result. Its benefit is observed, not established as a seed-stable property of the method.
The +1.34-pp value is a measured reference gain, not a guaranteed retraining gain.
Seed ranges and scene intervals answer different questions and are not interchangeable.

## Understanding the motion decoder

### Geometric components interact

A simple baseline uses stock query selection, uncleaned fitting support, centroid origins, and
no connectivity rescoring. It obtains 45.66 part AP50 and 12.24 motion-gated AP50. The full
configuration reaches 47.93 and 40.98, respectively.

Each component is evaluated in two contexts: added alone to the baseline, and removed from the
full configuration. The right column reports **full minus ablated** motion AP, so a positive
number is the cost of removal.

| Component | Added to the baseline (pp) | Removed from the full system (pp) |
|---|---:|---:|
| Per-query class selection | +0.13 | +0.19 |
| Largest-component fitting support | −0.11 | +2.95 |
| Handle-guided origins | +23.79 | +27.25 |
| Connectivity rescoring | +1.49 | +1.60 |

Support cleanup is useful in combination with hinge selection: it changes the box whose candidate
lines are selected by the handle. Its negative standalone result and positive removal cost show
why component effects cannot be added. Cleanup and handle guidance change geometry, not masks.
The [method guide](METHOD.md#running-the-pipeline) lists the released controls and their flags.

### Comparison with learned decoding on frozen inputs

The paper additionally compares motion heads on the same frozen part masks, classes, scores,
and features. All rows below have access to the predicted handle cue and have part AP50 47.93.
Learned-head values are ranges over three head-training seeds, not confidence intervals.

| Decoder | Motion-gated AP50 |
|---|---:|
| Continuous motion regression | 29.61–31.30 |
| Learned axis/hinge selection, box axes and vertical prior | 37.27–38.63 |
| Learned axis/hinge selection, box axes only | 37.11–38.78 |
| Fixed rule axes, learned hinge selection | 37.69–39.36 |
| **Training-free geometric decoder** | **40.98** |

The rule is effective without motion-regression training. These point estimates do **not**
establish general superiority over learned articulation: the comparison concerns frozen-feature
decoding, and paired intervals for the closest head comparisons include zero. These experimental
heads and their training harness are not included in this release.

## Remaining failure modes

The validation set contains 390 ground-truth parts: 236 rotational and 154 translational.
A coverage-first analysis assigns each part to the first applicable failure below, so these
counts form a partition rather than independent, additive error estimates.

| Outcome | Rotation | Translation | Total |
|---|---:|---:|---:|
| No predicted mask matches at IoU > 0.5 | 50 | 89 | 139 |
| Covered, but fails the axis gate | 12 | 2 | 14 |
| Passes the preceding checks, but fails the origin gate | 14 | 0 | 14 |
| Passes all checks | 160 | 63 | 223 |
| **Ground-truth total** | **236** | **154** | **390** |

Missing masks are the largest remaining source of failure, especially for translations; a hinge
selector cannot repair an undetected part. The 12 covered rotation-axis failures expose the
vertical prior's limitation on non-vertical hinges. For handles, missed small regions and
oversized masks remain localization errors that class correction cannot resolve.

This is an instance-coverage diagnostic, **not** an AP decomposition: AP also depends on false
positives, ranking, and matching. The geometric assumptions are summarized in
[Method: Assumptions and limits](METHOD.md#assumptions-and-limits).

## Reproduce the reference results

After following [setup](../README.md#quick-start), [data preparation](DATA.md), and
[checkpoint download](../checkpoints/README.md), run from the repository root:

```bash
DATA_ROOT=data/pointcept_mov \
LITE_ROOT=data/pointcept_lite \
ARTI3D_GT_ROOT=data/a3d/processed \
OUT=runs/reproduce_val \
  bash scripts/reproduce_val.sh
```

The script produces the two motion configurations and all three handle stages shown in the main
controls. It saves complete precision in each `metrics.json` and prints a rounded summary.

<details>
<summary>Full-precision reference metrics</summary>

| Output directory | Metric key | Value in saved JSON |
|---|---|---:|
| `t1_centroid/` | `MO_ap50` | 0.14373294936305364 |
| `t1_centroid/` | `MAO_ST_ap50` | 0.1373870573931177 |
| `t1/` | `AP50` | 0.47929751799455056 |
| `t1/` | `MA_ap50` | 0.43747368928511543 |
| `t1/` | `MO_ap50` | 0.431147838595354 |
| `t1/` | `MAO_ST_ap50` | 0.4098392259468343 |
| `t2_single/` | `AP50` | 0.24634765846937887 |
| `t2_union/` | `AP50` | 0.29649211039671153 |
| `t2_voted/` | `AP50` | 0.3099074679392516 |

</details>

These are reference outputs, not a promise of bit-identical behavior across GPU architectures or
software builds. Thresholded masks, floating-point reductions, and component connectivity can
make small numerical differences affect the final metrics. Keep the released precision, query
association, preprocessing, and checkpoint settings when comparing results.

The source-isolation controls, paired jackknife, multi-seed studies, learned-head comparisons,
and failure census above summarize the paper's research experiments. They are **not all rerun by
`reproduce_val.sh`**, and their dedicated research harnesses are not bundled in this repository.
The public release covers the reference inference pipeline, its no-handle control, and the
training recipes for its three predictors.

## Challenge context

Our challenge entry achieved first place for both outputs, with **48.28 motion-gated part AP50**
and **34.46 handle AP50** on the hidden test set. These are the standings recorded on
September 6, 2026; see the
[official leaderboard](https://art3d-challenge.mooo.com/web/challenges/challenge-page/1/leaderboard/).
The entry used additional engineering and differs from the released validation configuration.
There are no test-set component studies in this guide.
