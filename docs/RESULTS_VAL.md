# Validation results, at full precision, with the provenance of every number

**Basis.** The released configuration — the three train-only checkpoints in
[`checkpoints/README.md`](../checkpoints/README.md) through the six stages of
`scripts/reproduce_val.sh` — on the 42 validation scenes (390 ground-truth parts: 236 rotations,
154 translations), one RTX 5070 Ti, fp32 with TF32 disabled. Every number below comes from
probability fields and predictions regenerated on this machine with this code. Rows marked **R** are
printed by `reproduce_val.sh`; the other rows are the same scripts with the stated flags. Bracketed
intervals are 95 % intervals of a *difference* from a paired leave-one-scene-out jackknife over the
42 scenes (the interval script is not part of this release). Per-class numbers are the evaluator's
own class averages over the same outputs; the metric is their macro average.

Metric names: `AP50` is mask AP at IoU 0.5; `AP50_axis` additionally requires the axis within 15°;
`AP50_origin` the origin within 0.25 m of the ground-truth hinge line; **`AP50_axis_origin`**
requires both and is the ranking column (`MAO_ST_ap50` in the code). Translation instances have no
origin test.

---

## Track 1 — movable parts and motion

### The released decode and its fixed-mask control

| configuration | AP50 | AP50_axis | AP50_origin | **AP50_axis_origin** | |
|---|---:|---:|---:|---:|---|
| **released** — origins from the predicted handles | 0.47929751799455056 | 0.43747368928511543 | 0.431147838595354 | **0.4098392259468343** | R |
| control — origins at the box centroid (`infer_t1.py` without `--handles`) | 0.47929751799455056 | 0.43747368928511543 | 0.14373294936305364 | 0.1373870573931177 | R |

AP50 and the axis-gated column agree to the last digit, so the comparison changes only origins:
**+0.27245 on the ranking column, 95 % [+0.2186, +0.3263]**, no single scene carrying the sign. It
is a factor of 2.98 over the no-handle baseline; the absolute figure is the one that transfers (on
ground-truth masks the same handles give 0.54158 → 0.83147, +0.290 at ×1.53).

Handles from a differently trained dense model move the number by less than a hundredth
(0.40119 and 0.41495 from two other members of the training family): the gain is in having handles,
not in which model produced them.

### The same comparison per class

| | rotation | translation |
|---|---:|---:|
| AP50 (both rows) | 0.69650 | 0.26209 |
| AP50_axis (both rows) | 0.61900 | 0.25594 |
| AP50_origin, released / centroid | 0.60020 / 0.02537 | 0.26209 / 0.26209 |
| **AP50_axis_origin**, released / centroid | **0.56374 / 0.01883** | 0.25594 / 0.25594 |

The evaluator ignores the origin of a translation, so the coupling's +0.27245 is **+0.5449 on
rotations** (95 % [+0.437, +0.653]) and **exactly zero on translations** in every leave-one-out fit.
Every origin-side number in this document is a rotation-side number at half size.

### The component ladder — every component measured at both ends

The pipeline is not additive: the cleanup moves the box, the box moves the candidate edges, and
edges only matter once a handle chooses among them. So each component is measured twice, added to
the naive pipeline and dropped from the released one. Every arm is `scripts/infer_t1.py` on the
released checkpoint with the flags shown; the naive pipeline is upstream's selection, no cleanup,
centroid origins, no rescoring.

| arm | flags | instances | AP50 | AP50_axis_origin |
|---|---|---:|---:|---:|
| naive | `--topk-rule cap --snap-largest-cc 0 --gamma 0` | 5556 | 0.4565821868864147 | 0.12240843946412548 |
| naive + selection rule | `--snap-largest-cc 0 --gamma 0` | 6285 | 0.4619851117236487 | 0.12369270931045799 |
| naive + cleanup | `--topk-rule cap --gamma 0` | 5556 | 0.4565821868864147 | 0.12127528556596077 |
| naive + handles | `--topk-rule cap --snap-largest-cc 0 --gamma 0 --handles …` | 5556 | 0.4565821868864147 | 0.36032360431578153 |
| naive + rescoring | `--topk-rule cap --snap-largest-cc 0` | 5556 | 0.4739021477843395 | 0.1372983194741269 |
| **released** | `--handles …` (all defaults) | 6285 | **0.47929751799455056** | **0.4098392259468343** |
| released − selection rule | `--topk-rule cap --handles …` | 5556 | 0.4739021477843395 | 0.40795340316077444 |
| released − cleanup | `--snap-largest-cc 0 --handles …` | 6285 | 0.47929751799455056 | 0.38032246725540053 |
| released − handles | (no `--handles`) | 6285 | 0.47929751799455056 | 0.1373870573931177 |
| released − rescoring | `--gamma 0 --handles …` | 6285 | 0.4619851117236487 | 0.393859672655914 |

| component | added to naive | dropped from released |
|---|---:|---:|
| per-query argmax selection | +0.0013 | +0.0019 |
| largest-component cleanup before the fit | −0.0011 | +0.0295 |
| handle-guided origins | +0.2379 | +0.2725 |
| connectivity rescoring | +0.0149 | +0.0160 |
| the whole decode (naive → released) | | +0.2874, 95 % [+0.2212, +0.3537] |

The decode is worth +0.287 on the ranking column and +0.023 on AP50: almost all of this pipeline is
motion decoding, which a mask-only metric cannot see. Cleanup and handles touch only geometry and
move AP50 by exactly zero at both ends. The naive and released arms cover 248 ground-truth parts
between them with bit-identical covering masks on 244, so AP50 sees the selection rule and the
rescoring reorder the same instances, while the ranking column sees the origin (different on 184
of the 248).

### The selection rule alone

`--topk-rule argmax` (released) against `--topk-rule cap` (upstream's global top-k), everything else
released: AP50 +0.0054 [+0.0005, +0.0103], separated; ranking column +0.0019 [−0.0008, +0.0045],
not separated. The rule is kept because it is free and its mask-level gain is real.

### Sensitivity of the two hand-chosen constants

AP50 is 0.47929751799455056 in every cell; only the ranking column moves.

| handle gate (`--handle-max-dist`) | 0.25 m | **0.5 m** | 1.0 m |
|---|---:|---:|---:|
| AP50_axis_origin | 0.40855 | **0.40984** | 0.40783 |

| cleanup radius (`--snap-largest-cc`) | 0.025 m | **0.05 m** | 0.1 m |
|---|---:|---:|---:|
| AP50_axis_origin | 0.40841 | **0.40984** | 0.40984 |

The alternative rotation-axis rule `--axis-rule most_vertical` scores 0.41202 (+0.00218) on the
released model; across training seeds the sign of that difference flips, so the two rules are not
resolved and the release keeps the vertical prior.

### Where the released configuration fails

Of the 390 ground-truth parts, 139 are not covered by any prediction at IoU 0.5 (50 rotations, 89
translations — 58 % of translations against 21 % of rotations), 14 fail only the axis gate (12
rotations with a non-vertical hinge, 2 translations), 14 fail only the origin gate (all rotations),
and 223 (57.2 %) are scored. Coverage, not motion, is the translation story; the origin side is
close to done — the best of the four candidate lines would add at most +0.027.

---

## Track 2 — interactable handles

### The chain

| stage | AP50 | instances | |
|---|---:|---:|---|
| dense semantic model (`instances_t2.py`) | 0.24634765846937887 | 342 | R |
| + proposals from the joint model appended below (`--union-child`) | 0.29649211039671153 | 4471 | R |
| + class vote from Track 1's parts (`classvote.py`) | **0.3099074679392516** | 4471 | R |

**The appended proposals: +0.05014, 95 % [+0.0147, +0.0856]**, no carrier scene. The same 4129
proposals with their locations permuted across scenes, or replaced by random blobs of the same
sizes, return the no-proposal number 0.24634765846937887 to every digit: a detection that recovers
no ground truth is a false positive below every true positive and adds no area. Fixing the
association (below) is the joint model's contribution; conditioning its child head on the part was
measured at −0.003 in a matched unconditioned comparison, sign undetermined.

**The class vote: +0.01342, 95 % [+0.0037, +0.0231]**, 885 of 4129 proposals relabelled. Per class
it is +0.02707 on translation handles (0.10740 → 0.13447, 95 % [+0.008, +0.046]) and −0.00024 on
rotation handles (0.48559 → 0.48535, [−0.0015, +0.0010]): its job is fixing labels on drawer pulls.
With the parts as the only voting source the gain is +0.0098; the dense instances alone give
+0.0102; the two overlap. On the 1463 proposals the evaluator can score, the vote breaks no correct
label, fixes 453 of 527 wrong ones and leaves 74 unreached; it also relabels 432 proposals that
match no ground truth, which the score cannot see. Across three training seeds of the joint model
the append gain stays at +0.039 to +0.050 while the vote gain ranges +0.002 to +0.013, so the vote
is reported as observed, not claimed as a property of the method.

### The dense-instance floor, a choice with its price

`--min-points 3` is chosen for coverage: 17.9 % of interactable instances have fewer than 10 points.
`--min-points 10` scores 0.3246549753004043 on the union (+0.0282, 95 % [−0.0035, +0.0598], not
separated) by removing 105 of the 342 dense instances — 96 covering no ground truth, 9 covering
one — and with them the only cover of 7 handles. The release keeps 3.

### The released dense model among its training family

Five dense models were trained on the training split; all five run through the released chain with
the same proposals and the same parts (`configs/arti3d/`: the base recipe, its 400-epoch variant,
no curriculum, background weight 0.3, a second seed of the base):

| member | dense model | + proposals | + class vote |
|---|---:|---:|---:|
| `armA_r1` (base, 200 epochs) | 0.24419433753787376 | 0.28964084379094157 | 0.3037153005073384 |
| **`armA_long` (400 epochs) — released** | **0.24634765846937887** | **0.29649211039671153** | **0.3099074679392516** |
| `armA_noc2f` | 0.24298559066468342 | 0.29260264684662357 | 0.30599230759641377 |
| `armA_seed2` | 0.23976661668599863 | 0.28377763255052746 | 0.29696429562048565 |
| `armA_bgw03` | 0.24526803111330406 | 0.29465881092905916 | 0.3093244363393815 |

The released member leads at every stage, by 0.00058 over the runner-up after the vote — while
`armA_r1` and `armA_seed2`, the same configuration differing only in the seed, are 0.00675 apart.
The selection is therefore not resolved by the evidence, and any of the top three would be a
defensible release; the table is published so a reader can see that. The append gain is +0.044 to
+0.050 on every member.

### The proposal association

`child_prob` is emitted per query while the instance head emits a reordered, filtered subset, so a
proposal must be fetched by the query index its parent came from. Measured on the `armA_r1` field
(`infer_s2_child.py --association position` reproduces the alternative):

| association | proposals | ≥ 90 % inside its own part | class = matched handle's | + proposals AP50 | + class vote |
|---|---:|---:|---:|---:|---:|
| by output position | 3317 | 7.7 % | 54.8 % | 0.27720366232298227 | 0.2918250868800495 |
| **by query index (released)** | 4129 | 54.6 % | 63.1 % | **0.28964084379094157** | **0.3037153005073384** |

+0.0124 at the union stage and +0.0119 after the vote. Of the 1691 proposal masks the two share,
1688 are scored differently (a proposal inherits its parent's score), so the association is a
re-ranking rather than a re-detection; a misassociated proposal is still a handle in the right
room, which is why the positional union still gains. All numbers in this repository use the
query-index association; our competition entry used the positional one.

---

## Reproducibility

**Against the original research code, on the same GPU:**

| stage | masks | axes / origins | scores |
|---|---|---|---|
| Track 1 | bit-identical 6285 / 6285 | bit-identical 6285 / 6285 | 1222 / 6285 identical, max Δ 2.06·10⁻⁵ |
| Track 2 dense | probability fields bit-identical 42 / 42 | — | — |
| joint model proposals | bit-identical 3317 / 3317 | — | 597 / 3317 identical, max Δ 5.90·10⁻⁶ |

The score residuals are not a property of the rewrite: two runs of the *same* code on the same GPU
differ by the same amount (Track 1 max Δ 2.06·10⁻⁵; proposals 2.17·10⁻⁵) because the instance head
uses non-deterministic reductions, and a perturbation of that size reorders no ranking — two runs
give identical metrics to the last digit in every column.

**Across hardware.** Against a Track-1 artifact produced during the competition on an RTX 5090
(AP50 0.47930101442199596, AP50_axis_origin 0.4149497069549261 with that artifact's handle field),
this code reproduces the ranking column to 4·10⁻⁶: nine of 6285 masks flip by whole superpoints at
the per-superpoint threshold. The stored float16 probability fields from that machine differ from
this machine's at a median of exactly one float16 ULP (2.44·10⁻⁴), which moves the dense-instance
count by one and AP50 by 1.5·10⁻⁴ before the union and 1.6·10⁻³ after it. Neither artifact feeds any
table above.

**Runtime** (batch 1, median over validation scenes, `torch.cuda.synchronize()` around every
measurement): dense model 0.178 s per scene at 4.53 GiB peak; part model 0.225 s at 4.60 GiB, of
which the query decoder is 0.014 s; the geometric motion decode 0.391 s on the CPU with no GPU at
all. The slowest stage is per-instance Python with a KD-tree per mask — an implementation fact,
and the obvious place to spend effort if latency ever matters.

## Scope

Every number here is a validation result of the released, train-only configuration. Our
competition entry, built on this method, placed first on both tracks of the Articulate3D challenge
test set; it included additional engineering that is not part of this release.
