# Validation results, at full precision, with the provenance of every number

All numbers below were produced **on this repository's code from the three released checkpoints**,
on a single RTX 5070 Ti, on the 42 validation scenes. The released checkpoints are train-only: they
never saw a validation scene.

Where a number came from somewhere else, it says so.

---

## Track 1 — movable parts and motion

Released configuration: Track-1 checkpoint `epoch_14`, handles from the released Track-2 member
(`armA_long`), largest-component cleanup at 5 cm, vertical-prior rotation axis / plane-normal
translation axis, handle-far origin, connectivity
rescoring at γ=1.

| | AP50 | AP50_axis | AP50_origin | **AP50_axis_origin** |
|---|---|---|---|---|
| **released configuration** | 0.47929751799455056 | 0.43747368928511543 | 0.431147838595354 | **0.4098392259468343** |
| handles from `armA_bgw03` | 0.47929751799455056 | 0.43747368928511543 | 0.42221878951236624 | 0.4011903161123294 |
| handles from `armA_r1` (the identity-gate configuration) | 0.47929751799455056 | 0.43747368928511543 | 0.4337349925277356 | 0.41495389387181164 |
| **ablation: no handles, origin = box centroid** | 0.47929751799455056 | 0.43747368928511543 | 0.14373294936305364 | **0.1373870573931177** |

**The cross-track coupling is worth ×2.98** on the ranking column (0.4098392 / 0.1373871).

**AP50 and AP50_axis are identical across all four rows, to the last digit.** Handles enter the
pipeline only through the hinge origin, so masks and axes cannot move — the table demonstrates that
rather than asserting it, which is what makes the ablation a controlled comparison rather than two
separate runs.

### Connectivity rescoring, in isolation
| γ | AP50 | AP50_axis_origin |
|---|---|---|
| 0 (no rescoring) | 0.4619851117236487 | 0.3987483782984572 |
| **1 (released)** | 0.47929751799455056 | 0.41495389387181164 |

*(both with `armA_r1` handles, the identity-gate configuration)*. The optimum is interior and
shallow — γ ∈ {2,3,4,6} give 0.4138, 0.4121, 0.4097, 0.4071 — so the result is the existence of the
rule, not the exponent.

---

## Track 2 — interactable handles

The five train-only members, each through the full released chain, **every probability field and the
child set regenerated on this machine with one code path**:

| member | single model | + child union | + class vote | union gain |
|---|---|---|---|---|
| `armA_r1` | 0.24419433753787376 | 0.28964084379094157 | 0.3037153005073384 | +0.04545 |
| **`armA_long` (RELEASED)** | **0.24634765846937887** | **0.29649211039671153** | **0.3099074679392516** | +0.05014 |
| `armA_noc2f` | 0.24298559066468342 | 0.29260264684662357 | 0.30599230759641377 | +0.04962 |
| `armA_seed2` | 0.23976661668599863 | 0.28377763255052746 | 0.29696429562048565 | +0.04401 |
| `armA_bgw03` | 0.24526803111330406 | 0.29465881092905916 | 0.3093244363393815 | +0.04939 |

`armA_long` is the best member at **every** stage.

### ⚠️ This selection is not resolved by the evidence
`armA_long` beats `armA_bgw03` by **0.00058**. `armA_r1` and `armA_seed2` are the *same
configuration* differing only in the training seed, and they are **0.00675** apart — **11.6×
larger**. Any of the top three members would be a defensible release. The table is published so a
reader can see that, rather than only the winner.

### The union is a property of the method, not of one base
It is worth **+0.044 to +0.050 on every one of five independently trained bases**.

---

## The child association

*("S2" below is the joint model — Track 1's architecture plus a per-point child head; see
[`METHOD.md`](METHOD.md).)*

`child_prob` is emitted **per query**, while the instance head emits a reordered and filtered subset.
The child must therefore be fetched by the query index its parent came from — tracked through top-k
and NMS — not by the parent's position in the output. Measured on the `armA_r1` base, from the same
regenerated probability field as every other table here:

| association | children | child ≥90 % inside **its own** parent | class == matched GT handle | union AP50 | + class vote |
|---|---|---|---|---|---|
| by position | 3317 | **7.7 %** | 54.8 % | 0.27720366232298227 | 0.2918250868800495 |
| **by query index (released)** | 4129 | **54.6 %** | 63.1 % | **0.28964084379094157** | **0.3037153005073384** |

54.8 % class agreement on a two-class problem is barely distinguishable from chance, which is what a
random parent's label gives you. On validation, fixing the association is worth **+0.0124** at the
union stage and **+0.0119** after the class vote. The released implementation uses the query-index
association; an earlier implementation of this pipeline used the positional one.

**The effect is robust to the probability field it is measured on.** Recomputed on the stored
field described below (produced on different hardware), the same comparison gives +0.01176 against
+0.01244 here — the absolute numbers shift by the cross-GPU margin, the conclusion does not.

Why a misassociated union still gains anything: a misattributed child is **still a real handle
detection in the right room** — the parent supplies only the score and the class, not the mask — so
the mask can still match a ground-truth handle. What is destroyed is the class label and the score
ordering. That is also why permuting children *across scenes* measures +0.0000 while mis-indexing
within a scene does not: permutation moves masks out of the room entirely.

---

## Reproducibility

**Against the original research code, on the same GPU**, the released implementation is:

| stage | masks | axes / origins | scores |
|---|---|---|---|
| Track 1 | bit-identical 6285/6285 | bit-identical 6285/6285 | 1222/6285 identical, max Δ 2.06e-05 |
| Track 2 semantic | probability fields **bit-identical 42/42** | — | — |
| S2 child | bit-identical 3317/3317 | — | 597/3317 identical, max Δ 5.90e-06 |

The score residuals are **not** a property of the rewrite. Two runs of the *same* code on the same
GPU differ by the same amount — Track 1: 1235/6285 identical, max Δ 2.06e-05; S2 child: 639/3317,
max Δ 2.17e-05 — so the rewrite differs from the original by no more than the original differs from
itself. The instance head's forward pass uses non-deterministic reductions; masks and geometry are
thresholded or derived and are unaffected. **Two runs of the same code give identical metrics to the
last digit in all five columns**, because a 2e-5 perturbation does not reorder a ranking.

Inference is fp32 with TF32 disabled on both the matmul and cuDNN paths. The original never set
these; the release pins them, because a policy inherited from the framework default is not a policy.

### Numbers that came from elsewhere
Two references used to gate the rewrite were produced during the competition on an **RTX 5090**:

* a Track-1 artifact produced during the competition (AP50 0.47930101442199596, AP50_axis_origin
  0.4149497069549261) — the released code reproduces it to **4×10⁻⁶**, from nine of 6285 masks
  flipping **by whole superpoints** at the per-superpoint threshold;
* the Track-2 probability fields for `armA_r1`, whose stored float16 values differ from this
  machine's at a **median of exactly one float16 ULP** (2.44e-04), moving the instance count by one
  and AP50 by 1.5e-04 at the single-model stage — and by 1.6e-03 after the union, where one fewer
  incumbent changes how 4129 appended children interleave with the rest of the ranking.

Neither feeds any number in the tables above; both are recorded because they are what the rewrite
was checked against. The only place a stored-field figure appears at all is the cross-hardware
robustness note in the association section, where it is labelled as such. **Every table number comes
from fields regenerated on this machine.**
