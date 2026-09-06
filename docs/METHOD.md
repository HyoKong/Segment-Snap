# Method

Two tracks over the same indoor scans: **movable parts with their motion** (Track 1) and
**interactable handles** (Track 2). They are trained separately and they feed each other at
inference, which is the part of this system worth reading about.

---

## Track 1 — parts, then motion as a discrete geometric choice

### The model
A Volt-B backbone with an SPFormer decoder: 200 learned queries attending over superpoints, two
classes {rotation, translation}. Superpoints come from a Felzenszwalb–Huttenlocher segmentation of a
kNN graph, swept for the achievable ceiling on movable parts rather than left at library defaults.

### Selection: per-query argmax, not a global top-k
Upstream flattens the (query × class) score grid and takes one global top-k over it. With two
classes that grid has 2Q entries, so a confident query can occupy two output slots with both of its
class hypotheses over the same mask while a weaker query gets none. Taking each query's best class
exactly once means no mask is ever emitted twice under different labels. Worth **+0.005 AP50**, at
no cost. `arti3d/models/spformer_argmax.py`.

### Motion: no learned motion head at all
This is the claim the track rests on. **Articulation is decoded as a discrete choice among the
part's own geometry**, not regressed:

1. **Clean before fitting.** Fit the oriented bounding box to the mask's largest connected component
   at 5 cm — *not* to the whole mask, and **without modifying the submitted mask**. Instance masks
   carry no connectivity constraint, so a handful of stray superpoints across the room inflate the
   box and relocate every quantity derived from it. About a quarter of instances are affected.
   Worth **+0.037** on the ranking column.
2. **Axis**, and the two classes are decided differently — this is worth being precise about.
   * A **translation** slides along the part's own **fitted plane normal** (the thinnest box axis):
     a per-part geometric quantity, and 2100 distinct values across 2808 predicted instances.
   * A **rotation** hinges about a **fixed vertical direction (world Z)**. That is a *dataset-level
     prior* — hinges in indoor scans are overwhelmingly vertical — and not a per-part quantity: all
     3477 predicted rotation axes are the same vector. It passes the 15° gate on ~93 % of matched
     rotations, which measures the prior, not any decoding.

   `AXIS_RULES` in `arti3d/geom/snap.py` also carries `most_vertical`, which selects the box axis
   nearest vertical per part. **It is not the released default**, and it is not equivalent: the
   angle between world Z and the per-part most-vertical box axis has a median of 3.5° but a p90 of
   30°, and 27 % of rotation instances differ by more than the 15° gate.
3. **Origin** (rotations only; the metric ignores translation origins) = the box edge, of the four
   parallel to the axis, **farthest from the part's handle**, with the origin at the perpendicular
   foot of the box centroid on that edge.
4. **Connectivity rescoring**: score × largest-component fraction. Fragmented masks are usually
   spurious, and average precision integrates the whole ranking, so demoting them cleans it without
   deleting anything.

The box fit uses a **minimum-area rectangle** (rotating calipers over the convex hull), not PCA:
PCA's in-plane orientation follows the point distribution rather than the shape and reproduces the
annotation box far less often (62.8 % vs 75.8 % axis coverage on ground-truth masks).

### The cross-track coupling
**The handle in step 3 comes from the Track 2 model.** Ablating it — origins fall back to the box
centroid — costs a factor of **2.98** on the ranking column, while leaving AP50 and the axis-gated
column *bit-identical*, because handles touch nothing but origins. That controlled comparison is in
`RESULTS_VAL.md`.

---

## Track 2 — handles, then two more sources of evidence

### The model
Point-level 3-class semantic segmentation: {background, rotation-handle, translation-handle}, where
a handle's class is its parent part's motion type. **No queries and no superpoints on this track,
deliberately.** Handles are 16–21 points; a query decoder loses badly at that size and superpoint
pooling destroys about a fifth of the handle targets outright. A coarse-to-fine curriculum dilates
the targets early in training and relaxes to raw labels by the end.

### Instances
Connected components at 2.5 cm on the foreground argmax — 2.0 shatters handles, 3.0 merges
neighbours — with the class by majority vote and the score the **mean per-point probability**. That
is why inference saves probabilities rather than labels: the argmax alone throws the scoring signal
away.

### The union: a second source, appended strictly below
The joint model — called **S2** in the code, after the second training stage — shares Track 1's backbone and adds a per-point **child head** predicting, for each
part query, which points are that part's handle. Its detections are appended **strictly below** every
incumbent instance rather than merged into the ranking: the child's score is a product of sigmoids
and the incumbent's is a mean probability, so the two are not on a common scale and only their
relative position matters. Average precision integrates the whole ranking, so tail detections are
free as long as the head is undisturbed. The gain is flat for every child scale in [0.05, 0.5] — a
plateau, not a fitted parameter — and the strictly-below property is asserted on the assembled
predictions rather than assumed. Worth **+0.044 to +0.050** on every one of five independently
trained bases.

**Association matters more than the scale does.** `child_prob` is emitted per *query*, while the
instance head emits a reordered, filtered subset — so the child must be fetched by the query index
its parent came from, tracked through top-k and NMS, not by the parent's position in the output.
Getting this wrong leaves only 7.7 % of children inside their own parent and drives class agreement
to chance. See `RESULTS_VAL.md`.

### The class vote: Track 1 corrects Track 2's labels
A handle's class is its parent part's motion type, so a handle inside a predicted drawer is a
translation handle whatever the semantic net called it. Among Track-1 parts containing ≥ 90 % of a
child detection, the highest-scoring one lends its class if its score is ≥ 0.3; otherwise the
best-containing incumbent does. **Masks and scores are untouched and incumbents are never
relabelled** — only child class labels change, which is exactly the quantity the rule claims to fix.

This is the second half of the coupling: Track 2 supplies Track 1's handles for the origin rule, and
Track 1 supplies Track 2's class prior. The dependency graph is acyclic — the handles come from
Track 2's *instances* stage, before the union and the vote — and `reproduce_val.sh` runs it in the
only order that satisfies it.

---

## Scope
This repository is the **method**, trained on the training split only and evaluated on validation.
Our competition entry, built on it, placed first on both tracks of the Articulate3D challenge test
set and included additional engineering that is not part of this release.
