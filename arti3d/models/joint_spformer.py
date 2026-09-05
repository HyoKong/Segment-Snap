"""S2 — SPFormer with a point-level CHILD (handle) branch on the same queries.

Query *i* predicts a movable part (parent, superpoint resolution — the Track-1 instance model unchanged) **and** that
part's handle (child, voxel resolution). One decoder, one set of queries, one Hungarian match.

WHY THE CHILD RIDES THE PARENT, with the number that justifies it. Handles are hard to find
directly: Arm A matches 134 of 388 interactable GT (34.5%). Their parents are much easier: Track 1
matches 252 of 390 movable GT at IoU 0.5. The association was measured and found very nearly a
bijection — **0 orphans, 387 of 388 handles have exactly one parent** — so a child attached to a
matched parent is reachable for ~252 handles, about **65%** against 34.5%. Upper bound, not a
forecast, but it is the quantified reason this model exists rather than a third arm.

SAFE BY CONSTRUCTION. The child target is `points carrying interactable AND movable labels` — a pure
intersection of two existing label arrays. It imposes no parent/child convention, which matters
because a post-hoc parent merge that did impose one was measured dead. The rule was verified against
ground truth before any of this was written.

FOUR DESIGN DECISIONS, each with its reason:

1. **VOXEL resolution, not superpoint.** `check_sp_target_survival.py` measured superpoints emptying
   21.13% of interactable targets and capping ever-matchable at 33.76% — handles are ~13-21 points
   and superpoints are the wrong primitive for them. The voxel grid was measured instead:
   **0.38% emptied**, median 20 pts -> 12 voxels. So the child predicts over voxels, which is also
   exactly the substrate that was measured — the evidence and the implementation describe the same
   thing.

2. **Child logits are computed for a QUERY SUBSET, never densely.** The loss only ever reads
   `child_logits[idx_q]` — the ~9 matched queries — so a dense (200 x ~200k) tensor would be 160 MB
   per sample of pure waste, on a card already at 29.5/32.6 GiB under the Track-1 instance model. Passing the subset makes
   it ~7 MB. Inference likewise needs children only for instances that survive selection.

3. **Targets are built at ORIGIN resolution and pooled with `scatter_max`, mirroring the parent.**
   Two reasons. The parent path already resolves ids through `process_instance` on `origin_instance`,
   so mirroring it means the child never depends on a second id space agreeing with the first. And
   pooling a ~20-point handle with `scatter_mean > 0.5` empties it — the majority vote is the
   defect, `scatter_max` ("any point of this instance landed in this voxel") is the fix.

4. **The child does NOT enter the Hungarian cost in v1.** the design notes calls for a child-dice term in
   the matcher after the first 10% of training. It is deferred deliberately: leaving it out keeps
   the assignment byte-identical to the proven the Track-1 instance model, which makes this model **the Track-1 instance model plus a child head
   and nothing else** — a single-variable comparison against a baseline whose every number is on the
   books. It also avoids a fraction-keyed schedule, which is its own silent-failure surface.
   Planner pre-approved this as the first descope; it returns in v2.

THE LOSS IS SHAPED FOR TIGHTNESS, WHICH IS THE POINT OF THE HEAD. The arithmetic: a 20-point GT
handle with a 40-point prediction is **IoU 0.5 exactly** — the matching knife-edge. At handle scale
there is no tolerance for over-covering at all, and the failure census measured the pathology at population
scale: predictions containing handles whole at a median **27x** their size. A loss that is merely
unbiased with respect to size does not penalise that, so the child carries an explicit asymmetric
(Tversky) term with over-coverage weighted above under-coverage. Symmetric dice and BCE cannot
express the asymmetry; that is why the term exists rather than tuning dice's weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from pointcept.models.builder import MODELS
from pointcept.models.spformer.decoder import SPFormerDecoder
from pointcept.models.spformer.loss import SPFormerCriterion
from pointcept.models.spformer.misc import process_instance, split_offset
from pointcept.models.spformer.spformer import SPFormer
from pointcept.models.utils.structure import Point


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float, beta: float,
                 eps: float = 1.0) -> torch.Tensor:
    """1 - TP / (TP + alpha*FP + beta*FN), per instance, then averaged.

    `alpha > beta` makes a false POSITIVE voxel cost more than a false negative, i.e. it punishes
    over-coverage harder than under-coverage. At alpha = beta = 0.5 this is exactly dice, so the
    asymmetry is the entire contribution and setting them equal is a clean no-op control.

    Computed on probabilities (soft), per row, with a small `eps` so an instance whose prediction is
    empty gives 1.0 rather than a NaN — a NaN here would poison the whole batch and is easy to hit
    early in training on 20-voxel targets.
    """
    p = logits.sigmoid()
    t = target.float()
    tp = (p * t).sum(-1)
    fp = (p * (1 - t)).sum(-1)
    fn = ((1 - p) * t).sum(-1)
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


class ChildMaskHead(nn.Module):
    """Per-voxel mask logits for a SUBSET of queries: `logits[j, v] = f(query_j) . g(feat_v)`.

    Separate projections from the parent's `x_mask`/`out_norm` on purpose (the AISFormer pattern):
    a handle and the part containing it are different regions, so forcing one embedding to express
    both puts them in direct competition. The queries themselves are shared — that sharing is the
    joint model — but what is read off them is not.
    """

    def __init__(self, in_channel: int, d_model: int):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(),
                                  nn.Linear(d_model, d_model))
        self.query = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                   nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, query: torch.Tensor, vx_feat: torch.Tensor) -> torch.Tensor:
        """query (n_sel, d_model) x vx_feat (n_vox, in_channel) -> (n_sel, n_vox)."""
        return torch.einsum("nd,md->nm", self.query(query), self.feat(vx_feat))


@MODELS.register_module("ArtiJointDecoder")
class ArtiJointDecoder(SPFormerDecoder):
    """SPFormerDecoder that also exposes its final queries, so the child head can read them."""

    def __init__(self, child_in_channel: int = None, **kwargs):
        super().__init__(**kwargs)
        d_model = kwargs.get("d_model", 256)
        self.child_head = ChildMaskHead(child_in_channel or kwargs.get("in_channel", 32), d_model)

    def forward(self, input_dict):
        """Upstream's forward, with the final `query` tensor kept in the output.

        This duplicates ~15 lines of `SPFormerDecoder.forward` rather than calling it, and the
        reason is that upstream **discards** the query tensor — it returns only what
        `_forward_head` derived from it. The child head needs the queries themselves. The
        alternatives were worse: re-running the decoder stack to recover them doubles its cost, and
        editing the vendored file breaks Part H's one-line cap on `third_party/volt`. Everything
        below is upstream's control flow verbatim; the only additions are `queries` in the returned
        dict and this comment.
        """
        sp_feat = input_dict["sp_feat"]
        batch_size = len(sp_feat)

        inst_feats = [self.input_proj(x) for x in sp_feat]
        mask_feats = [self.x_mask(x) for x in sp_feat]
        query, pe = self._get_query(batch_size)

        pred_labels, pred_masks, pred_scores = [], [], []
        pred_label, pred_score, pred_mask, attn_mask = self._forward_head(query, mask_feats)
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, attn_mask, query_pe=pe)
            query = self.self_attn_layers[i](query, pe)
            query = self.ffn_layers[i](query)
            pred_label, pred_score, pred_mask, attn_mask = self._forward_head(query, mask_feats)
            pred_labels.append(pred_label)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        out = {"labels": pred_label, "masks": pred_mask, "scores": pred_score, "queries": query}
        if self.iter_pred:
            out["aux_outputs"] = [
                {"labels": a, "masks": b, "scores": c}
                for a, b, c in zip(pred_labels[:-1], pred_masks[:-1], pred_scores[:-1])
            ]
        return out


@MODELS.register_module("ArtiJointCriterion")
class ArtiJointCriterion(SPFormerCriterion):
    """Parent losses unchanged, plus the child mask loss over the parent's own assignment."""

    def __init__(self, child_weight=(1.0, 1.0, 2.0), tversky=(0.7, 0.3),
                 child_pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.child_weight = tuple(child_weight)      # (bce, dice, tversky)
        self.tversky_alpha, self.tversky_beta = tversky
        self.child_pos_weight = child_pos_weight

    def match(self, pred, target):
        """The parent assignment, exposed. Identical computation to `__call__`'s own."""
        insts = target["inst_gt"]
        cls_preds, pred_masks = pred["labels"], pred["masks"]
        pred_insts = [dict(scores=cls_preds[i], masks=pred_masks[i])
                      for i in range(len(cls_preds))]
        return [self.matcher(pred_insts[i], insts[i]) for i in range(len(insts))]

    def child_loss(self, child_logits, child_targets):
        """child_logits / child_targets: lists of (n_matched, n_vox) per batch item."""
        bce, dice, tv = [], [], []
        for logit, tgt in zip(child_logits, child_targets):
            if logit is None or logit.numel() == 0 or tgt.shape[0] == 0:
                continue
            t = tgt.float()
            # pos_weight, per batch item: handles are ~0.01% of voxels, so unweighted BCE is
            # minimised by predicting all-zero and the head would never leave that basin. Computed
            # from the batch rather than fixed, because the ratio varies by an order of magnitude
            # across scenes.
            if self.child_pos_weight is not None:
                pw = logit.new_tensor(float(self.child_pos_weight))
            else:
                pos = t.sum().clamp(min=1.0)
                pw = ((t.numel() - pos) / pos).clamp(max=1e4)
            bce.append(F.binary_cross_entropy_with_logits(logit, t, pos_weight=pw))
            # dice per instance (size-normalised by construction)
            p = logit.sigmoid()
            num = 2 * (p * t).sum(-1) + 1
            den = p.sum(-1) + t.sum(-1) + 1
            dice.append((1 - num / den).mean())
            tv.append(tversky_loss(logit, t, self.tversky_alpha, self.tversky_beta))
        z = None
        if not bce:
            return None
        z = (self.child_weight[0] * torch.stack(bce).mean()
             + self.child_weight[1] * torch.stack(dice).mean()
             + self.child_weight[2] * torch.stack(tv).mean())
        return z


@MODELS.register_module("ArtiJointSPFormer-v1")
class ArtiJointSPFormer(SPFormer):
    """the Track-1 instance model's SPFormer + the child branch. Parent behaviour is untouched."""

    #: Emit child probabilities from the eval path. **OFF by default, and that is deliberate.**
    #: `forward` runs the eval branch during the TRAINER's validation pass too, ten times over a
    #: run, and the trainer has no use for child masks -- it scores parent AP50. Emitting them there
    #: would allocate a (num_query x ~200k) float tensor per scene, ~160 MB plus intermediates, at
    #: the one moment when training state is still resident. That is wasted compute at best and an
    #: OOM in hour four at worst. `scripts/s2_infer.py` sets this True; nothing else does.
    emit_children: bool = False

    #: What the child branch is trained to produce. **"handle" is v1 and is the default**, so no
    #: existing run's behaviour can change — including a resume of one that started before this
    #: parameter existed.
    #:
    #:   "handle"     child(M) = points of instance M that also carry an interactable label.
    #:                A DETECTOR of small parts inside a big one. Shipped in the Track 2 union.
    #:   "full_part"  child(M) = ALL points of instance M. An EXTENT COMPLETER: same machinery,
    #:                same matched-query targets, but the head is asked to finish the parent's own
    #:                mask instead of to find something inside it. Motivated by the 
    #:                census: 80 movable GT are found but under-covered, worth +0.2610 on the
    #:                ranking column — the largest repairable family on Track 1, and every cheap
    #:                path to it (merge, split, selective extent) is already tombstoned.
    CHILD_TARGETS = ("handle", "full_part")

    def __init__(self, child_loss_weight: float = 1.0, freeze_except: str = None,
                 child_target: str = "handle", **kwargs):
        super().__init__(**kwargs)
        self.child_loss_weight = float(child_loss_weight)
        assert child_target in self.CHILD_TARGETS, (
            f"child_target={child_target!r} is not one of {self.CHILD_TARGETS} — a typo here would "
            "silently train the v1 target and produce a run whose config claims otherwise")
        self.child_target = child_target
        # v2b: train ONLY the child branch, with the parent frozen. Default None = no freezing, so
        # this cannot change any existing run's behaviour -- including a RESUME of a run that
        # started before this parameter existed, which is the provenance hazard of editing model
        # code while a job trains. (Editing the .py itself is safe: Python compiles a module at
        # import and never re-reads the source, unlike bash reading a script incrementally -- .)
        #
        # WHY IT LIVES IN __init__ RATHER THAN A HOOK. Pointcept builds the optimizer from
        # `model.parameters()` during trainer construction, and hooks fire at before_train, i.e.
        # AFTER. Freezing here guarantees the optimizer never sees a trainable parent. (AdamW would
        # skip them anyway once their grads are None, so this is belt-and-braces rather than the
        # only defence -- but "the optimizer never had them" is a property that can be asserted,
        # and "AdamW happens to skip them" is a behaviour that has to be trusted.)
        #
        # THE MOTIVATION IS THE MARGIN, not elegance: Track 1 leads by +0.0785 and Track 2 by
        # +0.0055, and v1 measured the joint model costing the parent ~0.06 AP50. Freezing makes
        # that cost ZERO BY CONSTRUCTION instead of small by tuning.
        self.freeze_except = freeze_except
        if freeze_except:
            n_frozen = n_train = 0
            for name, p in self.named_parameters():
                if freeze_except in name:
                    n_train += p.numel()
                else:
                    p.requires_grad_(False)
                    n_frozen += p.numel()
            assert n_train > 0, (
                f"freeze_except={freeze_except!r} matched NO parameters — everything is frozen and "
                "this run would do nothing while reporting a falling loss. Check the name.")
            print(f"[freeze] trainable {n_train:,} params matching {freeze_except!r} · "
                  f"frozen {n_frozen:,}")

    # ------------------------------------------------------------------ child target construction
    @torch.no_grad()
    def prepare_child_target(self, input_dict, inv):
        """Per batch item: (n_inst, n_vox) bool, rows ordered like `prepare_target`'s masks.

        Built at ORIGIN resolution then pooled with `scatter_max`, mirroring the parent path so the
        two never depend on separate id spaces agreeing:

            child(M) = origin points with (origin_child_instance is foreground) AND
                       (processed origin_instance == M)        <- the child rule, imposes nothing

        `scatter_max` and NOT `scatter_mean > 0.5`. A majority vote over a ~20-point handle
        empties it before training starts, which is the exact failure that made superpoints unusable
        here (21.13% emptied). `scatter_max` asks "did ANY point of this instance land in this
        voxel", which is the right question for a target.
        """
        pt_offset = input_dict["origin_offset"].int()
        pt_ins = split_offset(input_dict["origin_instance"], pt_offset)
        pt_sem = split_offset(input_dict["origin_segment"], pt_offset)
        pt_child = split_offset(input_dict["origin_child_instance"], pt_offset)

        out = []
        for p_ins, p_cls, p_child, _inv in zip(pt_ins, pt_sem, pt_child, inv):
            p_ins = process_instance(p_ins.clone(), p_cls.clone(),
                                     self.segment_ignore_index, self.instance_ignore_index)
            insts = p_ins.unique()
            insts = insts[insts != self.instance_ignore_index]
            n_vox = int(_inv.max()) + 1 if _inv.numel() else 0
            if len(insts) == 0 or n_vox == 0:
                out.append(p_ins.new_zeros((0, max(n_vox, 0)), dtype=torch.bool))
                continue
            # "handle": intersect with the interactable labels (the child rule).
            # "full_part": no intersection — the target IS the parent instance, so the head learns
            # to complete an extent rather than to find a part inside one. The two differ by exactly
            # this mask and nothing else in the pipeline, which is what makes the comparison single-
            # variable.
            child_fg = (p_child > 0 if self.child_target == "handle"
                        else torch.ones_like(p_child, dtype=torch.bool))
            rows = []
            for inst_id in insts:
                m = (p_ins == inst_id) & child_fg        # the intersection ("handle" only)
                pooled = m.new_zeros(n_vox, dtype=torch.bool)
                if m.any():
                    idx = _inv[m]
                    pooled[idx] = True                   # == scatter_max over a boolean
                rows.append(pooled)
            out.append(torch.stack(rows))
        return out

    # ------------------------------------------------------------------------------------ forward
    def forward(self, input_dict):
        vx_offset = input_dict["offset"].int()
        pt_offset = input_dict["origin_offset"].int()

        feats = self.backbone(Point(input_dict))
        input_dict["feats"] = feats

        inv = split_offset(input_dict["inverse"], pt_offset)
        sp_raw = split_offset(input_dict["superpoint"], pt_offset)
        sp = [torch.unique(_sp, return_inverse=True)[1] for _sp in sp_raw]

        vx_feat = split_offset(feats, vx_offset)
        vx_coord = split_offset(input_dict["coord"], vx_offset)
        vx_grid_coord = split_offset(input_dict["grid_coord"], vx_offset)
        input_dict["sp_feat"] = [scatter_mean(_f[_i], _s, dim=0)
                                 for _f, _i, _s in zip(vx_feat, inv, sp)]
        input_dict["sp_coord"] = [scatter_mean(torch.floor(_c[_i] * 50), _s, dim=0)
                                  for _c, _i, _s in zip(vx_coord, inv, sp)]
        input_dict["sp_grid_coord"] = [scatter_mean(_c[_i].float(), _s, dim=0)
                                       for _c, _i, _s in zip(vx_grid_coord, inv, sp)]
        input_dict["sp"] = sp

        out = self.decoder(input_dict)

        return_dict = {}
        if self.criterion is not None and "origin_segment" in input_dict:
            target = self.prepare_target(input_dict)
            return_dict.update(self.criterion(out, target))
            child_ready = (hasattr(self.decoder, "child_head")
                           and hasattr(self.criterion, "child_loss"))
            assert child_ready or "origin_child_instance" not in input_dict, (
                "the dataset supplies `origin_child_instance` but the model has no child branch — "
                "use decoder=ArtiJointDecoder and criterion=ArtiJointCriterion, or the child "
                "targets are silently ignored and this trains as plain the Track-1 instance model under an S2 run-id.")
            if child_ready:
                assert "origin_child_instance" in input_dict, (
                    "the model has a child branch but the batch carries no "
                    "`origin_child_instance` — the config needs ArtiJointDataset plus a "
                    "Copy(child_instance -> origin_child_instance) BEFORE GridSample, and both "
                    "keys listed in Update(index_valid_keys) and Collect.")
                child_tgt = self.prepare_child_target(input_dict, inv)
                indices = self.criterion.match(out, target)
                logits, tgts = [], []
                for b, (idx_q, idx_gt) in enumerate(indices):
                    if child_tgt[b].shape[0] == 0 or len(idx_q) == 0:
                        continue
                    # ONLY the matched queries — see the module docstring. Dense (200 x ~200k)
                    # logits would be 160 MB/sample of tensor the loss never reads.
                    q = out["queries"][b][idx_q]
                    logits.append(self.decoder.child_head(q, vx_feat[b]))
                    tgts.append(child_tgt[b][idx_gt])
                cl = self.criterion.child_loss(logits, tgts) if logits else None
                if cl is not None:
                    return_dict["child_loss"] = cl
                    return_dict["loss"] = return_dict["loss"] + self.child_loss_weight * cl
        else:
            return_dict["loss"] = torch.tensor(0.0, device=feats.device,
                                               requires_grad=self.training)

        if not self.training:
            return_dict = self.prediction(out, return_dict, sp)
            if self.emit_children and hasattr(self.decoder, "child_head"):
                return_dict.update(self.predict_children(out, vx_feat, inv))
        return return_dict

    # ---------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_children(self, out, vx_feat, inv):
        """Per-query child probabilities, expanded to ORIGIN point resolution.

        Returns `child_prob` (n_query, n_origin_points) float32 for batch item 0 — inference runs
        one scene at a time, as the parent path already assumes.

        WHY PER-QUERY AND NOT PER-EMITTED-INSTANCE. `predict_by_feat` does not return the query
        indices it kept: it flattens (query, class) before its topk, then `mask_matrix_nms` reorders
        what survives. Reconstructing "which query is emitted instance k" through that is exactly
        the kind of index bookkeeping that fails silently. Under `--topk-rule argmax` — the rule the
        pipeline of record actually ships — each query contributes exactly one candidate, so
        candidate j IS query j and the caller can index this array directly. Emitting all queries
        here keeps that mapping the caller's explicit business rather than a hidden assumption.

        EXPANSION IS BY `inverse`, NOT INTERPOLATION. `inv` maps each origin point to its voxel, so
        an origin point simply inherits its voxel's probability. That is the same path the parent
        masks take (spformer.py:185 expands superpoint masks through the voxel cloud), so parent and
        child masks are commensurable — which matters, because a child mask is scored against the
        challenge cloud's own row order.

        NO THRESHOLD IS APPLIED HERE, deliberately. The binarisation cut is the one child-side knob
        the design notes explicitly wants calibrated post-hoc on val — the natural analogue of the cc
        rescoring that worked on Track 1 — so it belongs to the caller, not baked in. Returning
        probabilities also lets the caller measure the emitted-size-to-GT ratio across thresholds,
        which the design names as the leading indicator: a median much above ~1.5 caps achievable
        AP50 no matter how good the localisation looks.
        """
        q = out["queries"][0]                                  # (n_query, d_model)
        logits = self.decoder.child_head(q, vx_feat[0])        # (n_query, n_vox)
        prob = logits.sigmoid()
        return {"child_prob": prob[:, inv[0]].float()}         # -> (n_query, n_origin_points)
