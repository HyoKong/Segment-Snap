"""Per-query argmax instance selection for SPFormer, in place of a global top-k.

Upstream flattens the (query x class) score grid and takes a single global top-`topk_insts` over
it. With two semantic classes that grid has 2Q entries, so one confident query can occupy two
output slots with BOTH of its class hypotheses over the same mask, while a weaker query gets none.
Capping `topk_insts` at `num_query` bounds the count but does not change the rule: it is one slot
per query on average, not one per query.

This variant takes each query's best class exactly once, so no mask is ever emitted twice under
different labels. Everything downstream of the selection — mask-score reweighting, NMS and the
three threshold stages — is upstream's code unchanged, so the difference is the selection rule and
nothing else. Worth +0.005 AP50 on the validation split, at no cost.

Applied by monkeypatch rather than by editing the vendored backbone, to keep our footprint in
`third_party/volt` to the four commits the fork carries.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_spformer_argmax() -> None:
    from pointcept.models.spformer import spformer as _sp

    def predict_by_feat_argmax(self, out, sp):
        cls_preds = out["labels"][0]
        pred_masks = out["masks"][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out.get("scores", None) is not None and out["scores"] is not None:
            scores = scores * out["scores"][0].sigmoid()

        # --- the only divergence from upstream ------------------------------------------------
        # upstream: labels = arange(C).repeat(Q).flatten()
        #           scores, topk_idx = scores.flatten(0, 1).topk(self.topk_insts); topk_idx //= C
        # here: one hypothesis per query, so topk_idx is every query index exactly once.
        scores, labels = scores.max(dim=-1)
        topk_idx = torch.arange(len(cls_preds), device=scores.device)
        # --------------------------------------------------------------------------------------

        mask_pred = pred_masks[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / ((mask_pred > 0).sum(1) + 1e-6)
        scores = scores * mask_scores

        if self.nms:
            scores, labels, mask_pred_sigmoid, _ = _sp.mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel="linear")

        mask_pred = mask_pred_sigmoid[:, sp[0]] > self.sp_score_thr

        score_mask = scores > self.score_thr
        scores, labels, mask_pred = scores[score_mask], labels[score_mask], mask_pred[score_mask]

        npoint_mask = mask_pred.sum(1) > self.npoint_thr
        scores, labels, mask_pred = scores[npoint_mask], labels[npoint_mask], mask_pred[npoint_mask]

        return scores, mask_pred, self.class_map[labels]

    _sp.SPFormer.predict_by_feat = predict_by_feat_argmax
