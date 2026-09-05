"""Track WHICH QUERY each emitted instance came from, through selection and NMS.

`predict_by_feat` returns scores, masks and labels but not the query indices it kept. Under the
stock global top-k, emitted instance *j* is query `topk_idx[j]`, and NMS then reorders what
survives — so anything that needs a per-QUERY quantity (the joint model's child head emits one
child probability map per query) cannot simply index by emitted position.

This patch is upstream's `predict_by_feat` with the query indices carried through every step that
reorders or filters, and stashed for the caller. The selection itself is untouched: same top-k,
same NMS, same three thresholds, so the emitted instances are bit-identical to the stock path and
only the extra bookkeeping is new.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

QUERY_STASH: list = []


def patch_spformer_query_tracking() -> None:
    from pointcept.models.spformer import spformer as _sp

    def predict_by_feat_tracked(self, out, sp):
        cls_preds = out["labels"][0]
        pred_masks = out["masks"][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out.get("scores", None) is not None and out["scores"] is not None:
            scores = scores * out["scores"][0].sigmoid()

        labels = (torch.arange(self.semantic_num_classes, device=scores.device)
                  .unsqueeze(0).repeat(len(cls_preds), 1).flatten(0, 1))
        scores, topk_idx = scores.flatten(0, 1).topk(self.topk_insts, sorted=False)
        labels = labels[topk_idx]
        topk_idx = torch.div(topk_idx, self.semantic_num_classes, rounding_mode="floor")
        query_idx = topk_idx                       # <- the only addition, carried from here on

        mask_pred = pred_masks[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / ((mask_pred > 0).sum(1) + 1e-6)
        scores = scores * mask_scores

        if self.nms:
            scores, labels, mask_pred_sigmoid, keep_inds = _sp.mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel="linear")
            query_idx = query_idx[keep_inds]       # NMS reorders and drops

        mask_pred = mask_pred_sigmoid[:, sp[0]] > self.sp_score_thr

        score_mask = scores > self.score_thr
        scores, labels, mask_pred = scores[score_mask], labels[score_mask], mask_pred[score_mask]
        query_idx = query_idx[score_mask]

        npoint_mask = mask_pred.sum(1) > self.npoint_thr
        scores, labels, mask_pred = scores[npoint_mask], labels[npoint_mask], mask_pred[npoint_mask]
        query_idx = query_idx[npoint_mask]

        QUERY_STASH.clear()
        QUERY_STASH.append(query_idx.detach().cpu().numpy())
        return scores, mask_pred, self.class_map[labels]

    _sp.SPFormer.predict_by_feat = predict_by_feat_tracked
