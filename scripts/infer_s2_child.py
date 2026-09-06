"""S2 child-head inference: a second, independent set of handle detections.

    PYTHONPATH=.:third_party/volt python scripts/infer_s2_child.py \
        --run <run dir> --ckpt last --split validation --out runs/s2/child_validation.pkl

The joint model shares Track 1's backbone and decoder and adds a per-point CHILD head: for each
part query it predicts which points are that part's handle. That makes it a second source for
Track 2, produced by a different mechanism from the semantic net, and the two sources fail in
different places — which is the whole reason the union is worth anything.

FIXED SETTINGS, and why each is what it is:

  * threshold 0.30 on the child probability. The child head is thresholded by construction, so
    inference is fp32: half precision flips instances at a threshold, measured.
  * score = the PARENT query's score. The child's own mean probability is a product of sigmoids
    over a handful of points and is a far noisier estimator; the parent's score is measured over
    the whole part. Downstream the union multiplies this by a small constant, so only the ordering
    within the child set matters, and the parent's score orders it better.
  * masks are NOT clipped to the parent's mask. Clipping caps the child's quality at the parent's,
    and the child exists precisely to be tighter than the parent.
  * `min_points = 1`. Handles are ~20 points; the usual instance-head default of 100 would delete
    the entire population this head is for.

SELECTION AND ASSOCIATION ARE SEPARATE THINGS, and this script keeps them separate.

  * SELECTION is the model's own stock top-k over the (query x class) grid followed by NMS — the
    path the joint model was trained and evaluated under. Track 1's per-query argmax is NOT applied
    here: it changes which parents are emitted, and measured that is worse (3430 children scoring
    0.27461 against 3317 scoring 0.27625 through the same union).
  * ASSOCIATION is which row of `child_prob` belongs to which emitted parent. `child_prob` is
    emitted per QUERY, while selection emits a reordered, filtered subset — so the row is fetched
    by the query index the parent came from, tracked through top-k and NMS
    (`arti3d/models/spformer_qtrack.py`). `--association position` takes row j for emitted instance
    j instead. That is an earlier implementation of this pipeline, retained only so its behaviour
    can be reproduced; it puts 7.7% of children inside their own parent against 54.6% for the
    default.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch

from arti3d.io import md5, resolve_ckpt

CHILD_THR = 0.30
MIN_POINTS = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--split", default="validation", choices=("train", "validation"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--child-root", default=None,
                    help="root holding the interactable per-point labels the joint dataset reads "
                         "(default: the training config's own value)")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--thr", type=float, default=CHILD_THR)
    ap.add_argument("--association", default="query", choices=("position", "query"),
                    help="how an emitted parent is matched to a row of child_prob. child_prob is "
                         "emitted PER QUERY. 'query' (default) takes the row of the query the "
                         "instance actually came from, tracked through top-k and NMS. 'position' "
                         "takes row j for emitted instance j, an earlier implementation kept only "
                         "so its behaviour can be reproduced. Measured on validation: "
                         "'position' puts 7.7% of children inside their own parent, 'query' 54.6%, "
                         "and the union scores 0.27625 vs 0.28801.")
    ap.add_argument("--dump-parents", action="store_true",
                    help="also store each parent's mask, for alignment diagnostics")
    ap.add_argument("--out", required=True, help="output pkl of child predictions")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    from pointcept.datasets import build_dataset, point_collate_fn
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    import arti3d.datasets.insseg      # noqa: F401
    import arti3d.datasets.transforms  # noqa: F401
    import arti3d.datasets.joint       # noqa: F401
    import arti3d.models.joint_spformer  # noqa: F401
    from arti3d.models.spformer_qtrack import QUERY_STASH, patch_spformer_query_tracking

    if a.association == "query":
        patch_spformer_query_tracking()

    #: The per-query argmax selection used on Track 1 is deliberately NOT applied here. It would
    #: make "candidate j is query j" true, but it also CHANGES WHICH PARENTS ARE EMITTED, and
    #: measured that is worse (3430 children scoring 0.27461 against 3317 scoring 0.27625). The
    #: association is fixed instead, by tracking query indices through the stock selection — which
    #: separates the two things argmax was conflating and keeps the better selection.

    cfg = Config.fromfile(os.path.join(a.run, "config.py"))
    data_root = a.data_root or cfg.data_root
    dcfg = dict(cfg.data.val)
    dcfg["split"] = a.split
    dcfg["data_root"] = data_root
    if a.child_root:
        dcfg["child_root"] = a.child_root
    ds = build_dataset(dcfg)

    ckpt_path = resolve_ckpt(a.run, str(a.ckpt))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg.model).to(a.device).eval()
    #: the child branch is silent in the trainer's eval path so validation does not allocate child
    #: logits it never reads; this script is the consumer, so this script turns it on.
    model.emit_children = True
    sd = ck.get("ema_state_dict") or ck["state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith("decoder.child_head")], (
        f"this checkpoint has no child head, so it is not a joint model: {missing[:3]}")
    print(f"checkpoint {ckpt_path}  epoch {ck.get('epoch')}  md5 {md5(ckpt_path)}  "
          f"child head present (missing {len(missing)}, unexpected {len(unexpected)})")

    preds, n_inst, sizes = {}, 0, []
    for i in range(len(ds.data_list)):
        batch = point_collate_fn([ds[i]])
        batch = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        sid = ds.get_data_name(i)
        cprob = out["child_prob"].cpu().numpy()
        pcls, pscore, pmask = out["pred_classes"], out["pred_scores"], out["pred_masks"]
        N = cprob.shape[1]
        qidx = QUERY_STASH[0] if a.association == "query" else np.arange(len(pcls))
        assert len(qidx) == len(pcls), (sid, len(qidx), len(pcls))

        cols, cs, ss, par = [], [], [], []
        for j in range(len(pcls)):
            m = cprob[qidx[j]] > a.thr
            n = int(m.sum())
            if n < MIN_POINTS:
                continue
            cols.append(m)
            cs.append(int(pcls[j]))
            ss.append(float(pscore[j]))          # the PARENT's score, see the module docstring
            if a.dump_parents:
                par.append(np.asarray(pmask[j], bool))
            sizes.append(n)
        preds[sid] = {
            "pred_masks": np.stack(cols, 1) if cols else np.zeros((N, 0), bool),
            "pred_classes": np.array(cs, dtype=np.int64),
            "pred_scores": np.array(ss, dtype=np.float64),
            **({"parent_masks": np.stack(par, 1) if par else np.zeros((N, 0), bool)}
               if a.dump_parents else {})}
        n_inst += len(cs)
        del out
        print(f"  [{i+1}/{len(ds.data_list)}] {sid}: {len(cs)} child instances")

    med = float(np.median(sizes)) if sizes else float("nan")
    print(f"\nscenes {len(preds)} · child instances {n_inst} · median child mask {med:.0f} points "
          f"· threshold {a.thr}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"preds": preds, "meta": dict(run=a.run, ckpt=ckpt_path, thr=a.thr,
                                                  split=a.split, score_rule="parent",
                                                  association=a.association,
                                                  min_points=MIN_POINTS)}, fh, protocol=4)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
