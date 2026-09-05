"""Track 2 semantic inference: per-point 3-class probabilities at ORIGINAL cloud resolution.

    PYTHONPATH=.:third_party/volt python scripts/infer_t2_sem.py \
        --run <run dir> --ckpt best --split validation --out runs/armA/infer_validation

Writes `<out>/<sid>_prob.npy`, an (N, 3) float16 array of MEAN per-point probabilities over
{background, rotation-handle, translation-handle}, plus a manifest recording the checkpoint md5 and
per-scene coverage.

**The probabilities are the point, not the argmax.** An instance's score is the mean per-point
probability over its connected component, so saving only the class label would throw the scoring
signal away.

Two things are asserted rather than assumed:

  * COVERAGE. Test-mode voxelisation is supposed to be a full-coverage partition of the cloud. A
    point appearing in no fragment would divide by zero and, worse, silently become background in
    the argmax — so the fragment coverage of every point is checked before anything is written.
  * THE MEAN. Each accumulated row is a softmax that sums to 1, so the accumulator's row sum IS
    the number of times that point was predicted; dividing by it turns a sum back into a mean.

No geometric augmentation beyond voxelisation: the upstream instance-segmentation config ships a
fixed rotation in its validation pipeline, which would silently rotate everything we then score.
Inference is fp32 with TF32 disabled.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from arti3d.io import load_weights, md5, resolve_ckpt


def test_data_cfg(train_cfg, split: str, data_root: str):
    """The validation pipeline, spelled out rather than inherited, with one identity view."""
    from pointcept.utils.config import ConfigDict
    grid = float(train_cfg.data.test.test_cfg.voxelize.grid_size)
    #: feat_keys come from the RUN'S OWN test pipeline, never hardcoded: a model trained on a
    #: different per-point feature set silently mismatches the first matmul otherwise.
    fk = tuple(train_cfg.data.test.test_cfg.post_transform[-1].feat_keys)
    return ConfigDict(
        type=train_cfg.dataset_type, split=split, data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid, hash_type="fnv",
                          mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=fk),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0.0], axis="z",
                                 center=[0, 0, 0], p=1)]],
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best", help="best | last | N | epoch_N | path")
    ap.add_argument("--split", default="validation", choices=("train", "validation"))
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--data-root", default=None, help="overrides the training config's data_root")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", default=None, help="default <run>/infer_<split>")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    from pointcept.datasets import build_dataset
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    import arti3d.datasets.c2f  # noqa: F401  (registers the dataset type)

    cfg = Config.fromfile(os.path.join(a.run, "config.py"))
    data_root = a.data_root or cfg.data_root
    out_dir = a.out or os.path.join(a.run, f"infer_{a.split}")
    os.makedirs(out_dir, exist_ok=True)

    dataset = build_dataset(test_data_cfg(cfg, a.split, data_root))
    n_scenes = min(a.limit, len(dataset)) if a.limit else len(dataset)
    print(f"scenes {n_scenes} · grid {cfg.data.test.test_cfg.voxelize.grid_size} · "
          f"classes {cfg.data.num_classes} · data root {data_root}")

    ckpt_path = resolve_ckpt(a.run, str(a.ckpt))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    #: Inference computes no loss, and the criteria are what make this model CUDA-only: a weighted
    #: cross-entropy moves its weight tensor to the GPU at construction. Dropping them keeps a CPU
    #: verification path available.
    mcfg = dict(cfg.model)
    mcfg["criteria"] = []
    model = build_model(mcfg).to(a.device).eval()
    load_weights(model, ckpt, a.weights)
    ck_md5 = md5(ckpt_path)
    print(f"checkpoint {ckpt_path}  epoch {ckpt.get('epoch')}  weights={a.weights}  md5 {ck_md5}")

    manifest, t0 = {}, time.time()
    for i in range(n_scenes):
        d = dataset[i]
        sid = d["name"]
        n_pts = int(d["segment"].shape[0])
        frags = d["fragment_list"]

        seen = np.zeros(n_pts, dtype=np.int64)
        for fr in frags:
            seen[np.asarray(fr["index"])] += 1
        assert seen.min() > 0, (
            f"{sid}: {(seen == 0).sum()} of {n_pts} points appear in no fragment — test-mode "
            "voxelisation is supposed to be a full-coverage partition")

        acc = torch.zeros((n_pts, cfg.data.num_classes), dtype=torch.float32, device=a.device)
        for fr in frags:
            inp = {k: (v.to(a.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                   for k, v in fr.items()}
            with torch.inference_mode():
                logits = model(inp)["seg_logits"]
            acc.index_add_(0, inp["index"].long(), F.softmax(logits.float(), dim=-1))

        denom = acc.sum(1, keepdim=True)
        assert torch.all(denom > 0), sid
        prob = (acc / denom).cpu().numpy().astype(np.float16)
        np.save(os.path.join(out_dir, f"{sid}_prob.npy"), prob)

        pred = prob.argmax(1)
        manifest[sid] = dict(n_points=n_pts, fragments=len(frags),
                             cover_min=int(seen.min()), cover_max=int(seen.max()),
                             pred_counts=[int((pred == c).sum())
                                          for c in range(cfg.data.num_classes)])
        print(f"  [{i+1}/{n_scenes}] {sid}: {n_pts} pts, {len(frags)} frags, "
              f"pred {manifest[sid]['pred_counts']}")

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(dict(run=a.run, split=a.split, ckpt=ckpt_path, ckpt_md5=ck_md5,
                       weights=a.weights, scenes=n_scenes,
                       wall_s=round(time.time() - t0, 1), per_scene=manifest), fh, indent=1)
    print(f"\n{n_scenes} scenes in {round(time.time()-t0,1)}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
