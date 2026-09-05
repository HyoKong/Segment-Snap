"""Checkpoint resolution and weight loading, shared by both inference entry points."""
from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict


def md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def resolve_ckpt(run: str, spec: str) -> str:
    """`best`/`last` -> model/model_<spec>.pth; `N` or `epoch_N` -> model/epoch_N.pth; else a path.

    The two forms exist because the trainer names them differently: `model_best.pth` and
    `model_last.pth` carry the prefix, while the periodic checkpoints written at `save_freq` do
    not (`epoch_2.pth`, `epoch_4.pth`, ...). Selecting a checkpoint on the ranking metric rather
    than on the trainer's own metric means addressing the periodic ones, so both forms resolve.
    """
    mdir = os.path.join(run, "model")
    if spec in ("best", "last"):
        p = os.path.join(mdir, f"model_{spec}.pth")
    elif re.fullmatch(r"(epoch_)?\d+", spec):
        p = os.path.join(mdir, f"epoch_{spec.split('_')[-1]}.pth")
    else:
        p = spec
    if not os.path.exists(p):
        avail = sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []
        raise SystemExit(f"no checkpoint at {p}\n  available in {mdir}: "
                         f"{', '.join(avail) if avail else '(none)'}")
    return p


def load_weights(model, ckpt: dict, which: str = "ema"):
    """Load `ema_state_dict` (default) or `state_dict`, normalising the `module.` prefix.

    Training used an exponential moving average of the weights and the validation curve in each
    run's log is the EMA curve, so `ema` is the setting that reproduces the recorded numbers.
    """
    key = "ema_state_dict" if which == "ema" else "state_dict"
    sd = ckpt.get(key)
    assert sd is not None, (
        f"checkpoint has no {key!r}. A run trained without EMA needs --weights raw, but its "
        "numbers are then not comparable to the EMA validation curve in its log.")
    w = OrderedDict()
    for k, v in sd.items():
        if not k.startswith("module."):
            k = "module." + k
        w[k[7:]] = v
    return model.load_state_dict(w, strict=True)
