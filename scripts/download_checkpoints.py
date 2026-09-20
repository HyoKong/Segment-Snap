"""Fetch the three released checkpoints from the Hugging Face Hub and verify their md5s.

    python scripts/download_checkpoints.py --dest checkpoints            # from imsuperkong/Segment-Snap
    python scripts/download_checkpoints.py --repo <user>/<repo> --dest checkpoints   # a mirror

A silently truncated checkpoint loads without complaint and produces plausible, wrong numbers, so
the md5 check is not optional and this script fails rather than warns.
"""
from __future__ import annotations

import argparse
import hashlib
import os

#: file -> md5, as released. See checkpoints/README.md.
EXPECTED = {
    "t1_spformer/model/epoch_14.pth": "6b44303b6b93618862f74f3620c5cb42",
    "t2_semantic/model/model_best.pth": "2ab3ac49afee2a5fffcbf8f363e30a87",
    "s2_joint/model/model_last.pth": "6a2c0867f8191b1d0b201a720e92b18b",
}
CONFIGS = ["t1_spformer/config.py", "t2_semantic/config.py", "s2_joint/config.py"]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="imsuperkong/Segment-Snap",
                    help="Hugging Face repo id, <user>/<name> (default: the released checkpoints)")
    ap.add_argument("--dest", default="checkpoints")
    ap.add_argument("--verify-only", action="store_true", help="check what is already on disk")
    a = ap.parse_args()

    if not a.verify_only:
        from huggingface_hub import hf_hub_download
        for rel in list(EXPECTED) + CONFIGS:
            print(f"  fetching {rel}")
            hf_hub_download(repo_id=a.repo, filename=rel, local_dir=a.dest)

    bad = []
    for rel, want in EXPECTED.items():
        p = os.path.join(a.dest, rel)
        if not os.path.exists(p):
            bad.append((rel, "missing"))
            continue
        got = md5(p)
        ok = got == want
        print(f"  [{'ok  ' if ok else 'FAIL'}] {rel}  {got}")
        if not ok:
            bad.append((rel, f"{got} != {want}"))
    if bad:
        raise SystemExit(f"checkpoint verification FAILED: {bad}")
    print("\nall three checkpoints verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
