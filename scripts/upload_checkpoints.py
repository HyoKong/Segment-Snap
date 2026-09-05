"""Upload the three released checkpoints to the Hugging Face Hub.

    python scripts/upload_checkpoints.py --repo <user>/<repo> --src checkpoints

Reads no credentials of its own: authenticate first with `huggingface-cli login`, which stores a
token under your own account. Verifies every md5 against the released table BEFORE uploading, so a
corrupted local file cannot become the published one.
"""
from __future__ import annotations

import argparse
import os

from download_checkpoints import CONFIGS, EXPECTED, md5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, <user>/<name>")
    ap.add_argument("--src", default="checkpoints")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    for rel, want in EXPECTED.items():
        p = os.path.join(a.src, rel)
        assert os.path.exists(p), f"missing {p}"
        got = md5(p)
        assert got == want, f"{rel}: md5 {got} != released {want} — refusing to upload"
        print(f"  [ok  ] {rel}  {got}")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    for rel in list(EXPECTED) + CONFIGS:
        p = os.path.join(a.src, rel)
        if not os.path.exists(p):
            print(f"  [warn] {rel} absent, skipped")
            continue
        print(f"  uploading {rel}")
        api.upload_file(path_or_fileobj=p, path_in_repo=rel, repo_id=a.repo)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
