#!/usr/bin/env bash
# Bring the tree to a runnable state. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f third_party/volt/pointcept/__init__.py ]; then
  echo "== fetching the Volt submodule"
  git submodule update --init --recursive
fi

echo "== installing Python requirements"
python -m pip install -r requirements.txt

cat <<'NOTE'

== done

Add the backbone to PYTHONPATH when running anything:

    export PYTHONPATH="$PWD:$PWD/third_party/volt"

Volt's own compiled extensions (spconv, pointops, pointgroup_ops) are NOT required: the model
registry imports optional families defensively and warns about the ones it skips. They are only
needed to run the model families this project does not use.

flash-attn is optional too. Without a build for your architecture the attention falls back to
torch's scaled_dot_product_attention, which is numerically equivalent and slower.
NOTE
