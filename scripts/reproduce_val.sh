#!/usr/bin/env bash
# Reproduce every number in the README from the three released checkpoints, on one GPU.
#
# The two tracks depend on each other, so the order is forced and is not arbitrary:
#
#   1. Track 2 semantic  ->  probability field
#   2. Track 2 instances ->  connected components.  THESE ARE ALSO TRACK 1's HANDLES.
#   3. Track 1           ->  part masks + geometric motion, using those handles for hinge origins
#   4. S2 child head     ->  a second, independent set of handle detections
#   5. Track 2 union     ->  child appended strictly below the incumbents
#   6. Track 2 classvote ->  Track 1's parts correct the children's class labels
#
# Track 1's origins need Track 2's handles (step 2 -> 3) and Track 2's class vote needs Track 1's
# parts (step 3 -> 6). The cycle is only apparent: the handles come from step 2, before the union
# and the vote, so the graph is acyclic and this is the only order that satisfies it.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/pointcept_mov}"
LITE_ROOT="${LITE_ROOT:-data/pointcept_lite}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
# Root of the organisers' processed release, which the evaluator reads ground truth from.
export ARTI3D_GT_ROOT="${ARTI3D_GT_ROOT:-data/a3d/processed}"
OUT="${OUT:-runs/reproduce_val}"
SPLIT=validation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/third_party/volt"

T1_RUN="${T1_RUN:-$CKPT_DIR/t1_spformer}"     # holds config.py + model/epoch_14.pth
T2_RUN="${T2_RUN:-$CKPT_DIR/t2_semantic}"     # holds config.py + model/model_best.pth
S2_RUN="${S2_RUN:-$CKPT_DIR/s2_joint}"        # holds config.py + model/model_last.pth

mkdir -p "$OUT"
say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "1/6  Track 2 semantic inference"
python scripts/infer_t2_sem.py --run "$T2_RUN" --ckpt best --split $SPLIT \
    --data-root "$LITE_ROOT" --out "$OUT/t2_probs"

say "2/6  Track 2 instances (these are also Track 1's handles)"
python scripts/instances_t2.py --probs "$OUT/t2_probs" --split $SPLIT \
    --data-root "$LITE_ROOT" --out "$OUT/t2_single"

say "3/6  Track 1: parts + geometric motion, origins from the handles above"
python scripts/infer_t1.py --run "$T1_RUN" --ckpt epoch_14 --split $SPLIT \
    --data-root "$DATA_ROOT" --handles "$OUT/t2_probs" --out "$OUT/t1"

say "3b/6 Track 1 ablation: no handles, origins fall back to the box centroid"
python scripts/infer_t1.py --run "$T1_RUN" --ckpt epoch_14 --split $SPLIT \
    --data-root "$DATA_ROOT" --out "$OUT/t1_centroid"

say "4/6  S2 child head"
python scripts/infer_s2_child.py --run "$S2_RUN" --ckpt last --split $SPLIT \
    --data-root "$DATA_ROOT" --child-root "$LITE_ROOT" --out "$OUT/child.pkl"

say "5/6  Track 2 + union"
python scripts/instances_t2.py --probs "$OUT/t2_probs" --split $SPLIT \
    --data-root "$LITE_ROOT" --union-child "$OUT/child.pkl" --out "$OUT/t2_union"

say "6/6  Track 2 + class vote from Track 1's parts"
python scripts/classvote.py --t2 "$OUT/t2_union/t2_${SPLIT}_preds.pkl" \
    --t1 "$OUT/t1/t1_${SPLIT}_preds.pkl" --out "$OUT/t2_voted"

say "RESULTS"
python - "$OUT" <<'PY'
import json, sys, os
o = sys.argv[1]
def m(p):
    f = os.path.join(o, p, "metrics.json")
    return json.load(open(f))["metrics"] if os.path.exists(f) else {}
t1, t1c = m("t1"), m("t1_centroid")
print("  Track 1 (mov, validation)")
for k, lab in (("AP50", "AP50"), ("MA_ap50", "AP50_axis"), ("MO_ap50", "AP50_origin"),
               ("MAO_ST_ap50", "AP50_axis_origin  <- ranking column")):
    print(f"    {lab:34s} {t1.get(k, float('nan')):.5f}")
if t1c:
    a, b = t1c.get("MAO_ST_ap50", 0.0), t1.get("MAO_ST_ap50", 0.0)
    print(f"    ablation, origins = centroid        {a:.5f}"
          f"   ({b/a:.2f}x from the Track-2 handles)" if a else "")
print("  Track 2 (inter, validation)")
for p, lab in (("t2_single", "single model"), ("t2_union", "+ S2 child union"),
               ("t2_voted", "+ class vote")):
    print(f"    {lab:34s} {m(p).get('AP50', float('nan')):.5f}")
PY
