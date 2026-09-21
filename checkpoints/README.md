# Released checkpoints

Three checkpoints reproduce every number in `docs/RESULTS_VAL.md`. All three are **train-only**:
they saw the 195 training scenes and never the 42 validation scenes. They are hosted on Hugging Face
at [`imsuperkong/Segment-Snap`](https://huggingface.co/imsuperkong/Segment-Snap), with the
`config.py` each was trained with beside it.

| file | model | selected on | bytes | md5 |
|---|---|---|---:|---|
| `t1_spformer/model/epoch_14.pth` | Track-1 part model: SPFormer over superpoints on Volt-B | validation ranking column, post hoc | 1,705,908,057 | `6b44303b6b93618862f74f3620c5cb42` |
| `t2_semantic/model/model_best.pth` | Track-2 dense model: 3-class point semantic segmentation on Volt-B | the trainer's mIoU | 1,504,787,251 | `2ab3ac49afee2a5fffcbf8f363e30a87` |
| `s2_joint/model/model_last.pth` | joint model: the part model plus the child (handle) head | last | 1,714,611,973 | `6a2c0867f8191b1d0b201a720e92b18b` |

sha256, as stored on the Hub: `cbe543cd5229f635e99aba64450c3d384971d462b004be915891e766a7774930`,
`c96d78e53c047496276d59ab8ffdd0dc9f00a31193f8ce75e4245f2e347d87ad`,
`c5f0be9c0667cd4a07e1f6420044bb964a1b08535dd8b44e8c1a20f3744144d6`, in the order of the table.

## Download

```bash
python scripts/download_checkpoints.py --dest checkpoints          # fetches and md5-verifies all three
python scripts/download_checkpoints.py --dest checkpoints --verify-only   # re-check what is on disk
```

The script fails rather than warns on a mismatch: a silently truncated checkpoint loads without
complaint and produces plausible, wrong numbers. `scripts/reproduce_val.sh` reads each model's
configuration from the checkpoint directory (`checkpoints/<model>/config.py`), not from `configs/`,
so a released checkpoint always carries the configuration that produced it. The weights themselves
(`checkpoints/*/model/`) are gitignored.

## Three things to know

**The Track-1 checkpoint is not the trainer's choice.** The trainer selects on segmentation AP50,
which is not the metric this task ranks on. `epoch_14` is the 14th of the 20 evaluation checkpoints
of the 400-epoch schedule (70 % of training), chosen by scoring every saved checkpoint through the
full pipeline on validation and taking the best on the ranking column; motion quality degrades
measurably at later checkpoints even while segmentation keeps improving. The trainer's own best
scores 0.41351 on that column (+0.0037, inside scene-sampling noise); the selection is disclosed as
validation-based.

**The seeds are recorded.** `configs/arti3d/semseg-volt-B-armA-long.py` leaves `seed` unset, so the
trainer mints one per run; the shipped `t2_semantic/config.py` records the seed the released run
actually drew (55921373). That key is the only setting in which the three shipped `config.py` files
differ from their counterparts in `configs/arti3d/`, verified by comparing the loaded configuration
dictionaries rather than the file text.

**The backbone pretrain is not redistributed.** Inference from the released checkpoints does not
need it (the weights are inside them); retraining does. The configs load it from
`weights/volt-base-scannetpp.pth` in the repository root, from the upstream Volt release:

```bash
mkdir -p weights
curl -L -o weights/volt-base-scannetpp.pth \
  https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_base/scannetpp/model/model_last.pth
```

All three checkpoints hold EMA weights, which the inference scripts load by default
(`--weights ema`); a run trained without EMA is read with `--weights raw`.
