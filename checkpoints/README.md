# Released checkpoints

Three checkpoints reproduce every number in the top-level README. All three are **train-only**: they
saw the 195 training scenes and never the 42 validation scenes, so the validation numbers are an
honest held-out measurement.

| file | model | selected on | size | md5 |
|---|---|---|---|---|
| `t1_spformer/model/epoch_14.pth` | Track 1, SPFormer over superpoints on Volt-B | validation ranking metric, post-hoc | 1627 MB | `6b44303b6b93618862f74f3620c5cb42` |
| `t2_semantic/model/model_best.pth` | Track 2, 3-class point semantic segmentation | trainer's best | 1435 MB | `2ab3ac49afee2a5fffcbf8f363e30a87` |
| `s2_joint/model/model_last.pth` | joint model, child (handle) head | last | 1635 MB | `6a2c0867f8191b1d0b201a720e92b18b` |

Each directory must also contain the `config.py` the run was trained with; `scripts/reproduce_val.sh`
reads the config from the checkpoint directory, not from `configs/`, so a released checkpoint always
carries the configuration that produced it.

**Checkpoint selection is not the trainer's default for Track 1.** The trainer selects on
segmentation AP50, which is not the metric this task ranks on. The released Track-1 checkpoint was
chosen by scoring every saved epoch through the full pipeline on validation and taking the best on
the ranking column; motion quality degrades measurably at later epochs even while segmentation
accuracy keeps improving, so later is not better here.

**The Volt backbone pretrain is not redistributed, and training needs it.** Download
`volt-base-scannetpp.pth` from the upstream Volt release and put it at **`weights/`** in the
repository root — the configs load it from `weights/volt-base-scannetpp.pth`. Inference from the
released checkpoints does not need it (the weights are already in them); retraining does. These
scenes are ScanNet++ scenes, so the pretraining distribution is the evaluation distribution, which
is most of why the backbone transfers as well as it does.

**The Track-2 config's recorded seed.** `configs/arti3d/semseg-volt-B-armA-long.py` leaves `seed`
unset, so the trainer mints one per run. The released checkpoint's config records the seed that was
actually used, **55921373** — that one key is the only setting in which the three shipped
`checkpoints/*/config.py` differ from their counterparts in `configs/arti3d/`, verified by comparing
the loaded configuration dictionaries rather than the file text.

## Download

```
python scripts/download_checkpoints.py --dest checkpoints      # imsuperkong/Segment-Snap
```

Verify with the md5s above before running anything; a silently truncated checkpoint loads and
produces plausible, wrong numbers.
