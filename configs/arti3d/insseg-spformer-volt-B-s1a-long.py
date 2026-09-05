"""Track 1, released configuration: the base config trained for a longer schedule.

One variable against the base: the number of training epochs. Checkpoint selection is post-hoc on
the validation ranking metric rather than on the trainer's own segmentation metric, because those
are different quantities and the trainer optimises the wrong one for this task; the released
checkpoint is epoch 14, which motion quality peaks at.
"""
_base_ = ["./insseg-spformer-volt-B-s1a.py"]

# THE EXPERIMENT: 200 -> 400 passes over the train set. Nothing else about the model changes.
epoch = 400

# Logging honesty, not an experimental variable for reproducibility, not an experimental variable.
segment_ignore_index = (-1, 0)
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator", segment_ignore_index=segment_ignore_index,
         min_region_size=1),
    dict(type="CheckpointSaver", save_freq=2),
]

# A different seed would confound the comparison; keep it pinned to the parent's
# value so the ONLY difference between this run and the Track-1 instance model is the number of gradient steps.
seed = 20260816
