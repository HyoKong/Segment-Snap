"""Track 2 ablation: a less aggressive background down-weight. ONE variable, 0.1 -> 0.3.

Inherits the base config and changes one number, so the diff is the experiment. The base weighting
puts a tenfold relative premium on predicting a handle, and the model over-fires accordingly; this
probes whether that premium is set too high. 0.3 is a first probe, not a tuned value.
"""
_base_ = ["./semseg-volt-B-armA.py"]

# THE EXPERIMENT: background weight 0.1 -> 0.3. Everything else inherited.
model = dict(
    criteria=[
        dict(type="CrossEntropyLoss", weight=[0.3, 1.0, 1.0], loss_weight=1.0,
             label_smoothing=0.1, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Pinned so this is comparable to run #1 the same way the c2f ablation was — the A/B partner there
# shares run #1's drawn seed, and a third arm on a different seed could not be compared to either.
seed = 10516805
