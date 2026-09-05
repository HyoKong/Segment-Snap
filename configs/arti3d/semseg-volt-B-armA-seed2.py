"""Track 2 replicate: the base config with a different training seed. ONE variable, the seed.

This exists to measure the noise floor. Two runs of the same configuration differing only in the
seed are 0.00675 AP50 apart through the full released pipeline on validation — which is the number
any claimed difference between members has to beat before it means anything.

The seed reaches three places: weight initialisation, the data sampler, and the per-worker
augmentation streams.
"""
_base_ = ["./semseg-volt-B-armA.py"]

# THE EXPERIMENT: a different draw. Everything else inherited from run #1's config.
# Run #1's effective seed was 10516805 (drawn, not pinned); this must differ from it.
seed = 20260816
