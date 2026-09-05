"""Track 2, RELEASED member: the base config trained for a longer schedule.

One variable against the base: total training passes. Note that accuracy peaks partway through the
longer schedule rather than at its end, so the released checkpoint is the trainer's best rather
than its last.
"""
_base_ = ["./semseg-volt-B-armA.py"]

# THE EXPERIMENT: double the data passes. Everything else inherited from run #1's config.
epoch = 400
