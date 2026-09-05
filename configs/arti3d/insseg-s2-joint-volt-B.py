"""Joint model: the Track 1 architecture plus a per-point CHILD head.

Inherits the Track 1 released configuration, so the parent branch is the same model; the addition
is a head that predicts, for each part query, which points are that part's handle. That makes it a
second and independent source of handle detections for Track 2, produced by a different mechanism
from the semantic network, and the two fail in different places — which is what makes their union
worth anything.

The child loss is shaped for TIGHTNESS rather than coverage: a handle is a few tens of points
inside a part of several hundred, so a head rewarded for coverage simply predicts the parent.
"""
_base_ = ["./insseg-spformer-volt-B-s1a-long.py"]

import arti3d.datasets.joint as _joint_register  # noqa: F401
import arti3d.models.joint_spformer as _s2_register  # noqa: F401
del _joint_register, _s2_register

num_channels = 256
semantic_num_classes = 2
segment_ignore_index = (-1, 0)
CHILD_ROOT = "data/pointcept_lite"

model = dict(
    type="ArtiJointSPFormer-v1",
    # Weight on the child term in the total loss. 1.0 is a starting point, not a tuned value: the
    # parent's own weights sum to ~2.7, so this makes the child a comparable but not dominant
    # contributor. It is the first thing to sweep once the head trains at all.
    child_loss_weight=1.0,
    decoder=dict(
        type="ArtiJointDecoder",
        # The child head reads VOXEL features straight off the backbone, so its input width is the
        # backbone's output width -- NOT the decoder's d_model, and not the superpoint feature the
        # parent's x_mask consumes. Getting this wrong is a shape error at the first batch, which is
        # the good case; getting it "nearly right" by passing d_model would silently build a head on
        # a projection that never sees the backbone.
        child_in_channel=num_channels,
    ),
    criterion=dict(
        type="ArtiJointCriterion",
        # (bce, dice, tversky). Tversky carries the largest weight because it is the only term that
        # penalises over-coverage asymmetrically, which is the failure mode being targeted.
        child_weight=(1.0, 1.0, 2.0),
        tversky=(0.7, 0.3),
    ),
)

dataset_type = "ArtiJointDataset"
GRID = 0.02

_train_transform = [
    dict(type="SphereCrop", point_max=400000, mode="random"),
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
    # No RandomFlip, no ElasticDistortion, no RandomShift — none of these are rigid, and motion targets must survive the transform.
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.95),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.95),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.95),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
    dict(type="Copy", keys_dict={"coord": "origin_coord",
                                 "instance": "origin_instance",
                                 "segment": "origin_segment",
                                 "child_instance": "origin_child_instance"}),
    dict(type="Update", keys_dict={"index_valid_keys": [
        "coord", "color", "normal", "segment", "instance", "child_instance"]}),
    dict(type="GridSample", grid_size=GRID, hash_type="fnv", mode="train",
         return_grid_coord=True, return_inverse=True),
    dict(type="NormalizeColor"),
    dict(type="InstanceParser", segment_ignore_index=segment_ignore_index,
         instance_ignore_index=-1),
    dict(type="ToTensor"),
    dict(type="Collect",
         keys=("coord", "origin_coord", "grid_coord", "segment", "origin_segment",
               "instance", "origin_instance", "child_instance", "origin_child_instance",
               "superpoint", "inverse"),
         feat_keys=("color", "normal"),
         offset_keys_dict=dict(offset="coord", origin_offset="origin_coord")),
]

_val_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="Copy", keys_dict={"coord": "origin_coord",
                                 "instance": "origin_instance",
                                 "segment": "origin_segment",
                                 "child_instance": "origin_child_instance"}),
    dict(type="Update", keys_dict={"index_valid_keys": [
        "coord", "color", "normal", "segment", "instance", "child_instance"]}),
    # deterministic voxel representative, so val numbers do not carry voxel-choice jitter.
    dict(type="GridSampleDeterministic", grid_size=GRID, hash_type="fnv",
         return_grid_coord=True, return_inverse=True),
    dict(type="NormalizeColor"),
    dict(type="InstanceParser", segment_ignore_index=segment_ignore_index,
         instance_ignore_index=-1),
    dict(type="ToTensor"),
    dict(type="Collect",
         keys=("coord", "origin_coord", "grid_coord", "segment", "origin_segment",
               "instance", "origin_instance", "child_instance", "origin_child_instance",
               "superpoint", "inverse", "name"),
         feat_keys=("color", "normal"),
         offset_keys_dict=dict(offset="coord", origin_offset="origin_coord")),
]

data = dict(
    train=dict(type=dataset_type, child_root=CHILD_ROOT, transform=_train_transform),
    val=dict(type=dataset_type, child_root=CHILD_ROOT, transform=_val_transform),
    test=dict(type=dataset_type, child_root=CHILD_ROOT, transform=_val_transform),
)
