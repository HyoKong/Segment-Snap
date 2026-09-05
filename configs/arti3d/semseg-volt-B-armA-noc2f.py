"""Track 2 ablation: the base config with the coarse-to-fine label curriculum OFF.

The single-variable partner of the base run. The training seed is pinned to the base run's value,
because leaving it free would vary two things at once and make the comparison meaningless.

With the curriculum hook absent the dilation radius stays at zero for the whole run and the dataset
returns raw labels from the first step, so this trains on undilated targets throughout.
"""
_base_ = ["../../third_party/volt/configs/_base_/default_runtime.py"]

# Register ArtiC2FDataset + ArtiC2FHook into Pointcept's registries (needs src/ on PYTHONPATH).
# The bound name is deleted immediately: Pointcept turns every module-level name into a config
# attribute and then deepcopies the config, which fails with "cannot pickle 'module' object".
# The registration side-effect survives the del.
import arti3d.datasets.c2f as _c2f_register  # noqa: F401
del _c2f_register

# MEASURED, run #1 first attempt: bs=4 pinned VRAM at 32.0/32.6 GiB and OOM-skipped **18.8%** of
# batches — and the skips are biased toward the LARGEST scenes, so it was systematically
# under-training exactly the hard cases. The sweep's bs=4 figure was for a *median* scene; real
# batches vary well above that. bs=2 halves peak and costs nothing: the sweep measured bs=2 at
# 6.26 scenes/s vs bs=4 at 5.98. Gradient accumulation restores the proven effective batch of 16.
batch_size = 2
gradient_accumulation_steps = 8   # effective batch 16, the proven effective batch size
num_worker = 16
mix_prob = 0.0               # D-note: point_collate_fn's mix path knows nothing about our extra keys
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"       # R10.3
use_ema = True

# Pinned to run #1's drawn seed so the A/B varies the curriculum and nothing else. See docstring.
seed = 10516805

# No external telemetry. Pointcept's Trainer calls wandb.init()
# unconditionally when this is on, and wandb is deliberately not installed.
enable_wandb = False

weight = "weights/volt-base-scannetpp.pth"

# instance extraction at inference (kNN connected components) — measured ceiling 96.6% macro on the
# full 42-scene val set at r=0.025 (measured on the validation split)
min_points = 3               # NOT 10: 17.9% of interactable instances are under 10 points
knn = 32
knn_radius = 0.025

model = dict(
    type="DefaultSegmentorV2",
    num_classes=3,
    backbone_out_channels=256,
    backbone=dict(
        type="Volt",
        in_channels=6,           # color(3) + normal(3); normals ship in the data and USDNet ignores them
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        up_mlp_dim=256,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight=[0.1, 1.0, 1.0], loss_weight=1.0,
             label_smoothing=0.1, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 200
eval_epoch = 20

# ---------------------------------------------------------------------------------------------
# THE ABLATION. Run #1 carries, at this exact position:
#     c2f_schedule = ((0.0, 0.10), (0.5, 0.04), (0.8, 0.00))
#     ... dict(type="ArtiC2FHook", schedule=c2f_schedule),
# Both are gone here. `ArtiC2FDataset.current_radius` therefore stays at its 0.0 class default for
# the whole run and every epoch trains on raw, undilated labels.
# ---------------------------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
optimizer = dict(type="AdamW", lr=0.0003, weight_decay=0.05)
scheduler = dict(type="OneCycleLR", max_lr=optimizer["lr"], pct_start=0.05,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)

dataset_type = "ArtiC2FDataset"     # kept deliberately: same data path, dilation branch dormant
data_root = "data/pointcept_lite"
GRID = 0.02

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["background", "rotation_handle", "translation_handle"],
    train=dict(
        type=dataset_type, split="train", data_root=data_root,
        transform=[
            dict(type="SphereCrop", point_max=400000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),            # safe here: this model has no motion targets
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", grid_size=GRID, hash_type="fnv", mode="train",
                 return_grid_coord=True),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"),
                 feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type, split="validation", data_root=data_root,
        # NO geometric transform here beyond GridSample. The upstream ScanNet++ insseg config
        # ships a fixed 0.35*pi RandomRotate in its *val* pipeline, which would silently rotate
        # everything we then score and submit.
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=GRID, hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                 feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type, split="validation", data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"coord": "origin_coord", "segment": "origin_segment"}),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=GRID, hash_type="fnv", mode="test",
                          return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"),
                     feat_keys=("color", "normal")),
            ],
            # single identity view; no test-time augmentation.
            # aggregation must use the orientation tensor, never a raw vector mean
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z",
                                 center=[0, 0, 0], p=1)]],
        ),
    ),
)
