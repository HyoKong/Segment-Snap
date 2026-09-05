"""Track 1 base config: SPFormer instance segmentation over superpoints on a Volt-B backbone.

Two classes, {rotation, translation}, over the movable parts of an indoor scan. The non-default
choices that matter, and why:

  * `num_query` 800 -> 200. There are about nine movable parts per scene, not eighty-four object
    categories.
  * `npoint_thr` 100 -> 10. This is the most dangerous default in the file: handles and small
    drawers are tens of points, and the stock threshold deletes the entire population the task is
    about.
  * `topk_insts` 400 -> 200, matching `num_query`. The selection rule itself is replaced at
    inference by a per-query argmax (see arti3d/models/spformer_argmax.py); this cap bounds the
    count but does not change the rule.
  * `epoch` 1600 -> 200. The longer schedule costs many hours for no measured gain here.
  * initialised from the Volt ScanNet++ pretrain: these scenes ARE ScanNet++ scenes, so the
    pretraining distribution is the evaluation distribution.

Superpoints come from a Felzenszwalb-Huttenlocher segmentation of a kNN graph; the parameters were
swept for the achievable ceiling on movable parts rather than taken from the defaults.
"""
_base_ = ["../../third_party/volt/configs/_base_/default_runtime.py"]

import arti3d.datasets.insseg as _insseg_register  # noqa: F401
import arti3d.datasets.transforms as _tf_register  # noqa: F401  — GridSampleDeterministic
del _insseg_register, _tf_register

# ---------------------------------------------------------------------------------------------
# Superpoint parameters, chosen by a sweep for the achievable ceiling on movable parts.
# Chosen on the PER-CLASS bar, not the macro: (0.01, 5) clears the 94% macro gate at 95.8% but its
# TRANSLATION ceiling is 91.6%, under the 94% per-class floor, on the class that is half the score.
# Rotation is 100% at every setting from (0.01, 5) up, so the macro alone would have hidden the only
# binding constraint. Stopped at segMinVerts=5 rather than the 100.0% at segMinVerts=2, which costs
# 2.6x the superpoints (36k/scene) for +1.6pp of a ceiling we are nowhere near using.
# This is the record of what the checked-in superpoints were built with — changing it invalidates
# every checkpoint trained on the old ones.
SUPERPOINT_PARAMS = dict(k_thresh=0.005, seg_min=5)
# ---------------------------------------------------------------------------------------------

batch_size = 2
gradient_accumulation_steps = 8      # effective 16, the proven recipe (the semantic arm used the same)
# 16, NOT 28. 28 killed the first the Track-1 instance model attempt at the end of epoch 1 with
# `RuntimeError: can't start new thread`. Pointcept builds the VAL dataloader while the TRAIN
# loader's workers are still alive, so worker processes momentarily DOUBLE, and this container caps
# cgroup pids at 1792. The real fix is OMP_NUM_THREADS=1 (now set by launch_run.sh; nothing was
# capping OpenMP, so each worker could spawn up to 48 threads). 16 is Arm A's proven value and is
# used here deliberately rather than stacking an unproven worker count on top of a fresh incident —
# raise it only with a measured pids.current headroom figure from a completed run.
num_worker = 16
mix_prob = 0.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
use_ema = True
clip_grad = 10.0
evaluate = True
find_unused_parameters = True
enable_wandb = False                 # no external telemetry

# F4: pinned, not drawn. default_runtime.py ships seed=None and Pointcept mints a random one per
# run (defaults.py:125-126), which is how run #1 ended up needing its seed reverse-engineered out
# of a frozen config to make its ablation single-variable. Any future A/B against the Track-1 instance model now differs
# only where intended.
seed = 20260816

weight = "weights/volt-base-scannetpp.pth"

epoch = 200
eval_epoch = 20

num_classes = 3                      # {0 background, 1 rotation, 2 translation}
segment_ignore_index = (-1, 0)
semantic_num_classes = 2
num_channels = 256

model = dict(
    type="SPFormer-v1m1",
    backbone=dict(
        type="Volt",
        in_channels=6,
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
    decoder=dict(
        type="SPFormerDecoder",
        num_class=semantic_num_classes,
        in_channel=num_channels,
        num_layer=6,
        num_query=200,
        d_model=384,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="gelu",
        iter_pred=True,
        attn_mask=True,
        use_query_pos=False,
        use_score=True,
        use_param_query=True,
    ),
    criterion=dict(
        type="SPFormerCriterion",
        matcher=dict(
            type="SPFormerHungarianMatcher",
            costs=[
                dict(type="SPFormerQueryClassificationCost", weight=0.5),
                dict(type="SPFormerMaskBCECost", weight=1.0),
                dict(type="SPFormerMaskDiceCost", weight=1.0),
            ],
        ),
        loss_weight=[0.2, 1.0, 1.0, 0.5],
        num_classes=semantic_num_classes,
        non_object_weight=0.1,
        fix_dice_loss_weight=False,
        iter_matcher=True,
        fix_mean_loss=True,        # per-instance mask-loss normalisation, not per-point
    ),
    semantic_num_classes=semantic_num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    topk_insts=200,                # capped at num_query
    score_thr=0.0,
    npoint_thr=10,                 # handles are tens of points; the stock 100 deletes them
    nms=True,                      # decays scores, removes nothing
)

optimizer = dict(type="AdamW", lr=0.0003, weight_decay=0.1)
scheduler = dict(type="OneCycleLR", max_lr=optimizer["lr"], pct_start=0.05,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator", segment_ignore_index=segment_ignore_index),
    # Periodic checkpoints: selection is post-hoc on the validation ranking metric, over the last ~5
    # checkpoints, which requires more than one to exist — save_freq=None keeps only last+best,
    # and "best" is the trainer evaluator's own metric, not MAO-ST. save_freq=2 over
    # eval_epoch=20 leaves 10 epoch_N.pth files at 1.5 GB each = ~15 GB (468 GB free), which
    # comfortably covers "last ~5" while halving the disk and sync-back cost of saving every epoch.
    dict(type="CheckpointSaver", save_freq=2),
]

dataset_type = "ArtiInsSegDataset"
# F2: the MOVABLE root, not pointcept_lite. Both roots hold a key called `segment` and they mean
# different things — pointcept_lite's is the INTER label (handle points classed by their parent's
# motion type), this one is the MOV label (movable-part points, own motion type). pointcept_lite
# also has no instance.npy and no superpoint.npy, and DefaultDataset would substitute an all -1
# instance array rather than complain (defaults.py:126-138) — i.e. a full run against empty
# targets, with a plausible loss curve and nothing downstream to catch it. `require_assets` below
# turns that into a construction-time crash.
data_root = "data/pointcept_mov"
GRID = 0.02

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    # REQUIRED, and its absence is a launch-time crash rather than a warning: InsSegEvaluator
    # indexes cfg.data.names[i] for every i in range(num_classes) at before_train
    # (hooks/evaluator.py:248-258). The scannetpp template inherits `names` from
    # _base_/dataset/scannetpp.py; we do not include that base, so it has to be stated here.
    # Index order is the CHALLENGE encoding: 0 background, 1 rotation, 2 translation, and
    # segment_ignore_index=(-1, 0) drops background, leaving exactly the two scored classes.
    names=["background", "rotation", "translation"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        motion_targets=False,      # flip to True in S2; the guard then refuses flip/elastic
        require_assets=("coord", "color", "normal", "segment", "instance", "superpoint"),
        transform=[
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
                                         "segment": "origin_segment"}),
            # `superpoint` is DELIBERATELY ABSENT from this list. `index_operator`'s default
            # index_valid_keys DOES include it (transform.py:29-37), so without this Update the
            # GridSample below would subsample superpoint down to voxel resolution — and
            # SPFormer.forward splits `superpoint` by **origin_offset**, not voxel offset
            # (spformer.py:55,61), because superpoint pooling happens at full point resolution via
            # `inverse`. A voxel-length superpoint array there is a length mismatch at step 1.
            # That is what this Update is for in the upstream template too; it looks like a list of
            # things to keep, but its job is to DROP superpoint.
            dict(type="Update", keys_dict={"index_valid_keys": [
                "coord", "color", "normal", "segment", "instance"]}),
            dict(type="GridSample", grid_size=GRID, hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index,
                 instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "origin_coord", "grid_coord", "segment", "origin_segment",
                       "instance", "origin_instance", "superpoint", "inverse"),
                 feat_keys=("color", "normal"),
                 offset_keys_dict=dict(offset="coord", origin_offset="origin_coord")),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        motion_targets=False,
        require_assets=("coord", "color", "normal", "segment", "instance", "superpoint"),
        # NO geometric transform. The template applies a fixed 0.35*pi RandomRotate here with
        # always_apply=True, which would rotate every axis and origin we score and submit.
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"coord": "origin_coord",
                                         "instance": "origin_instance",
                                         "segment": "origin_segment"}),
            # `superpoint` is DELIBERATELY ABSENT from this list. `index_operator`'s default
            # index_valid_keys DOES include it (transform.py:29-37), so without this Update the
            # GridSample below would subsample superpoint down to voxel resolution — and
            # SPFormer.forward splits `superpoint` by **origin_offset**, not voxel offset
            # (spformer.py:55,61), because superpoint pooling happens at full point resolution via
            # `inverse`. A voxel-length superpoint array there is a length mismatch at step 1.
            # That is what this Update is for in the upstream template too; it looks like a list of
            # things to keep, but its job is to DROP superpoint.
            dict(type="Update", keys_dict={"index_valid_keys": [
                "coord", "color", "normal", "segment", "instance"]}),
            # deterministic voxelisation here, stock GridSample in train. Instance segmentation
            # cannot use the deterministic `mode="test"` form — that returns a LIST of fragments,
            # while SPFormer's superpoint-resolution masks and InsSegTester both need ONE cloud plus
            # `inverse` — so the RNG is removed at the source instead of avoided. Mechanism is in
            # src/arti3d/datasets/transforms.py. TRAIN deliberately keeps the random representative:
            # there it is augmentation, not noise, and determinism is only wanted where we measure.
            dict(type="GridSampleDeterministic", grid_size=GRID, hash_type="fnv",
                 return_grid_coord=True, return_inverse=True),
            dict(type="NormalizeColor"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index,
                 instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "origin_coord", "grid_coord", "segment", "origin_segment",
                       "instance", "origin_instance", "superpoint", "inverse", "name"),
                 feat_keys=("color", "normal"),
                 offset_keys_dict=dict(offset="coord", origin_offset="origin_coord")),
        ],
        test_mode=False,
    ),
    # `data.test` is the VAL SECTION, verbatim — which is what the upstream insseg template does
    # (`data["test"] = data["val"]`, insseg-spformer-volt-B-0-base.py:251) and NOT what a semseg
    # config does. The two testers want different shapes and this is easy to get backwards:
    #
    #   SemSegTester  consumes `fragment_list` — a LIST of voxelised fragments covering every point,
    #                 which is what test_mode=True + test_cfg.voxelize(mode="test") produces.
    #   InsSegTester  consumes ONE cloud with `origin_coord`/`origin_segment`/`origin_instance` plus
    #                 `inverse` (engines/test.py:991-1004), because SPFormer's masks live at
    #                 SUPERPOINT resolution and are expanded through the voxel cloud
    #                 (spformer.py:185). There is no fragment form of that.
    #
    # An earlier draft of this file carried the fragment-shaped section copied from Arm A. It
    # constructed and passed the coverage test, and would have failed the moment InsSegTester tried
    # to read `origin_instance` off it — a config that looks tested and is wrong for its own model.
    #
    # Determinism is preserved by `GridSampleDeterministic` in the val transform rather than by the
    # fragment form: the voxel-choice RNG is removed at the source, so this is reproducible too. The
    # validation split
    # ); the test-split twin is added then, not before.
)
data["test"] = data["val"]

# InsSegTester reports AP/AP50/AP25 only. MAO-ST — the ranking column — needs the geometric snap and
# the vendored evaluator, which is the post-hoc selection path, not this.
test = dict(
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=True,
)
