weight = 'weights/volt-base-scannetpp.pth'
resume = False
evaluate = True
test_only = False
seed = 20260816
save_path = 'exp/default'
num_worker = 16
batch_size = 2
gradient_accumulation_steps = 8
batch_size_val = None
batch_size_test = None
epoch = 400
eval_epoch = 20
clip_grad = 10.0
use_ema = True
ema_decay = 0.999
dataset_ratios = None
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = True
enable_wandb = False
wandb_project = 'Volt'
wandb_key = None
mix_prob = 0.0
param_dicts = None
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(
        type='InsSegEvaluator',
        segment_ignore_index=(-1, 0),
        min_region_size=1),
    dict(type='CheckpointSaver', save_freq=2)
]
train = dict(type='DefaultTrainer')
test = dict(
    type='InsSegTester',
    verbose=True,
    segment_ignore_index=(-1, 0),
    instance_ignore_index=-1)
SUPERPOINT_PARAMS = dict(k_thresh=0.005, seg_min=5)
num_classes = 3
segment_ignore_index = (-1, 0)
semantic_num_classes = 2
num_channels = 256
model = dict(
    type='SPFormer-v1m1',
    backbone=dict(
        type='Volt',
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
        up_mlp_dim=256),
    decoder=dict(
        type='SPFormerDecoder',
        num_class=2,
        in_channel=256,
        num_layer=6,
        num_query=200,
        d_model=384,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        use_query_pos=False,
        use_score=True,
        use_param_query=True),
    criterion=dict(
        type='SPFormerCriterion',
        matcher=dict(
            type='SPFormerHungarianMatcher',
            costs=[
                dict(type='SPFormerQueryClassificationCost', weight=0.5),
                dict(type='SPFormerMaskBCECost', weight=1.0),
                dict(type='SPFormerMaskDiceCost', weight=1.0)
            ]),
        loss_weight=[0.2, 1.0, 1.0, 0.5],
        num_classes=2,
        non_object_weight=0.1,
        fix_dice_loss_weight=False,
        iter_matcher=True,
        fix_mean_loss=True),
    semantic_num_classes=2,
    semantic_ignore_index=-1,
    segment_ignore_index=(-1, 0),
    instance_ignore_index=-1,
    topk_insts=200,
    score_thr=0.0,
    npoint_thr=10,
    nms=True)
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.1)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.0003,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'ArtiInsSegDataset'
data_root = 'data/pointcept_mov'
GRID = 0.02
data = dict(
    num_classes=3,
    ignore_index=-1,
    names=['background', 'rotation', 'translation'],
    train=dict(
        type='ArtiInsSegDataset',
        split='train',
        data_root='data/pointcept_mov',
        motion_targets=False,
        require_assets=('coord', 'color', 'normal', 'segment', 'instance',
                        'superpoint'),
        transform=[
            dict(type='SphereCrop', point_max=400000, mode='random'),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.95),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.95),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.95),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(type='SphereCrop', sample_rate=0.6, mode='random'),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    instance='origin_instance',
                    segment='origin_segment')),
            dict(
                type='Update',
                keys_dict=dict(index_valid_keys=[
                    'coord', 'color', 'normal', 'segment', 'instance'
                ])),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='NormalizeColor'),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, 0),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'origin_coord', 'grid_coord', 'segment',
                      'origin_segment', 'instance', 'origin_instance',
                      'superpoint', 'inverse'),
                feat_keys=('color', 'normal'),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False,
        loop=20),
    val=dict(
        type='ArtiInsSegDataset',
        split='validation',
        data_root='data/pointcept_mov',
        motion_targets=False,
        require_assets=('coord', 'color', 'normal', 'segment', 'instance',
                        'superpoint'),
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    instance='origin_instance',
                    segment='origin_segment')),
            dict(
                type='Update',
                keys_dict=dict(index_valid_keys=[
                    'coord', 'color', 'normal', 'segment', 'instance'
                ])),
            dict(
                type='GridSampleDeterministic',
                grid_size=0.02,
                hash_type='fnv',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='NormalizeColor'),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, 0),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'origin_coord', 'grid_coord', 'segment',
                      'origin_segment', 'instance', 'origin_instance',
                      'superpoint', 'inverse', 'name'),
                feat_keys=('color', 'normal'),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False),
    test=dict(
        type='ArtiInsSegDataset',
        split='validation',
        data_root='data/pointcept_mov',
        motion_targets=False,
        require_assets=('coord', 'color', 'normal', 'segment', 'instance',
                        'superpoint'),
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    instance='origin_instance',
                    segment='origin_segment')),
            dict(
                type='Update',
                keys_dict=dict(index_valid_keys=[
                    'coord', 'color', 'normal', 'segment', 'instance'
                ])),
            dict(
                type='GridSampleDeterministic',
                grid_size=0.02,
                hash_type='fnv',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='NormalizeColor'),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, 0),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'origin_coord', 'grid_coord', 'segment',
                      'origin_segment', 'instance', 'origin_instance',
                      'superpoint', 'inverse', 'name'),
                feat_keys=('color', 'normal'),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False))
num_worker_per_gpu = 16
batch_size_per_gpu = 2
batch_size_val_per_gpu = 1
batch_size_test_per_gpu = 1
