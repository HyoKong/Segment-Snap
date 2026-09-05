weight = 'weights/volt-base-scannetpp.pth'
resume = False
evaluate = True
test_only = False
seed = 55921373
save_path = 'exp/default'
num_worker = 16
batch_size = 2
gradient_accumulation_steps = 8
batch_size_val = None
batch_size_test = None
epoch = 400
eval_epoch = 20
clip_grad = None
use_ema = True
ema_decay = 0.999
dataset_ratios = None
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'Volt'
wandb_key = None
mix_prob = 0.0
param_dicts = None
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='ArtiC2FHook', schedule=((0.0, 0.1), (0.5, 0.04), (0.8, 0.0)))
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
min_points = 3
knn = 32
knn_radius = 0.025
model = dict(
    type='DefaultSegmentorV2',
    num_classes=3,
    backbone_out_channels=256,
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
    criteria=[
        dict(
            type='CrossEntropyLoss',
            weight=[0.1, 1.0, 1.0],
            loss_weight=1.0,
            label_smoothing=0.1,
            ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
c2f_schedule = ((0.0, 0.1), (0.5, 0.04), (0.8, 0.0))
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.0003,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'ArtiC2FDataset'
data_root = 'data/pointcept_lite'
GRID = 0.02
data = dict(
    num_classes=3,
    ignore_index=-1,
    names=['background', 'rotation_handle', 'translation_handle'],
    train=dict(
        type='ArtiC2FDataset',
        split='train',
        data_root='data/pointcept_lite',
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
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='ElasticDistortion',
                distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', sample_rate=0.6, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('color', 'normal'))
        ],
        test_mode=False,
        loop=20),
    val=dict(
        type='ArtiC2FDataset',
        split='validation',
        data_root='data/pointcept_lite',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'origin_segment',
                      'inverse'),
                feat_keys=('color', 'normal'))
        ],
        test_mode=False),
    test=dict(
        type='ArtiC2FDataset',
        split='validation',
        data_root='data/pointcept_lite',
        transform=[
            dict(
                type='Copy',
                keys_dict=dict(coord='origin_coord',
                               segment='origin_segment')),
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('color', 'normal'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }]])))
num_worker_per_gpu = 16
batch_size_per_gpu = 2
batch_size_val_per_gpu = 1
batch_size_test_per_gpu = 1
