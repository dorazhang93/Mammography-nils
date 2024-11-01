_base_ = [
    '../default_runtime.py',
]

######################################### dataset settings
dataset_type = 'CustomDataset'
data_root = 'Mammography/privateData/SSL/patches'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True)

view_pipeline1 = [
    dict(type='RandomResizedCrop',
        scale=224,
        interpolation='bicubic',
        backend='pillow'),# Defaults to uniform(0.08, 1.0)
    dict(type='RandomFlip',
         prob=0.5),
    dict(type='AutoContrast'),
    dict(type='Brightness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Sharpness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Equalize',
         prob=0.5)
]
view_pipeline2 = [
    dict(type='RandomResizedCrop',
        scale=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip',
         prob=0.5),
    dict(type='AutoContrast'),
    dict(type='Brightness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Sharpness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Equalize',
         prob=0.5)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train_cancer_added_opensource.txt',
        with_label=False,
        pipeline=train_pipeline))

######################################################### model settings
model = dict(
    type='BYOL',
    base_momentum=0.01,
    backbone=dict(
        type='ResNetWs',
        depth=50,
        expansion=2,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=False,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth',
            prefix='backbone',
            type='Pretrained'),
        ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=1024,
        hid_channels=512,
        out_channels=128,
        num_layers=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=128,
            hid_channels=512,
            out_channels=128,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
        loss=dict(type='CosineSimilarityLoss')),
)

########################################################## optimizer
optimizer = dict(type='LARS', lr=0.6, momentum=0.9, weight_decay=1e-6)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }),
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)

log_processor=dict(window_size=100)

####################################################### runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3,
                    out_dir='work_dirs_ssl/byol_ep100_cancer_240k/ckpt',
                    ))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=512)
work_dir='work_dirs_ssl/byol_ep100_cancer_240k'
resume = True