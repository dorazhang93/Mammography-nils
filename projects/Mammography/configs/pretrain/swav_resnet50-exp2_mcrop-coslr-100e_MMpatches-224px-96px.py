_base_ = [
    '../default_runtime.py',
]

############################################ dataset settings
dataset_type = 'CustomDataset'
data_root = 'Projects/Mammography/privateData/SSL/patches'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True)

num_crops = [2, 6]
view_pipeline1 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.25, 1.),
        backend='pillow'),
    dict(type='AutoContrast'),
    dict(type='Brightness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Sharpness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Equalize',
         prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]
view_pipeline2 = [
    dict(
        type='RandomResizedCrop',
        scale=96,
        crop_ratio_range=(0.1, 0.25),
        backend='pillow'),
    dict(type='AutoContrast'),
    dict(type='Brightness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Sharpness',
         magnitude_range=(-0.5, 0.5),
         prob=0.5),
    dict(type='Equalize',
         prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiView',
        num_views=num_crops,
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackInputs')
]

batch_size=256
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train_cancer_added_opensource.txt',
        with_label=False,
        pipeline=train_pipeline))

########################################### model settings
model = dict(
    type='SwAV',
    data_preprocessor=dict(
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    backbone=dict(
        type='ResNetWs',
        depth=50,
        expansion=2,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220825-5b3fc7fc.pth',
            prefix='backbone',
            type='Pretrained'),
        ),
    neck=dict(
        type='SwAVNeck',
        in_channels=1024,
        hid_channels=512,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='SwAVHead',
        loss=dict(
            type='SwAVLoss',
            feat_dim=128,  # equal to neck['out_channels']
            epsilon=0.05,
            temperature=0.1,
            num_crops=num_crops,
            num_prototypes=300,
        )))

###################################################### optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='LARS', lr=0.6, weight_decay=1e-6, momentum=0.9))# original set: lr 0.6
find_unused_parameters = True

# learning policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=100,# original 200
        eta_min=6e-3,# original 6e-4
        by_epoch=True,
        begin=0,
        end=100,
        convert_to_iter_based=True)
]

##################################################### runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
log_processor=dict(window_size=100)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3,
                    out_dir='work_dirs_ssl/swav_ep100_cancer_240k/ckpt',
                    ))

# additional hooks
custom_hooks = [
    dict(
        type='SwAVHook',
        priority='VERY_HIGH',
        batch_size=batch_size,
        epoch_queue_starts=8,
        crops_for_assign=[0, 1],
        feat_dim=128,
        queue_length=1280,
        frozen_layers_cfg=dict(prototypes=1200))# num_iters_each_epoch=2360, when batch size=256
]
work_dir='work_dirs_ssl/swav_ep100_cancer_240k'
