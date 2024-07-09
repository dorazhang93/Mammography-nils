_base_ = [
    '../default_runtime.py',
]

# >>>>>>>>>>>>>>> Override data settings here >>>>>>>>>>>>>>>>>>>
dataset_type = 'MammoSingleTaskDataset'
data_root = '/home/avesta/daqu/Projects/Mammography/privateData/cohort12_ROIs_certaintyNone_2layers_5fold/ROI500/0'
data_preprocessor = dict(
    # RGB format normalization parameters, this is decided by the pretrianed model
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=False,
)
policies = [
        dict(type='AutoContrast'),
        dict(type='Rotate', magnitude_range=(-20, 20)),
        dict(type='Brightness', magnitude_range=(-0.25,0.25)),
        dict(type='Sharpness', magnitude_range=(-0.25,0.25)),
        dict(type='Equalize',prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=500, crop_ratio_range=(0.5, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandAugment',
         policies=policies,
         num_policies=3,
         magnitude_level=6,
         magnitude_std=0.5,
         ),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=500),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='0/meta/train_N.txt',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='0/meta/val_N.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='0/meta/test_N.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [dict(type='BinaryCrossEntropy'),dict(type='AUC')]
test_evaluator = val_evaluator

# >>>>>>>>>>>>>>> Override model settings here >>>>>>>>>>>>>>>>>>>
model = dict(
    type='ImageTabClassifier',
    backbone=dict(
        type='ResNetWs', # width scaling
        depth=50,
        expansion=2,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(type='SingleLinearClsHead',
              num_classes=1,
              in_channels=1024,
              loss_weight=1.0,
              loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),# use_sigmoid=True call function F.binary_cross_entropy_with_logits
              ),
)

# >>>>>>>>>>>>>>> Override schedules settings here >>>>>>>>>>>>>>>>>>>

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.05, eps=1e-08, betas=(0.9, 0.999), weight_decay=0.001))

# learning policy
param_scheduler = dict(type='StepLR',step_size=1, by_epoch=True, gamma=0.95)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)


# >>>>>>>>>>>>>>> Override default_runtime settings here >>>>>>>>>>>>>>>>>>>
log_processor=dict(window_size=40)

# configure default hooks
default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=True),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook',
                    interval=5,
                    max_keep_ckpts=1,
                    by_epoch=True,
                    save_best='auto'),

)
# set log level
log_level = 'DEBUG'
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=0, deterministic=False)

