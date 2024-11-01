_base_ = [
    '../default_runtime.py',
]

################################### dataset ##################################
dataset_type = 'MammoMultiTaskDataset'
data_root = 'Projects/Mammography/privateData'
data_preprocessor = dict(
    # RGB format normalization parameters, this is decided by the pretrianed model
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=False,)

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
    dict(type='MammoPackMultiTaskInputs', multi_task_fields=('gt_label',),
         task_handlers=dict(N=dict(type='PackInputs',algorithm_keys=['clinic_vars']),
                                LVI=dict(type='PackInputs',algorithm_keys=['clinic_vars']),
                                multifocality=dict(type='PackInputs'),
                                NumPos=dict(type='PackInputs'),
                                tumor_size=dict(type='PackInputs'))),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=500),# scale=(w,h)
    dict(type='MammoPackMultiTaskInputs', multi_task_fields=('gt_label',),
         task_handlers=dict(N=dict(type='PackInputs',algorithm_keys=['clinic_vars']),
                                LVI=dict(type='PackInputs',algorithm_keys=['clinic_vars']),
                                multifocality=dict(type='PackInputs'),
                                NumPos=dict(type='PackInputs'),
                                tumor_size=dict(type='PackInputs'))),
]


train_dataloader = dict(
    batch_size=72,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='0/meta/train.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='0/meta/val.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)


test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='0/meta/test.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator=dict(type='MultiTasksMetric',
                   task_metrics=dict(
                       N=[dict(type='BinaryCrossEntropy'),dict(type='AUC')],
                       LVI=[dict(type='BinaryCrossEntropy'),dict(type='AUC')],
                       multifocality=[dict(type='BinaryCrossEntropy'),dict(type='AUC')],
                       NumPos=[dict(type='MSELoss'), dict(type='R2')],
                       tumor_size=[dict(type='MSELoss'), dict(type='R2')],
                   ))
test_evaluator = val_evaluator

# >>>>>>>>>>>>>>>  model settings >>>>>>>>>>>>>>>>>>>
model = dict(
    type='ImageTabClassifier',
    backbone=dict(
        type='ResNetWs',
        depth=50,
        num_stages=4,
        expansion=2,
        out_indices=(3,),
        style='pytorch',
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs_ssl/barlowtwins_ep100_cancer_240k/ckpt/epoch_80.pth',
            prefix='backbone',
        ),),
    neck=dict(type='TransformerNeck',
              in_dims=1024,
              out_dims=5,
              depth=1,
              num_heads=8,
              head_dim=64,
              pooling_size=1,
              forward_layer=True,
              ff_hidden_layer=True,
              is_LSA=True,
              ),
    head=dict(type='MultiTaskHead',
              task_heads=dict(
                  N=dict(num_classes=1, loss_weight=1.0,task_idx=0,
                         loss=dict(type='CrossEntropyLoss', use_sigmoid=True), type='LymphNodeClsHead'),
                  LVI=dict(num_classes=1, in_channels=1, task_idx=1, loss_weight=0.5,
                           loss=dict(type='CrossEntropyLoss', use_sigmoid=True), type='LVIClsHead'),
                  multifocality=dict(num_classes=1, in_channels=1, task_idx=2, loss_weight=0.5,
                                     loss=dict(type='CrossEntropyLoss', use_sigmoid=True), type='MultifocalityClsHead'),
                  NumPos=dict(num_classes=1,in_channels=1,task_idx=3,loss_weight=0.5,
                              loss=dict(type='MSELoss'),type='NposRegHead'),
                  tumor_size=dict(num_classes=1,in_channels=1,task_idx=4,loss_weight=0.5,
                              loss=dict(type='MSELoss'),type='TsizeRegHead'),
              )),
)

# >>>>>>>>>>>>>>> Override schedules settings here >>>>>>>>>>>>>>>>>>>

# optimizer
optim_wrapper = dict(
    # optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.001),
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.001,
        eps=1e-8,
        betas=(0.9, 0.999)),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         # 'backbone.layer0': dict(lr_mult=0, decay_mult=0),
    #         # 'backbone': dict(lr_mult=1),
    #         'neck': dict(lr_mult=0.1)
    #     }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning policy
param_scheduler = dict(type='StepLR',step_size=1, by_epoch=True, gamma=0.95)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)


# >>>>>>>>>>>>>>> Override default_runtime settings here >>>>>>>>>>>>>>>>>>>
log_processor=dict(window_size=20)
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

