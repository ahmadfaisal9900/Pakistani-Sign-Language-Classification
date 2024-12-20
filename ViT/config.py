# Example of a config file: configs/recognition/tsm/tsm_r50_video_1x1x8_50e_custom.py

_base_ = [
    '../../_base_/models/tsm_r50.py',
    '../../_base_/schedules/sgd_50e.py',
    '../../_base_/default_runtime.py'
]

# Modify dataset type and path
dataset_type = 'RawframeDataset'
data_root = 'E:/Projects/Unfinished/Sign Language/PkSLMNM'
data_root_val = 'E:/Projects/Unfinished/Sign Language/PkSLMNM'
ann_file_train = 'E:/Projects/Unfinished/Sign Language/annotations/train.txt'
ann_file_val = 'E:/Projects/Unfinished/Sign Language/annotations/val.txt'
ann_file_test = 'E:/Projects/Unfinished/Sign Language/annotations/test.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # Mean values used for pre-processing of images.
    std=[58.395, 57.12, 57.375],  # Std values used for pre-processing of images.
    to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline)
)

# Model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        depth=50,
        num_segments=8,
        pretrained='torchvision://resnet50',
        out_indices=(3,),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMHead',
        num_segments=8,
        num_classes=7,  # Change this to the number of classes in your dataset
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.001),
    # Change the neck type according to your needs.
    neck=dict(
        type='TPN',
        in_channels=2048,
        out_channels=256,
        spatial_modulation_config=dict(
            inplanes=2048,
            planes=256,
        ),
        temporal_modulation_config=dict(
            downsample_ratios=(1, 1, 1),
        ),
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(
            type='MaxPooling3d',
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0),
        level_fusion_config=dict(
            in_channels=[256, 256, 256],
            mid_channels=[256, 256, 256],
            out_channels=256,
            ds_scales=[(1, 1, 1), (1, 1, 1), (1, 1, 1)],
        ),
        aux_head_config=dict(
            in_channels=768,
            out_channels=256,
            inplanes=2304,
            spatial_type='avg',
            num_classes=7  # Change this to the number of classes in your dataset
        )),
    # Change the train_cfg and test_cfg according to your needs.
    train_cfg=None,
    test_cfg=dict(average_clips=None)
)

# Optimizer settings
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# Learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

# Runtime settings
checkpoint_config = dict(interval=5)
workflow = [('train', 1), ('val', 1)]
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
find_unused_parameters = False
