_base_ = [
    '../../_base_/models/resnet3d_2s_r50.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='Recognizer3D2sLFAtt',
    backbone=dict(
        type='ResNet3dSlowOnly2sLFAtt',
        full_pathway=dict(
            pretrained2d=False,
            pretrained='pretrained/epoch_55_changekey.pth',
        ),
        local_pathway=dict(
            pretrained2d=False,
            pretrained='pretrained/epoch_100.pth',)),
    cls_head=dict(
        num_classes=1067))

# dataset settings
dataset_type = 'LGRawframeDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root = 'Datas/NMFs-CSL_video/jpg_video'
data_root_val = 'Datas/NMFs-CSL_video/jpg_video'
data_root_test = 'Datas/NMFs-CSL_video/jpg_video'
ann_file_train = 'Datas/NMFs-CSL_video/NMFs_CSL_train_split_rgb.txt'
ann_file_val = 'Datas/NMFs-CSL_video/NMFs_CSL_test_split_rgb.txt'
ann_file_test = 'Datas/NMFs-CSL_video/NMFs_CSL_test_split_rgb.txt'
ann_skeleton_train = 'Datas/NMFs-CSL_video/train_data_joint_r_npy.npy'
ann_skeleton_val = 'Datas/NMFs-CSL_video/test_data_joint_r_npy.npy'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='SaveOriginImgs'),
    dict(type='RandomResizedCrop', lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.01, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FuseLocal'),
    dict(type='SkeletonGuide8Crop', box_scale=1.5),
    dict(type='ResizeLocalImgs', scale=(224, 224), keep_ratio=False),
    dict(type='NormalizeLocalImgs', **img_norm_cfg),
    dict(type='CreateCropBoxes'),
    dict(type='FormatShapeLocalImgs', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'local_imgs', 'label', 'all_crop_boxes'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'local_imgs', 'label', 'all_crop_boxes'])
]


val_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='SaveOriginImgs'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='CenterCrop', crop_size=224, lazy=True),
    dict(type='Fuse'),
    dict(type='SkeletonGuide8Crop', box_scale=1.5),
    dict(type='ResizeLocalImgs', scale=(224, 224), keep_ratio=False),
    dict(type='NormalizeLocalImgs', **img_norm_cfg),
    dict(type='CreateCropBoxes'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FormatShapeLocalImgs', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'local_imgs', 'label', 'all_crop_boxes'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'local_imgs', 'label', 'all_crop_boxes'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=10, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='SkeletonGuide8Crop', box_scale=1.5),
    dict(type='ResizeLocalImgs', scale=(224, 224), keep_ratio=False),
    dict(type='NormalizeLocalImgs', **img_norm_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='FormatShapeLocalImgs', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'local_imgs', 'label', 'all_crop_boxes'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'local_imgs', 'label', 'all_crop_boxes'])
]
data = dict(
    videos_per_gpu=5,
    workers_per_gpu=5,
    test_dataloader=dict(videos_per_gpu=1),
    val_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        ann_skeleton_file=ann_skeleton_train,
        data_prefix=data_root,
        filename_tmpl='image_{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        ann_skeleton_file=ann_skeleton_val,
        data_prefix=data_root_val,
        filename_tmpl='image_{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        ann_skeleton_file=ann_skeleton_val,
        data_prefix=data_root_val,
        filename_tmpl='image_{:05}.jpg',
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.08, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', warmup='linear', warmup_by_epoch=True, warmup_iters=5, step=[60,75] )
total_epochs = 80

# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(interval=200)
workflow = [('train', 5), ('val', 1)]

work_dir = './work_dirs/slowonly_2s_full_local_r50_32x2x1_80e_NMFs_CSL_rgb'
find_unused_parameters = False
