_base_ = [
    'models/twins_pcpvt-s_upernet.py',
    'datasets/dataset.py',
    'default_runtime.py', 'schedules/schedule.py'
]

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(custom_keys={
#         'pos_block': dict(decay_mult=0.),
#         'norm': dict(decay_mult=0.)
#     }))

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)
model = dict(decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))
# data = dict(samples_per_gpu=16)
# python tools/train.py configs/my_config/twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k.py