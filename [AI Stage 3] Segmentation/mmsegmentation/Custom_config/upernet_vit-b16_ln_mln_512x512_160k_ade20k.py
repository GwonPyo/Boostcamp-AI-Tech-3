_base_ = [
    './models/upernet_vit-b16_ln_mln.py',
    './datasets/dataset.py',
    './default_runtime.py', './schedules/schedule.py'
]

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_embed': dict(decay_mult=0.),
#             'cls_token': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
# python tools/train.py configs/my_config/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py 
# python tools/model_converters/vit2mmseg.py pretrain/jx_vit_base_p16_224-80ecf9dd.pth pretrain/vit_base_patch16_224.pth