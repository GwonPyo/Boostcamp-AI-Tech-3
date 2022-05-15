_base_ = ['./twins_svt-s_uperhead_8x2_512x512_160k_ade20k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_large_20220308-fb5936f3.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512))
# python tools/train.py configs/my_config/twins_svt-l_uperhead_8x2_512x512_160k_ade20k.py --load-from work_dirs/twins_svt-l_offline/best_mIoU_epoch_33.pth