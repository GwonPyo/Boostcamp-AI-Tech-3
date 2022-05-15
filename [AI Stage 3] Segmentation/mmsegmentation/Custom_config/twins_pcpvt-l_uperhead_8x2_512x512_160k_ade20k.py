_base_ = ['./twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_large_20220308-37579dc6.pth'  # noqa

model = dict(
    backbone=dict(
        # decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        depths=[3, 8, 27, 3],
        drop_path_rate=0.3))
data = dict(samples_per_gpu=2, workers_per_gpu=2)
# python tools/train.py configs/my_config/twins_pcpvt-l_uperhead_8x2_512x512_160k_ade20k.py