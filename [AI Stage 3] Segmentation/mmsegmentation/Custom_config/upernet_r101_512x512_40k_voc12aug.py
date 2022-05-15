_base_ = './upernet_r50_512x512_40k_voc12aug.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
# python tools/train.py configs/my_config/upernet_r101_512x512_40k_voc12aug.py   
