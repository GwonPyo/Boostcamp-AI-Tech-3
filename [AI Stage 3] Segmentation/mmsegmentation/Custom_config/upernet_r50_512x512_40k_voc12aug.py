_base_ = [
    '../_base_/models/upernet_r50.py',
    './datasets/dataset.py',
    './default_runtime.py', './schedules/schedule.py'
]
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))
# python tools/train.py configs/my_config/upernet_r50_512x512_40k_voc12aug.py   