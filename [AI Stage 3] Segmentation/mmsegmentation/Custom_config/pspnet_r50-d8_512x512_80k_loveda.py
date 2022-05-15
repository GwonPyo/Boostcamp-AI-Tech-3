_base_ = [
    './models/pspnet_r50-d8.py', './datasets/dataset.py',
    './default_runtime.py', './schedules/schedule.py'
]
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))
