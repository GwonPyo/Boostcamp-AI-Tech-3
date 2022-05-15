_base_ = [
    './models/upernet_vit-b.py', './datasets/dataset.py',
    './default_runtime.py', './schedules/schedule.py'
]
model = dict(
    decode_head=dict(num_classes=11), 
    # auxiliary_head=dict(num_classes=11)
    )
# python tools/train.py configs/my_config/upernet_vit-b.py 