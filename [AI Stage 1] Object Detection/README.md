# [AI Stage 1] Object Detection

## The Used Model: Yolo-V5

My Team trained some models for ensemble and I take care of training yolo-v5. 
When I train the model, I used the Public Repo of [Yolo-V5](https://github.com/ultralytics/yolov5).

If you want to use yolo-v5 public repo, you can refer to my folder(Yolo_V5_Manual).

## Test 1. Base Model vs. Label Smoothing

| Model | Public_mAP | Private_mAP |
| :---: | :---: | :---: |
| Base | 0.4348 | 0.4083 |
| Label Smoothing | 0.4308 |  0.4061 |
| LS + Hyperparameter Tuning | 0.4623 | 0.4385 |
| LS + HT + TTA | 0.5017 | 0.4811 |

I want to use **Label Smoothing(LS)** for improving model's generalization. But Public_mAP and Private_mAP are reduced slightly. My Team think there is still room for experimentation. So I take experiments of yolo-v5 applied label smoothing and other team member conduct experiments of yolo-v5 not applied label smoothing. Then I use **Hyperparameter Tuning(HT)** which is called Hyperparameter Evolution and Test **Time Augmentation(TTA)** and I can check both methods increase model performance.

## Test 2. Multi-scale

| Model | Public_mAP | Private_mAP |
| :---: | :---: | :---: |
| Multi Scale (HT) | 0.4927 |  0.4613 |
| Multi Scale (LS + HT) | 0.4410 | 0.4195 |
| Multi Scale (LS + HT + TTA) | 0.4938 | 0.4758 |

I apply **Multi Scale** to yolo-v5. The performence of model which is not applied Label Smoothing is increased but the other is decreased. TTA method increases model performance as before. 

## Test 3. Pseudo Labeling
| Model | Public_mAP | Private_mAP |
| :---: | :---: | :---: |
| Pseudo Labeling (LS + HT) | 0.5171 | 0.4954 |

I use **Pseudo Labeling(PL)** and this method have a positive effect on model. 

## Test 4. Augmentation

I use **Augmentation Methods** like below.

```python
self.transform = A.Compose([
                A.OneOf([
                    A.Flip(p=1.0),
                    A.RandomRotate90(p=1.0)
                ]),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.15, p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.Blur(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(p=1.0)
                ], p=0.1),
                A.CLAHE(p=0.01)],  bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)
```

| Model | Public_mAP | Private_mAP |
| :---: | :---: | :---: |
| LS + HT | 0.5312 | 0.5001 |
| LS + HT + PL | 0.5214 | 0.5002 |

Augmentations improve model's generalization. But this time Pseudo Labeling(PL) does not have significant difference.

## Test 5. Yolov5x6

When I train models before, I used **Yolov5x** model. This model has faster training/inference time than more complicated models but lower mAP performance than those. So I change model bigger than Yolov5x. I apply Label Smoothing, Hyperparameter Tuning, TTA, Pseudo Labeling to this model.

| Model | Public_mAP | Private_mAP |
| :---: | :---: | :---: |
| Yolov5x6 | 0.5587 | 0.5392 |

## Test 6. Offline Augmentation

The Dataset has imbalance. So we use **Offline Augmentation** to solve this problem. We apply augmentations and make datas which is included insufficient class.

| Model | Public_mAP | Private_mAP |
| :---: | :---: | :---: |
| Offline Augmentation | 0.5636 | 0.5403 |

## Test 7. Ensemble

I conduct Ensemble three models below.

| Method | Yolov5x | Yolov5x6 | Yolov5x6
| :---: | :---: | :---: | :---: |
| Label Smoothing | o | o | o |
| Hyperparameter Tuning | o | o | o |
| Test Time Augmentation | o | o | o |
| Augmentation | x | o | o |
| Pseudo Labeling | x | x | o |
| Offline Augmentation | x | x | o |

| Model | Public_mAP | Private_mAP |
| :---: | :---: | :---: |
| Ensemble | 0.6003 | 0.5801 |

## Other Models' results

If you want to check other models' results please check Report Folder or click on the link below.

Team Repository: https://github.com/boostcampaitech3/level2-object-detection-level2-cv-18