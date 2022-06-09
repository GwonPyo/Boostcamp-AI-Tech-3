# [AI Stage 1] Object Detection

## Test 1. Base Model vs. Label Smoothing

| Model | Public_mAP | Private_mAP |
| --- | --- | --- |
| Base | 0.4348 | 0.4083 |
| Label Smoothing | 0.4308 |  0.4061 |
| LS + Hyperparameter Tuning | 0.4623 | 0.4385 |
| LS + HT + TTA | 0.5017 | 0.4811 |

Result: I want to use Label Smoothing(LS) for improving model's generalization. 
<br> But Public_mAP and Private_mAP are reduced slightly. 
<br> My Team think there is still room for experimentation. 
<br> So after I take experiments of Yolo-V5 applied label smoothing and other team member conduct experiments of Yolo-V5 not applied label smoothing.
<br> And I use Hyperparameter Tuning(HT) which is called Hyperparameter Evolution and Test Time Augmentation(TTA). 
<br> Both Methods increase model's performance. 

 