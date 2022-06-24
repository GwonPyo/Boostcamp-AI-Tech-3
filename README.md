# Boostcamp-AI-Tech-3

## <h3> <strong> [AI Stage 1] Object Detection </strong> </h3>

Object Detection 대회의 테스크는 <b>이미지에서 쓰레기를 인식하고 인식한 쓰레기들을 10개의 클래스중 하나로 분류하는 것</b>이었습니다. <br>
사용했던 데이터 셋은 아래와 같습니다.

### 🗂️Dataset
- Train Images : 4883 images
- Test Images : 4871 images
- Class Names : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- Image Size : 1024x1024

### 🏆Result

아래의 팀 레파지토리에서 전체적인 결과를 확인할 수 있습니다.

Team Repository Link: https://github.com/boostcampaitech3/level2-object-detection-level2-cv-18

제가 맡았던 모델인 Yolov5의 결과는 Object Detection 폴더에서 확인할 수 있습니다.

---
<h3> <strong> [AI Stage 2] Data Production </strong> </h3>

<b>모델을 고정해 데이터만을 수정</b>하여 이미지 내에서 글자를 인식하는 모델의 성능을 올리는 대회였습니다. 
주어진 데이터 셋은 다음과 같습니다.

### 🗂️Dataset
Upstage_data : 1288 images

### 🏆Result
아래의 팀 레파지토리에서 전체적인 결과를 확인할 수 있습니다.

Team Repository Link: https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-18

제가 만든 augmentation 코드는 Data Production 폴더에서 확인할 수 있습니다.

---
<h3> <strong> [AI Stage 3] Segmentation </strong> </h3>

Segmentation 대회의 주제는 <b>이미지를 배경 및 여러 쓰레기 종류 총 11개의 클래스로 segmentation</b>하는 것이었습니다. 
<br> 데이터셋은 아래와 같습니다.  

### 🗂️Dataset
- Train Images : 3272 images
- Test Images : 624 images
- Class Names : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- Image Size : 512x512

### 🏆Result

아래의 팀 레파지토리에서 전체적인 결과를 확인할 수 있습니다.

Team Repository Link: https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-18

제가 사용했던 mmsegmentation 사용법 및 사용했던 config 파일들 그리고 해당 모델들의 결과는 Segmentation 폴더에서 확인할 수 있습니다.

---
<h3> <strong> Final Project </strong> </h3>

<b>사용자의 피부를 평가해주고 해당 평가들이 어떻게 나오게 된건지 Grad Cam을 통해 표시해주는 과제</b>를 진행했습니다. <br> artlab에서 제공된 데이터셋을 사용했으며 해당 데이터셋의 공개는 금지되었기 때문에 데이터셋에 대한 정보는 작성하지 않았습니다.

팀 레파지토리는 다음과 같으며 <b>시연 영상 및 데모 사이트 그리고 평가 항목들에 대한 전체적인 실험 결과</b>들을 확인할 수 있습니다..

Team Repository Link: https://github.com/boostcampaitech3/final-project-level3-cv-18

제가 사용했던 모델(Twin V2, Efficientnet) 및 Grad Cam 방법들은 Final Project 폴더에서 확인할 수 있습니다.
