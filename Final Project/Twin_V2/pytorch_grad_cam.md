아래 명령으로 pytorch-grad-cam을 설치한다. 

```
pip install grad-cam
```

해당 파일의 baseline 코드 아래에 실행하면 정상적으로 작동힌다.

먼저 학습한 모델을 불러온다.

```
model = timm.create_model('swinv2_large_window12to24_192to384_22kft1k', pretrained=True).to(device)
in_features = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(in_features, 5)
).to(device)

model.load_state_dict(torch.load('twin_v2_lr1e-4/model.pt'))
model.eval()
```
그리고 valid 이미지가 존재하는 파일에 접근해 이미지 파일들을 가져온다. 
valid_path 부분은 알맞게 변경해야 한다.

```
valid_path = 'dataset/valid/JPEGImages'
file_list = os.listdir(valid_path)
image_list = []

for file in file_list:
    if '.jpg' in file:
        image_list.append(file)
```

마지막으로 아래 코드를 실행시켜주면 된다.
```
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def reshape_transform(tensor, height=12, width=12): # 아래 설명한 에러 발생 시 height와 width를 수정하면 된다.
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layers = [model.layers[-1].blocks[-1].norm1] # 모델에 맞는 층으로 변경해야 한다.

cam = GradCAM(model=model, 
            target_layers=target_layers, 
            reshape_transform=reshape_transform, 
            use_cuda=torch.cuda.is_available())

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
rgb_img = cv2.imread(os.path.join(valid_path, image_list[10]), 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (384, 384)) # 이미지 크기에 맞게 수정해야 한다.
axs[0].imshow(rgb_img)
axs[0].axis('off')

rgb_img = np.float32(rgb_img) / 255

input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

grayscale_cam = cam(input_tensor=input_tensor,
                    targets=None,)

grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

axs[1].imshow(cam_image)
axs[1].axis('off')
plt.show()
```

위 코드에서 바꿔야 할 부분은 target_layer리스트 안에 있는  model.layers[-1].blocks[-1].norm1 이다. 
twin_v2의 경우 해당 층을 입력하면 cam을 그려주지만 모델마다 입력해야 할 값이 다르다.

* [Pytorch-Grad-Cam Gihub](https://github.com/jacobgil/pytorch-grad-cam)

어떤 층을 사용해야 하는지는 위 링크에서 참고하면 된다.


위 코드 실행 시 이미지 크기에 따라, target_layer의 위치에 따라 아래와 같은 에러가 발생할 수 있다.

* shape '[1, 12, 12, 768]' is invalid for input of size 442368

이때는 target_layer의 채널 수(위 경우 768)를 확인하고 size(442368)에 나눠주면 임의의 수에 대한 제곱수(ex) 24*24)가 나오게 된다. 이때 def reshape_transform(tensor, height=12, width=12): 부분의 height와 width를 알맞게 바꿔주면 된다. (위 경우 24로 변경)

GradCam 이외에 다른 방식을 사용하고 싶다면 cam = GradCAM 부분에서 GradCAM 부분을 바꿔주시면 된다.
단, EigenCAM, FullGrad는 작동 시 에러가 발생하며 ablationcam의 경우는 약간의 수정이 필요하다.