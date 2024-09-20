import os
import time
import torch
import cv2
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import mobilenet_v2

# 양자화된 모델 로드 및 백엔드 설정
torch.backends.quantized.engine = 'qnnpack'  # 양자화 엔진 설정 ('fbgemm'도 가능)

# 모델 정의 (양자화된 MobileNetV2 사용)
class selfdrivingCNN(nn.Module):
    def __init__(self):
        super(selfdrivingCNN, self).__init__()
        
        # MobileNetV2 백본을 사용 (양자화 지원)
        self.backbone = mobilenet_v2(pretrained=True)
        
        # MobileNetV2의 출력 클래스를 3개로 수정
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 3)  # 클래스 수 3개
        )

    def forward(self, x):
        # MobileNetV2를 통과한 결과 리턴
        x = self.backbone(x)
        return x

# 장치 설정 (CPU 사용)
device = torch.device('cpu')

# 양자화된 모델 로드
model = selfdrivingCNN().to(device)
model.load_state_dict(torch.load('quantized_model.pth', map_location=device))
model.eval()  # 양자화된 모델은 반드시 평가 모드로 전환

# 이미지 전처리 (양자화된 모델에서도 float32 입력을 받음)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 경로
image_folder = 'drive_00005'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# 추론 시간 측정 및 FPS 계산
total_time = 0
num_images = len(image_files)

for image_file in image_files:
    # 이미지 로드 및 전처리
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

    # 추론 시간 측정 시작
    start_time = time.time()

    # 모델 예측 (양자화된 모델로 예측)
    with torch.no_grad():
        outputs = model(image)

    # 추론 시간 측정 종료
    end_time = time.time()
    inference_time = end_time - start_time
    total_time += inference_time

    print(f"이미지 {image_file} 추론 시간: {inference_time:.4f}초")

# FPS 계산
average_time_per_image = total_time / num_images
fps = 1 / average_time_per_image

print(f"평균 FPS: {fps:.2f}")
