import os
import time
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn

# 모델 정의 (학습에서 사용한 모델과 동일해야 함)
class selfdrivingCNN(nn.Module):
    def __init__(self):
        super(selfdrivingCNN, self).__init__()
        
        # ResNet50 백본을 사용, 사전 학습된 가중치 로드
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # ResNet의 마지막 FC 레이어를 제거하고 추가 레이어 구성
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 새로운 FC 레이어들 추가
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # 3개의 클래스 출력
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_layers(x)
        return x

# 장치 설정 (CPU 사용)
device = torch.device('cpu')

# 모델 로드
model = selfdrivingCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()  # 평가 모드로 전환

# 이미지 전처리 (모델 학습 시와 동일한 전처리 적용)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 경로 (drive_00001 폴더 내 PNG 파일)
image_folder = 'drive_00001'
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

    # 모델 예측
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

