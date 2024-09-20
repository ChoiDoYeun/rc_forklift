import os
import time
import cv2
import numpy as np

# OpenCV DNN 모듈로 ONNX 모델 로드
onnx_model_path = 'best_model_pruned.onnx'
net = cv2.dnn.readNetFromONNX(onnx_model_path)

# 테스트할 이미지 폴더 경로 설정
image_folder = 'data/drive_00001'

# 이미지 전처리 함수
def preprocess_image(image_path, target_size=(64, 64)):
    # 이미지 로드 (그레이스케일)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    # 이미지 리사이즈
    resized_image = cv2.resize(image, target_size)
    # 이미지 정규화 및 차원 변경 (OpenCV DNN은 입력 형식이 [N, C, H, W])
    blob = cv2.dnn.blobFromImage(resized_image, scalefactor=1/255.0, size=target_size)
    # 모델 학습 시 적용한 Normalize(mean=0.5, std=0.5)를 반영
    blob = (blob - 0.5) / 0.5
    return blob

# 추론 속도 계산 함수 (FPS 측정)
def calculate_fps(net, image_folder, num_images=0):
    # 이미지 파일 리스트 불러오기
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # 처리할 이미지 개수 설정 (전체 이미지 중 일부만 처리하고 싶을 경우)
    if num_images > 0:
        image_files = image_files[:num_images]
    
    total_images = len(image_files)
    print(f"Total images to process: {total_images}")

    # FPS 측정을 위한 시간 기록 시작
    start_time = time.time()

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # 이미지 전처리
        blob = preprocess_image(image_path)
        if blob is None:
            continue
        
        # 추론 수행
        net.setInput(blob)
        output = net.forward()

    # 총 소요 시간 계산
    total_time = time.time() - start_time
    
    # FPS 계산
    fps = total_images / total_time
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"FPS: {fps:.2f} frames per second")

# 이미지 폴더 내 이미지들을 추론하고 FPS 계산
calculate_fps(net, image_folder)
