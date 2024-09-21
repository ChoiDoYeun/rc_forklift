import os
import time
import cv2
import numpy as np

# OpenCV DNN 모듈로 ONNX 모델 로드
onnx_model_path = '0921_newtrack.onnx'
net = cv2.dnn.readNetFromONNX(onnx_model_path)

# OpenCV DNN 백엔드 및 타겟 설정 (필요에 따라 설정)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 테스트할 이미지 폴더 경로 설정
image_folder = 'drive_00001'

# 이미지 전처리 함수 (OpenCV 사용)
def preprocess_image(image):
    # 이미지 크기 조정
    resized_image = cv2.resize(image, (64, 64))
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # 이미지 정규화: 0 ~ 255 범위를 0 ~ 1 범위로 변환
    gray_image = gray_image / 255.0
    # Normalize(mean=0.5, std=0.5) 적용
    gray_image = (gray_image - 0.5) / 0.5
    # 채널 차원 추가 (C, H, W 형태로 만들기 위해)
    gray_image = np.expand_dims(gray_image, axis=0)
    # 배치 차원 추가 (N, C, H, W 형태로 만들기 위해)
    blob = np.expand_dims(gray_image, axis=0).astype(np.float32)
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

    # 이미지들을 미리 로드하여 리스트에 저장 (로딩 시간 제외를 위해)
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        images.append(image)

    # FPS 측정을 위한 시간 기록 시작
    start_time = time.time()

    for image in images:
        # 이미지 전처리
        blob = preprocess_image(image)
        if blob is None:
            continue

        # 추론 수행
        net.setInput(blob)
        output = net.forward()

    # 총 소요 시간 계산
    total_time = time.time() - start_time

    # FPS 계산
    fps = total_images / total_time if total_time > 0 else 0
    print(f"Total time taken for inference: {total_time:.4f} seconds")
    print(f"Inference FPS: {fps:.2f} frames per second")

# 이미지 폴더 내 이미지들을 추론하고 FPS 계산
calculate_fps(net, image_folder)
