import threading
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 모터 제어 클래스
class MotorController:
    def __init__(self, en, in1, in2):
        self.en = en
        self.in1 = in1
        self.in2 = in2
        GPIO.setup(self.en, GPIO.OUT)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        self.pwm = GPIO.PWM(self.en, 100)  # PWM 주파수 100Hz
        self.pwm.start(0)

    def set_speed(self, speed):
        self.pwm.ChangeDutyCycle(speed)

    def forward(self):
        self.set_speed(50)  # 모터 속도를 항상 50으로 설정
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)

    def stop(self):
        self.set_speed(0)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)

    def cleanup(self):
        self.pwm.stop()

# 모터 초기화
motor1 = MotorController(18, 17, 27)  # 모터1: en(18), in1(17), in2(27)
motor2 = MotorController(16, 13, 26)  # 모터2: en(16), in1(13), in2(26)

# 모델 정의
class selfdrivingCNN(nn.Module):
    def __init__(self):
        super(selfdrivingCNN, self).__init__()
        # MobileNetV3 백본 사용, 사전 학습된 가중치 로드
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # 출력 클래스를 3개로 수정
        num_ftrs = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.backbone(x)

# 장치 설정 (CPU 사용)
device = torch.device('cpu')

# 모델 로드
model = selfdrivingCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()  # 평가 모드로 전환

# 모델 양자화 적용
from torch.quantization import quantize_dynamic
model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

# 서보모터 설정
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 85  # 스티어링 휠 서보모터 중립 (채널 0)
kit.servo[1].angle = 60  # 첫 번째 카메라 서보모터 초기 설정 (채널 1)
kit.servo[2].angle = 80  # 두 번째 카메라 서보모터 초기 설정 (채널 2)

# 서보모터 각도 설정 (클래스별)
class_angles = {
    0: 85,   # 중립
    1: 55,   # 우회전
    2: 125   # 좌회전
}

# 전역 변수로 frame 선언
frame = None

# 이미지 전처리 함수 (OpenCV 사용)
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # 이미지 크기 조정
    image = image.astype(np.float32) / 255.0  # [0,1] 범위로 정규화
    # 데이터 타입을 np.float32로 지정
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)
    image = (image - mean) / std  # 정규화
    image = np.transpose(image, (2, 0, 1))  # 채널 순서 변경 (HWC -> CHW)
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    image = torch.from_numpy(image).to(device)
    return image

# 실시간 예측 함수
def predict_steering(image):
    # 이미지 전처리
    input_tensor = preprocess_image(image)

    # 모델 예측
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

# 서보모터 각도 제어 함수
def set_servo_angle(predicted_class):
    angle = class_angles[predicted_class]
    kit.servo[0].angle = angle

# 카메라 프레임 캡처 함수 (스레드에서 실행)
def capture_camera():
    global frame
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽을 수 없습니다.")
            break
    cap.release()

# 카메라 스레드 시작
camera_thread = threading.Thread(target=capture_camera)
camera_thread.start()

# 실시간 예측 루프 (모터 제어)
try:
    # 모터 주행 시작
    motor1.forward()
    motor2.forward()

    while True:
        if frame is not None:
            # 예측된 클래스에 따라 서보모터 각도 조정
            predicted_class = predict_steering(frame)
            set_servo_angle(predicted_class)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # 카메라 및 GPIO 정리
    motor1.stop()
    motor2.stop()
    GPIO.cleanup()
    print("프로그램 종료")
