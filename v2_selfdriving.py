import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 장치 설정 (GPU 또는 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
        self.set_speed(40)  # 모터 속도를 항상 40으로 설정
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)

    def stop(self):
        self.set_speed(0)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)

    def cleanup(self):
        self.pwm.stop()

# GPIO 초기화
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# 모터 초기화
motor1 = MotorController(18, 17, 27)  # 모터1: en(18), in1(17), in2(27)
motor2 = MotorController(16, 13, 26)  # 모터2: en(16), in1(13), in2(26)

# 모델 정의 (훈련 시 사용한 모델과 동일해야 함)
class selfdrivingCNN(nn.Module):
    def __init__(self):
        super(selfdrivingCNN, self).__init__()
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=self.backbone.features[0][0].out_channels,
            kernel_size=self.backbone.features[0][0].kernel_size,
            stride=self.backbone.features[0][0].stride,
            padding=self.backbone.features[0][0].padding,
            bias=False
        )
        self.remove_batch_norm_layers()  # 배치 정규화 레이어 제거
        num_ftrs = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(num_ftrs, 3)  # 출력 클래스 수를 3으로 변경

    def remove_batch_norm_layers(self):
        # 안전하게 배치 정규화 레이어 제거
        modules_to_replace = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                modules_to_replace.append(name)

        for name in modules_to_replace:
            parent_name = '.'.join(name.split('.')[:-1])
            last_name = name.split('.')[-1]
            parent_module = self.backbone.get_submodule(parent_name)
            setattr(parent_module, last_name, nn.Identity())

    def forward(self, x):
        return self.backbone(x)

# 모델 로드
model = selfdrivingCNN().to(device)
model.load_state_dict(torch.load('0921_newtrack.pth', map_location=device))
model.eval()

# 서보모터 설정
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 85  # 스티어링 휠 서보모터 중립 (채널 0)
kit.servo[1].angle = 90  # 첫 번째 카메라 서보모터 초기 설정 (채널 1)
kit.servo[2].angle = 110  # 두 번째 카메라 서보모터 초기 설정 (채널 2)

# 서보모터 각도 설정 (클래스별)
class_angles = {
    0: 85,   # 중립
    1: 120,  # 좌회전
    2: 50    # 우회전
}

# 서보모터 각도 제어 함수
def set_servo_angle(predicted_class):
    angle = class_angles.get(predicted_class, 85)  # 기본값은 중립(85도)
    kit.servo[0].angle = angle

# 이미지 전처리 함수 (훈련 과정과 동일하게)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image):
    # 이미지 크기 조정
    resized_image = cv2.resize(image, (64, 64))
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # PIL 이미지로 변환
    pil_image = Image.fromarray(gray_image, mode='L')
    # 변환 적용
    input_tensor = transform(pil_image)
    # 배치 차원 추가
    input_tensor = input_tensor.unsqueeze(0).to(device)
    return input_tensor

# 실시간 예측 함수
def predict_steering(image):
    # 이미지 전처리
    input_tensor = preprocess_image(image)
    # 모델 예측
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# 모터 주행 시작
motor1.forward()
motor2.forward()

try:
    # 카메라 캡처 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)  # 해상도 낮추기
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 예측된 클래스에 따라 서보모터 각도 조정
        predicted_class = predict_steering(frame)
        set_servo_angle(predicted_class)

        # 프레임 당 지연 시간 조정 (선택 사항)
        # time.sleep(0.01)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    # 리소스 해제
    cap.release()
    motor1.stop()
    motor2.stop()
    GPIO.cleanup()
    print("프로그램 종료")
