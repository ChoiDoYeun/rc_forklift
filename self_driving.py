import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit
import cv2
import numpy as np
import time

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

# OpenCV DNN 모듈로 ONNX 모델 로드
onnx_model_path = '0921_newtrack.onnx'
net = cv2.dnn.readNetFromONNX(onnx_model_path)

# OpenCV DNN 백엔드 및 타겟 설정
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 서보모터 설정
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 85  # 스티어링 휠 서보모터 중립 (채널 0)
kit.servo[1].angle = 90  # 첫 번째 카메라 서보모터 초기 설정 (채널 1)
kit.servo[2].angle = 110  # 두 번째 카메라 서보모터 초기 설정 (채널 2)

# 서보모터 각도 설정 (클래스별)
class_angles = {
    0: 85,   # 중립
    1: 130,  # 좌회전
    2: 50    # 우회전
}

# 서보모터 각도 제어 함수
def set_servo_angle(predicted_class):
    angle = class_angles.get(predicted_class, 85)  # 기본값은 중립(85도)
    kit.servo[0].angle = angle

# 이미지 전처리 함수 수정
def preprocess_image(image):
    # 이미지 크기 조정
    resized_image = cv2.resize(image, (64, 64))
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # 이미지 정규화: 0 ~ 255 범위를 0 ~ 1 범위로 변환
    gray_image = gray_image.astype(np.float32) / 255.0
    # Normalize(mean=[0.5], std=[0.5]) 적용
    gray_image = (gray_image - 0.5) / 0.5
    # 채널 차원 추가 (C, H, W 형태)
    gray_image = np.expand_dims(gray_image, axis=0)
    # 배치 차원 추가 (N, C, H, W 형태)
    blob = np.expand_dims(gray_image, axis=0)
    return blob

# 실시간 예측 함수
def predict_steering(image):
    # 이미지 전처리
    blob = preprocess_image(image)
    # 모델 예측
    net.setInput(blob)
    output = net.forward()
    # 결과 해석
    probs = output.flatten()
    predicted_class = np.argmax(probs)
    # 예측 결과 출력 (디버깅용)
    print(f"Predicted class: {predicted_class}, Probabilities: {probs}")
    return predicted_class

# 모터 주행 시작
motor1.forward()
motor2.forward()

try:
    # 카메라 캡처 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)  # 해상도 낮추기
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)

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
