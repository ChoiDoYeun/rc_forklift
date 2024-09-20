import threading
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit
import cv2
import numpy as np

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
        self.set_speed(40)  # 모터 속도를 항상 50으로 설정
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
onnx_model_path = 'best_model_pruned.onnx'
net = cv2.dnn.readNetFromONNX(onnx_model_path)

# OpenCV DNN 백엔드 및 타겟 설정 (라즈베리 파이에서는 CPU 사용)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 서보모터 설정
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 85  # 스티어링 휠 서보모터 중립 (채널 0)
kit.servo[1].angle = 60  # 첫 번째 카메라 서보모터 초기 설정 (채널 1)
kit.servo[2].angle = 80  # 두 번째 카메라 서보모터 초기 설정 (채널 2)

# 서보모터 각도 설정 (클래스별)
class_angles = {
    0: 85,   # 중립
    1: 130,   # 우회전
}

# 전역 변수로 frame 선언
frame = None

# 이미지 전처리 함수 (OpenCV 사용)
def preprocess_image(image):
    # 이미지 크기 조정
    resized_image = cv2.resize(image, (64, 64))
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # blob 생성
    blob = cv2.dnn.blobFromImage(gray_image, scalefactor=1/255.0, size=(64, 64))
    # Normalize(mean=0.5, std=0.5) 적용
    blob = (blob - 0.5) / 0.5
    return blob

# 실시간 예측 함수
def predict_steering(image):
    # 이미지 전처리
    blob = preprocess_image(image)
    # 모델 예측
    net.setInput(blob)
    output = net.forward()
    # 결과 해석
    probs = output[0]
    predicted_class = np.argmax(probs)
    return predicted_class

# 서보모터 각도 제어 함수
def set_servo_angle(predicted_class):
    angle = class_angles.get(predicted_class, 85)  # 기본값은 중립(85도)
    kit.servo[0].angle = angle

# 카메라 프레임 캡처 함수 (스레드에서 실행)
def capture_camera():
    global frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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

        # ESC 키를 누르면 종료 (라즈베리 파이에서는 필요 없을 수 있음)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # 카메라 및 GPIO 정리
    motor1.stop()
    motor2.stop()
    GPIO.cleanup()
    print("프로그램 종료")
