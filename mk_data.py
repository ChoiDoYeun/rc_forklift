import RPi.GPIO as GPIO
import pygame
import time
import cv2  # OpenCV 사용
import csv
import os  # 폴더 생성에 사용
from adafruit_servokit import ServoKit

# GPIO 설정
GPIO.setmode(GPIO.BCM)

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

    def forward(self, speed):
        self.set_speed(speed)
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)

    def backward(self, speed):
        self.set_speed(speed)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)

    def stop(self):
        self.set_speed(0)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)

    def cleanup(self):
        self.pwm.stop()

# 모터 초기화 (왼쪽 모터: motor1, 오른쪽 모터: motor2)
motor1 = MotorController(18, 17, 27)  # 모터1: en(18), in1(17), in2(27)
motor2 = MotorController(16, 13, 26)  # 모터2: en(16), in1(13), in2(26)

# PCA9685 모듈 초기화 (서보모터)
kit = ServoKit(channels=16)

# 서보모터 초기 설정 (스티어링 휠, 채널 0 사용)
kit.servo[0].angle = 90  # 스티어링 휠 서보모터 중립 (채널 0)

# 카메라 조향용 서보모터 (채널 1과 채널 2 사용)
kit.servo[1].angle = 90  # 첫 번째 카메라 서보모터 초기 설정 (채널 1)
kit.servo[2].angle = 45  # 두 번째 카메라 서보모터 초기 설정 (채널 2)

# Pygame 초기화
pygame.init()

# 조이스틱 초기화
pygame.joystick.init()

# 첫 번째 조이스틱을 선택
joystick = pygame.joystick.Joystick(0)
joystick.init()

# 카메라 초기화 (OpenCV 사용)
cap = cv2.VideoCapture(0)  # 카메라 장치 선택

# 주행 폴더 생성
def create_drive_folder():
    drive_number = 1
    while True:
        folder_name = f'drive_{drive_number:05d}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        drive_number += 1

# 새로운 주행 폴더 생성
drive_folder = create_drive_folder()

# CSV 파일 작성 준비
csv_file_path = os.path.join(drive_folder, 'drive_data.csv')
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'servo_angle', 'motor_speed'])  # CSV 헤더 작성

# 서보모터 각도 초기값 설정
servo_angle = 90  # 스티어링 서보모터 중립
speed = 0  # 초기 속도 값 선언
frame_count = 0  # 프레임 카운트

# 메인 루프
running = True
while running:
    for event in pygame.event.get():
        # 버튼 10을 눌러 정지
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 10:  # 버튼 9: 정지 버튼
                speed = 0  # 속도 정지
                motor1.stop()
                motor2.stop()
                print("정지 버튼을 눌렀습니다. 속도: 0")

        # 종료 이벤트 처리
        elif event.type == pygame.QUIT:
            running = False

    # 카메라에서 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # 프레임 크기 조정 (256x256)
    frame_resized = cv2.resize(frame, (256, 256))

    # 좌측 스틱 (스티어링 휠 서보모터 각도 제어: 축 0)
    axis_value = joystick.get_axis(0)  # 축 0: 좌측 스틱 좌우 움직임
    servo_angle = (1 - axis_value) * 35 + 55  # 각도를 55 ~ 125도 범위로 변환
    servo_angle = max(55, min(125, servo_angle))  # 각도를 55 ~ 125도로 제한
    kit.servo[0].angle = servo_angle  # 스티어링 서보모터 각도 설정
    print(f"스티어링 각도: {servo_angle}")

    # 우측 스틱 위아래 (속도 제어: 축 3)
    axis_value = joystick.get_axis(3)  # 축 3: 우측 스틱 위아래
    if axis_value < -0.1:  # 스틱을 위로 올리면
        speed = min(max(speed + 1, 50), 100)  # 속도 증가, 최소 50, 최대 100%
        motor1.forward(speed)
        motor2.forward(speed)
        print(f"속도 상승: {speed}")
    elif axis_value > 0.1:  # 스틱을 아래로 내리면
        speed = max(speed - 1, 0)  # 속도 감소, 최소 0%
        motor1.forward(speed)
        motor2.forward(speed)
        print(f"속도 감소: {speed}")

    # 프레임 저장
    frame_filename = f'frame_{frame_count:05d}.png'
    frame_path = os.path.join(drive_folder, frame_filename)
    cv2.imwrite(frame_path, frame_resized)

    # CSV 파일에 데이터 저장
    csv_writer.writerow([frame_filename, servo_angle, speed])

    frame_count += 1
    time.sleep(0.1)  # 0.1초마다 프레임 저장 및 상태 기록

# Pygame 종료
pygame.quit()

# 카메라와 CSV 파일 정리
cap.release()
csv_file.close()

# GPIO 정리
GPIO.cleanup()
