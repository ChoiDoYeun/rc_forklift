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

    def forward(self, speed=40):
        self.set_speed(speed)
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)

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

kit.servo[1].angle = 85
kit.servo[2].angle = 110

# 모터를 앞으로 움직이기 (속도 40으로 설정)
motor1.forward(speed=40)
motor2.forward(speed=40)

# OpenCV를 사용하여 카메라에서 영상을 캡처하고 저장하기
camera = cv2.VideoCapture(0)  # 카메라 장치 열기 (0은 기본 카메라)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 해상도 설정 (640x640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
camera.set(cv2.CAP_PROP_FPS, 60)  # FPS 설정 (60fps)

# 동영상 저장 설정
video_filename = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (XVID)
out = cv2.VideoWriter(video_filename, fourcc, 60.0, (640, 640))  # 60fps, 640x640 해상도

# predicted_servo_angle.csv 파일에서 각 프레임당 servo_angle 값을 불러오기
predicted_servo_angle_path = 'v2_predicted_servo_angle.csv'  # 예측 CSV 파일 경로

with open(predicted_servo_angle_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 스킵
    frame_count = 0
    for row in reader:
        servo_angle = int(row[0])  # 서보 각도 값
        kit.servo[0].angle = servo_angle  # 서보모터에 각도 적용
        print(f"Frame {frame_count}: Servo Angle Set to {servo_angle}")
        
        # 카메라에서 프레임을 읽고 동영상에 저장
        ret, frame = camera.read()
        if ret:
            frame = cv2.resize(frame, (640, 640))  # 해상도 640x640으로 변환
            out.write(frame)  # 동영상에 프레임 쓰기
        else:
            print("카메라에서 프레임을 읽는 데 실패했습니다.")
            break
        
        time.sleep(1/60)  # 60fps에 맞춰 대기 (1초에 60프레임)

        frame_count += 1

# CSV 파일 끝까지 읽은 후 모터 정지
motor1.stop()
motor2.stop()

# 카메라와 동영상 저장 파일 해제
camera.release()
out.release()

# 동작 종료 시 모터와 GPIO 정리
motor1.cleanup()
motor2.cleanup()
GPIO.cleanup()

# OpenCV 윈도우 정리
cv2.destroyAllWindows()
