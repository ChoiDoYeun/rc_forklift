import RPi.GPIO as GPIO
import time
import cv2
import csv
import os
from adafruit_servokit import ServoKit
import threading
import numpy as np

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

# 모터 초기화
motor1 = MotorController(18, 17, 27)
motor2 = MotorController(16, 13, 26)

# PCA9685 모듈 초기화 (서보모터)
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 90  # 스티어링 휠 서보모터 중립 (채널 0)
kit.servo[1].angle = 85
kit.servo[2].angle = 110

# 각 색상의 HSV 범위 정의
colors = {
    "red": [(0, 120, 70), (10, 255, 255)],
    "blue": [(110, 50, 50), (130, 255, 255)],
    "yellow": [(20, 100, 100), (30, 255, 255)],
    "black": [(0, 0, 0), (180, 255, 50)]
}

# 카메라 설정
cap = cv2.VideoCapture(0)
initial_color = None
waiting_time = 0.5  # 모터가 동작한 후 처음 색상 감지를 무시할 시간 (초)
last_frame_number = 0  # 마지막으로 읽은 프레임 번호를 저장
stop_servo = False  # 서보모터를 중지하는 플래그

# 초기 색상 감지 함수
def detect_initial_color():
    global initial_color
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 500:  # 일정 크기 이상일 경우
                    initial_color = color_name
                    print(f"Initial color detected: {color_name}")
                    return

# 사용자 명령을 입력받는 함수
def wait_for_start_command():
    command = input("Enter 'start' to begin motor operation: ")
    if command.lower() == 'start':
        print("Motor started!")
        return True
    return False

# 서보 앵글 제어 함수 (CSV 파일에서 각도 불러오기)
def control_servo_from_csv(start_frame=0):
    global last_frame_number, stop_servo
    predicted_servo_angle_path = 'v2_predicted_servo_angle.csv'  # 예측 CSV 파일 경로

    with open(predicted_servo_angle_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        frame_count = 0
        for row in reader:
            if frame_count >= start_frame:
                if stop_servo:
                    last_frame_number = frame_count  # 서보모터가 중지된 시점의 프레임 번호 저장
                    print(f"Servo stopped at frame {last_frame_number}")
                    break  # 서보모터를 멈추고 중지
                servo_angle = int(row[0])  # 서보 각도 값
                kit.servo[0].angle = servo_angle  # 서보모터에 각도 적용
                print(f"Frame {frame_count}: Servo Angle Set to {
