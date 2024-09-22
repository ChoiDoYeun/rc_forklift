import RPi.GPIO as GPIO
import pygame
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
kit.servo[0].angle = 90  # 스티어링 휠 서보모터 중립 (채널 0)

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

# 색상 감지 및 모터 제어 함수
def color_based_motor_control():
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
                if cv2.contourArea(largest_contour) > 500:
                    # 감지된 색상이 처음 감지한 색상과 같으면 멈춤
                    if color_name == initial_color:
                        motor1.stop()
                        motor2.stop()
                        print(f"Initial color {color_name} detected again. Motors stopped.")
                        return
                    else:
                        # 다른 색상이 감지되면 계속 주행
                        motor1.forward(40)
                        motor2.forward(40)
                        print(f"{color_name} detected. Motors running...")

# 메인 실행 로직
try:
    # 초기 색상 감지
    detect_initial_color()

    # 사용자 명령 대기
    if wait_for_start_command():
        # 색상에 따라 모터 제어
        color_based_motor_control()

finally:
    # 종료 시 모터와 GPIO 정리
    motor1.cleanup()
    motor2.cleanup()
    cap.release()
    GPIO.cleanup()
    print("Program terminated.")
