import RPi.GPIO as GPIO
import pygame
import time
import cv2  # OpenCV 사용
import csv
import os  # 폴더 생성에 사용
from adafruit_servokit import ServoKit
import threading

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

# 서보모터 초기 설정
kit.servo[0].angle = 90  # 스티어링 휠 서보모터 중립 (채널 0)
kit.servo[1].angle = 85
kit.servo[2].angle = 90

# 모터를 앞으로 움직이기 (속도 40으로 설정)
motor1.forward(speed=100)
motor2.forward(speed=100)

# predicted_servo_angle.csv 파일에서 각 프레임당 servo_angle 값을 불러오기
predicted_servo_angle_path = 'predicted_servo_angle.csv'  # 예측 CSV 파일 경로

with open(predicted_servo_angle_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 스킵
    frame_count = 0
    for row in reader:
        servo_angle = int(row[0])  # 서보 각도 값
        kit.servo[0].angle = servo_angle  # 서보모터에 각도 적용
        print(f"Frame {frame_count}: Servo Angle Set to {servo_angle}")
        time.sleep(0.1)  # 0.1초 대기 (프레임당 0.1초)
        frame_count += 1

# CSV 파일 끝까지 읽은 후 모터 정지
motor1.stop()
motor2.stop()

# 비디오 캡처 스레드 종료 요청
stop_event.set()
video_thread.join()

# 동작 종료 시 모터와 GPIO 정리
motor1.cleanup()
motor2.cleanup()
GPIO.cleanup()
