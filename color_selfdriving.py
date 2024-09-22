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
                print(f"Frame {frame_count}: Servo Angle Set to {servo_angle}")
                time.sleep(0.1)  # 0.1초 대기 (프레임당 0.1초)
            frame_count += 1
        last_frame_number = frame_count  # 마지막 프레임 번호 저장

# 색상 감지 및 모터 제어 함수
def color_based_motor_control():
    global last_frame_number, stop_servo
    start_time = time.time()  # 모터가 동작한 시간을 기록

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
                    # 모터가 시작한 후 대기 시간을 지나야 색상을 확인
                    elapsed_time = time.time() - start_time
                    if elapsed_time > waiting_time:
                        # 감지된 색상이 처음 감지한 색상과 같으면 멈춤
                        if color_name == initial_color:
                            motor1.stop()
                            motor2.stop()
                            kit.servo[0].angle = 90  # 서보모터 중립
                            stop_servo = True  # 서보모터 중지 플래그 설정
                            print(f"Initial color {color_name} detected again. Motors and servo stopped.")
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
        # 서보모터 앵글 제어 스레드 시작 (비동기 실행)
        stop_servo = False  # 서보모터가 동작할 수 있도록 플래그 초기화
        servo_thread = threading.Thread(target=control_servo_from_csv, args=(last_frame_number,))
        servo_thread.start()
        
        # 모터 동작 시작 후 색상 감지 및 제어
        motor1.forward(40)  # 모터 동작 시작
        motor2.forward(40)  # 모터 동작 시작
        color_based_motor_control()

        # 서보모터 제어 스레드 종료 대기
        servo_thread.join()

    # 색상 감지 후 사용자 입력을 기다려서 서보모터 제어 재개
    while True:
        command = input("Enter 'resume' to continue servo operation: ")
        if command.lower() == 'resume':
            print("Resuming servo operation from frame:", last_frame_number)
            stop_servo = False  # 서보모터 동작 플래그 초기화
            servo_thread = threading.Thread(target=control_servo_from_csv, args=(last_frame_number,))
            servo_thread.start()
            servo_thread.join()

finally:
    # 종료 시 모터와 GPIO 정리
    motor1.cleanup()
    motor2.cleanup()
    cap.release()
    GPIO.cleanup()
    print("Program terminated.")
