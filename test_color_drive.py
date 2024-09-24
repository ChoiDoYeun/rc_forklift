import RPi.GPIO as GPIO
import time
import cv2
import csv
import os
from adafruit_servokit import ServoKit
import threading
import pygame  # 조이스틱 입력 처리
import numpy as np

# GPIO 설정
GPIO.setmode(GPIO.BCM)

# 모터 제어 클래스 (기존 코드)
class MotorController:
    def __init__(self, en, in1, in2):
        self.en = en
        self.in1 = in1
        self.in2 = in2
        GPIO.setup(self.en, GPIO.OUT)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        self.pwm = GPIO.PWM(self.en, 100)
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
speed = 100
frame_time = 0.1

# PCA9685 모듈 초기화 (서보모터)
kit = ServoKit(channels=16)
kit.servo[0].angle = 90  # 서보모터 중립 (채널 0)
kit.servo[1].angle = 85
kit.servo[2].angle = 90

# 조이스틱 초기화
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# 각종 변수 및 플래그
initial_color = None
waiting_time = 0.5  # 모터가 동작한 후 대기 시간
last_frame_number = 0  # 마지막으로 읽은 프레임 번호
stop_servo_event = threading.Event()  # 서보모터를 중지하는 이벤트
resume_lock = threading.Lock()  # 서보모터 재개 시 락
color_detection_active = True  # 색상 감지 활성화 여부

# 초기 색상 감지 함수 (기존 코드)
def detect_initial_color():
    global initial_color
    cap = cv2.VideoCapture(0)
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
                    cap.release()
                    return

# 서보모터 제어 함수 (CSV 파일 읽기)
def control_servo_from_csv(start_frame=0):
    global last_frame_number
    predicted_servo_angle_path = 'drive_data.csv'  # CSV 파일 경로

    with open(predicted_servo_angle_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        frame_count = 0
        for row in reader:
            with resume_lock:
                if frame_count < start_frame:
                    frame_count += 1
                    continue
                if stop_servo_event.is_set():
                    last_frame_number = frame_count
                    print(f"Servo stopped at frame {last_frame_number}")
                    return
            servo_angle = float(row[1])  # 서보 각도 값
            kit.servo[0].angle = servo_angle  # 서보모터에 각도 적용
            print(f"Frame {frame_count}: Servo Angle Set to {servo_angle}")
            time.sleep(frame_time)
            frame_count += 1

        last_frame_number = frame_count  # 마지막 프레임 번호 저장
        motor1.stop()
        motor2.stop()
        print("CSV 파일의 모든 프레임을 읽었으므로 모터가 정지됩니다.")

# 색상 감지 및 모터 제어 함수 (기존 코드와 유사)
def color_based_motor_control():
    global color_detection_active
    while color_detection_active:
        # 조이스틱 버튼 상태 확인
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                # 조이스틱 버튼 11로 모터 시작
                if joystick.get_button(11):
                    print("Joystick button 11 pressed: Motor started")
                    motor1.forward(speed)
                    motor2.forward(speed)
                    color_detection_active = False
                    return

                # 조이스틱 버튼 10으로 서보모터 재개
                elif joystick.get_button(10):
                    print(f"Joystick button 10 pressed: Resuming from frame {last_frame_number + 1}")
                    motor1.forward(speed)
                    motor2.forward(speed)
                    stop_servo_event.clear()
                    servo_thread = threading.Thread(target=control_servo_from_csv, args=(last_frame_number + 1,))
                    servo_thread.start()
                    color_detection_active = False
                    return

# 메인 실행 로직
try:
    # 초기 색상 감지
    detect_initial_color()

    # 조이스틱 버튼 대기 (색상 감지 후)
    while color_detection_active:
        color_based_motor_control()

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    motor1.cleanup()
    motor2.cleanup()
    GPIO.cleanup()
    pygame.quit()
    print("Program terminated.")
