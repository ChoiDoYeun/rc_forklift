import time
import cv2
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit
import numpy as np
from collections import deque

# PCA9685 모듈 초기화 (16채널 서보 컨트롤러 사용)
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 0
kit.servo[1].angle = 60
kit.servo[2].angle = 0
kit.servo[3].angle = 0
kit.servo[4].angle = 0
kit.servo[5].angle = 80
kit.servo[6].angle = 120

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

    def backward(self, speed):
        self.set_speed(speed)
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)

    def forward(self, speed):
        self.set_speed(speed)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)

    def stop(self):
        self.set_speed(0)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)

    def cleanup(self):
        self.pwm.stop()

# GPIO 핀 설정
GPIO.setmode(GPIO.BCM)

motor1 = MotorController(18, 17, 27)
motor2 = MotorController(22, 23, 24)
motor3 = MotorController(12, 5, 6)
motor4 = MotorController(16, 13, 26)

def stop_motors():
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()

def go_forward():
    motor1.forward(40)  # 모터 속도 설정
    motor2.forward(20)
    motor3.forward(40)
    motor4.forward(20)

def turn_left():
    motor1.backward(50)  # 좌측 앞 모터 속도 낮춤
    motor2.forward(50)  # 우측 앞 모터 속도 높임
    motor3.backward(50)  # 좌측 뒤 모터 속도 낮춤
    motor4.forward(50)  # 우측 뒤 모터 속도 높임

def turn_right():
    motor1.forward(50)  # 좌측 앞 모터 속도 높임
    motor2.backward(50)     # 우측 앞 모터 멈춤
    motor3.forward(50)  # 좌측 뒤 모터 속도 높임
    motor4.backward(50)      # 우측 뒤 모터 멈춤

# 카메라 입력 처리 및 모터 제어 부분
def binarize_image(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary

def apply_canny_edge_detection(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]  # H 채널 추출
    canny_edges = cv2.Canny(h_channel, 50, 150)
    return binarize_image(canny_edges)

def check_vector_errors(x_vector):
    if x_vector is None or len(x_vector) == 0:
        return False
    return True

def sliding_window_single_line(binary_warped, nwindows=9):
    margin = 50
    minpix = 50
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:], axis=0)
    base = np.argmax(histogram)

    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    current = base
    line_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_x_low = current - margin
        win_x_high = current + margin

        good_line_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

        line_inds.append(good_line_inds)

        if len(good_line_inds) > minpix:
            current = int(np.mean(nonzerox[good_line_inds]))

    line_inds = np.concatenate(line_inds)

    linex = nonzerox[line_inds]
    liney = nonzeroy[line_inds]

    if not check_vector_errors(linex):
        return None, None, None, None, None

    line_fit = np.polyfit(liney, linex, 2)
    
    min_y_idx = np.argmin(liney)
    max_y_idx = np.argmax(liney)
    top = (linex[min_y_idx], liney[min_y_idx])
    bottom = (linex[max_y_idx], liney[max_y_idx])

    return line_fit, liney, linex, bottom, top

def calculate_angle(bottom, top):
    x1, y1 = bottom
    x2, y2 = top
    if x2 - x1 == 0:
        return 90.0  # 수직선은 90도
    angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    return angle

def determine_turn(angle, threshold=100):
    if abs(angle) >= threshold:
        if angle < 0:
            return "좌회전"
        else:
            return "우회전"
    else:
        return "직진"

def main_pipeline(image):
    binary_canny = apply_canny_edge_detection(image)
    line_fit, liney, linex, bottom, top = sliding_window_single_line(binary_canny)

    if line_fit is None or bottom is None or top is None:
        return "좌회전"

    angle = calculate_angle(bottom, top)
    turn_direction = determine_turn(angle)

    return turn_direction

def process_camera_input():
    cap = cv2.VideoCapture(0)  # 카메라 입력 (0번 카메라)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    recent_turns = deque(maxlen=3)  # 최근 3프레임의 회전 방향 저장

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임마다 파이프라인 실행
            turn_direction = main_pipeline(frame)
            recent_turns.append(turn_direction)

            # 최근 3프레임이 모두 "좌회전"일 경우 1초 동안 좌회전
            if list(recent_turns) == ["좌회전", "좌회전", "좌회전"]:
                turn_left()
                time.sleep(1)
            else:
                # 회전 방향에 따른 모터 제어
                if turn_direction == "좌회전":
                    turn_left()
                elif turn_direction == "우회전":
                    turn_right()
                else:
                    go_forward()

    except KeyboardInterrupt:
        print("키보드 인터럽트 감지: 모터 클린업 실행")
        stop_motors()

    finally:
        cap.release()
        GPIO.cleanup()

# 카메라 입력을 통한 로봇 제어 시작
process_camera_input()
