import RPi.GPIO as GPIO
import pygame
import time
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

GPIO.setmode(GPIO.BCM)

# 모터 초기화 (왼쪽 모터: motor1, 오른쪽 모터: motor2)
motor1 = MotorController(18, 17, 27)  # 모터1: en(18), in1(17), in2(27)
motor2 = MotorController(16, 13, 26)  # 모터2: en(16), in1(13), in2(26)

# PCA9685 모듈 초기화 (서보모터)
kit = ServoKit(channels=16)

# 서보모터 초기 설정 (스티어링 휠, 채널 0 사용)
kit.servo[0].angle = 90  # 초기 서보모터 각도 (중립)

# Pygame 초기화
pygame.init()

# 조이스틱 초기화
pygame.joystick.init()

# 첫 번째 조이스틱을 선택
joystick = pygame.joystick.Joystick(0)
joystick.init()

# 속도 및 서보모터 각도 초기값 설정
servo_angle = 90  # 서보모터 각도 (중립)
speed = 15  # 초기 속도를 15로 설정

# 메인 루프
running = True
while running:
    for event in pygame.event.get():
        # 좌측 스틱 (서보모터 각도 제어: 축 0)
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:  # 좌측 스틱 좌우
                axis_value = joystick.get_axis(0)  # 축 0: 좌측 스틱 좌우 움직임
                servo_angle = (axis_value + 1) * 90  # 서보모터 각도 0 ~ 180도
                servo_angle = max(0, min(180, servo_angle))  # 각도 제한
                kit.servo[0].angle = servo_angle  # 서보모터 각도 설정
                print(f"서보 각도: {servo_angle}")

            # 우측 스틱 위아래 (속도 제어: 축 3)
            if event.axis == 3:  # 우측 스틱 위아래
                axis_value = joystick.get_axis(3)  # 축 3: 우측 스틱 위아래
                if axis_value < -0.1:  # 스틱을 위로 올리면
                    speed = min(speed + 1, 100)  # 속도 증가, 최대 100%
                    motor1.forward(speed)
                    motor2.forward(speed)
                    print(f"속도 상승: {speed}")
                elif axis_value > 0.1:  # 스틱을 아래로 내리면
                    speed = max(speed - 1, 0)  # 속도 감소, 최소 0%
                    motor1.forward(speed)
                    motor2.forward(speed)
                    print(f"속도 감소: {speed}")

        # 버튼 9를 눌러 정지
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 10:  # 버튼 10: 정지 버튼
                speed = 0  # 속도 정지
                motor1.stop()
                motor2.stop()
                print("정지 버튼을 눌렀습니다. 속도: 0")

        # 종료 이벤트 처리
        elif event.type == pygame.QUIT:
            running = False

    # 우측 스틱이 중립일 때도 속도 하강
    axis_value = joystick.get_axis(3)
    if -0.1 < axis_value < 0.1:  # 중립 상태일 때
        speed = max(speed - 0.5, 0)  # 속도를 점진적으로 하강
        motor1.forward(speed)
        motor2.forward(speed)
        print(f"속도 하강 중: {speed}")
        time.sleep(0.5)  # 0.5초마다 속도 하강

# Pygame 종료
pygame.quit()

# GPIO 정리
GPIO.cleanup()
