import RPi.GPIO as GPIO
import time
from adafruit_servokit import ServoKit
import pygame

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

GPIO.setmode(GPIO.BCM)

motor1 = MotorController(18, 17, 27)
motor2 = MotorController(22, 23, 24)
motor3 = MotorController(12, 5, 6)
motor4 = MotorController(16, 13, 26)

# PCA9685 모듈 초기화 (16채널 서보 컨트롤러 사용)
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 0
kit.servo[1].angle = 150
kit.servo[2].angle = 0
kit.servo[3].angle = 0
kit.servo[4].angle = 0

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((100, 100))

def stop_motors():
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()

try:
    print("W: Forward, S: Backward, A: Rotate left, D: Rotate right, Q: Quit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    motor1.forward(70)
                    motor2.forward(0)
                    motor3.forward(70)
                    motor4.forward(70)
                elif event.key == pygame.K_s:
                    motor1.backward(0)
                    motor2.backward(70)
                    motor3.backward(0)
                    motor4.backward(0)
                elif event.key == pygame.K_a:
                    motor1.forward(0)
                    motor2.forward(0)  # 안쪽 바퀴 속도 낮춤
                    motor3.forward(70)
                    motor4.forward(0)  # 안쪽 바퀴 속도 낮춤
                elif event.key == pygame.K_d:
                    motor1.forward(0)  # 안쪽 바퀴 속도 낮춤
                    motor2.forward(0)
                    motor3.forward(0)  # 안쪽 바퀴 속도 낮춤
                    motor4.forward(70)
                elif event.key == pygame.K_q:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    stop_motors()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    motor1.cleanup()
    motor2.cleanup()
    motor3.cleanup()
    motor4.cleanup()
    GPIO.cleanup()
    pygame.quit()
