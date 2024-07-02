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
    print("W: Motor1 Forward, S: Motor1 Backward, E: Motor2 Forward, D: Motor2 Backward")
    print("R: Motor3 Forward, F: Motor3 Backward, T: Motor4 Forward, G: Motor4 Backward, Q: Quit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    motor1.forward(70)
                elif event.key == pygame.K_s:
                    motor1.backward(70)
                elif event.key == pygame.K_e:
                    motor2.forward(70)
                elif event.key == pygame.K_d:
                    motor2.backward(70)
                elif event.key == pygame.K_r:
                    motor3.forward(70)
                elif event.key == pygame.K_f:
                    motor3.backward(70)
                elif event.key == pygame.K_t:
                    motor4.forward(70)
                elif event.key == pygame.K_g:
                    motor4.backward(70)
                elif event.key == pygame.K_q:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s]:
                    motor1.stop()
                elif event.key in [pygame.K_e, pygame.K_d]:
                    motor2.stop()
                elif event.key in [pygame.K_r, pygame.K_f]:
                    motor3.stop()
                elif event.key in [pygame.K_t, pygame.K_g]:
                    motor4.stop()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    motor1.cleanup()
    motor2.cleanup()
    motor3.cleanup()
    motor4.cleanup()
    GPIO.cleanup()
    pygame.quit()
