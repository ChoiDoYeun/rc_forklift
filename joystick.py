import RPi.GPIO as GPIO
import time
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

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((100, 100))

def stop_motors():
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()

try:
    print("W: Forward, X: Backward, A: Rotate left, D: Rotate right, S: Stop, Q: Quit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    motor1.forward(50)
                    motor2.forward(40)
                    motor3.forward(50)
                    motor4.forward(40)
                elif event.key == pygame.K_x:
                    motor1.backward(50)
                    motor2.backward(35)
                    motor3.backward(50)
                    motor4.backward(35)
                elif event.key == pygame.K_a:
                    motor1.stop()  # 좌측 앞 모터 속도 낮춤
                    motor2.forward(50)  # 우측 앞 모터 속도 높임
                    motor3.stop()  # 좌측 뒤 모터 속도 낮춤
                    motor4.forward(50)  # 우측 뒤 모터 속도 높임
                elif event.key == pygame.K_d:
                    motor1.forward(50)  # 좌측 앞 모터 속도 높임
                    motor2.stop()      # 우측 앞 모터 멈춤
                    motor3.forward(50)  # 좌측 뒤 모터 속도 높임
                    motor4.stop()      # 우측 뒤 모터 멈춤
                elif event.key == pygame.K_s:
                    stop_motors()
                elif event.key == pygame.K_q:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_x, pygame.K_a, pygame.K_d]:
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
