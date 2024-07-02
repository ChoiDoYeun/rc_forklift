import RPi.GPIO as GPIO
import time
from adafruit_servokit import ServoKit
import sys
import termios
import tty

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
        
    def backward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)

    def forward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)

    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)

    def cleanup(self):
        self.pwm.stop()

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

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

try:
    print("W: Forward, S: Backward, A: Rotate left, D: Rotate right, Q: Quit")
    while True:
        key = get_key()
        
        if key == 'w':
            # 직진
            motor1.forward(70)
            motor2.forward(70)
            motor3.forward(70)
            motor4.forward(70)
        
        elif key == 's':
            # 후진
            motor1.backward(70)
            motor2.backward(70)
            motor3.backward(70)
            motor4.backward(70)
        
        elif key == 'a':
            # 시계 반대 방향 회전
            motor1.backward(70)
            motor2.forward(70)
            motor3.backward(70)
            motor4.forward(70)
        
        elif key == 'd':
            # 시계 방향 회전
            motor1.forward(70)
            motor2.backward(70)
            motor3.forward(70)
            motor4.backward(70)
        
        elif key == 'q':
            print("Exiting program.")
            break
        
        # 모터 정지
        time.sleep(0.1)
        motor1.stop()
        motor2.stop()
        motor3.stop()
        motor4.stop()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    motor1.cleanup()
    motor2.cleanup()
    motor3.cleanup()
    motor4.cleanup()
    GPIO.cleanup()
