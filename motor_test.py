import RPi.GPIO as GPIO
import time

# GPIO 설정
GPIO.setmode(GPIO.BCM)

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

# 모터 초기화 (왼쪽 모터: motor1, 오른쪽 모터: motor2)
motor1 = MotorController(18, 17, 27)  # 모터1: en(18), in1(17), in2(27)
motor2 = MotorController(16, 13, 26)  # 모터2: en(22), in1(23), in2(24)


# 모터 테스트
try:
    # 모터1 전진 테스트
    print("모터1 전진")
    motor1.forward(50)  # 50% 속도로 전진
    time.sleep(2)

    # 모터1 후진 테스트
    print("모터1 후진")
    motor1.backward(50)  # 50% 속도로 후진
    time.sleep(2)

    # 모터1 정지
    print("모터1 정지")
    motor1.stop()
    time.sleep(1)

    # 모터2 전진 테스트
    print("모터2 전진")
    motor2.forward(50)  # 50% 속도로 전진
    time.sleep(2)

    # 모터2 후진 테스트
    print("모터2 후진")
    motor2.backward(50)  # 50% 속도로 후진
    time.sleep(2)

    # 모터2 정지
    print("모터2 정지")
    motor2.stop()
    time.sleep(1)

finally:
    # GPIO 정리
    GPIO.cleanup()
