import RPi.GPIO as GPIO
import time

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
        
    def forward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)

    def backward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)

    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)

    def cleanup(self):
        self.pwm.stop()

GPIO.setmode(GPIO.BCM)

motor1 = MotorController(18, 17, 27)
motor2 = MotorController(22, 23, 24)
motor3 = MotorController(12,5,6)
motor4 = MotorController(16,13,26)

try:
    # 제자리에서 90도 회전 (시계 방향)
    motor1.backward(50)  # 왼쪽 앞 모터 뒤로
    motor2.forward(50)   # 오른쪽 앞 모터 앞으로
    motor3.backward(50)  # 왼쪽 뒤 모터 뒤로
    motor4.forward(50)   # 오른쪽 뒤 모터 앞으로

    time.sleep(1)  # 회전 시간 조절 (테스트를 통해 적절히 설정)

    # 모터 정지
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    motor1.cleanup()
    motor2.cleanup()
    GPIO.cleanup()
