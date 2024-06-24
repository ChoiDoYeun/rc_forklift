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
    motor1.backward(50)
    motor2.backward(50)
    motor3.backward(50)
    motor4.backward(50)
    time.sleep(3)
    motor1.forward(50)
    motor2.forward(50)
    motor3.forward(50)
    motor4.forward(50)
    time.sleep(3)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    motor1.cleanup()
    motor2.cleanup()
    GPIO.cleanup()
