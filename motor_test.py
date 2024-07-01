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
motor3 = MotorController(12, 5, 6)
motor4 = MotorController(16, 13, 26)

try:
    while True:
        user_input = input("Enter 1 for forward, 2 for clockwise rotation, 3 for counter-clockwise rotation, q to quit: ")

        if user_input == '1':
            # 직진
            motor1.forward(70)
            motor2.forward(70)
            motor3.forward(70)
            motor4.forward(70)
            time.sleep(0.75)
        
        elif user_input == '2':
            # 시계 방향 회전
            motor1.backward(70)
            motor2.forward(70)
            motor3.backward(70)
            motor4.forward(70)
            time.sleep(0.75)
        
        elif user_input == '3':
            # 시계 반대 방향 회전
            motor1.forward(70)
            motor2.backward(70)
            motor3.forward(70)
            motor4.backward(70)
            time.sleep(0.75)
        
        elif user_input == 'q':
            print("Exiting program.")
            break
        
        else:
            print("Invalid input. Please enter 1, 2, 3, or q.")
        
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
    motor3.cleanup()
    motor4.cleanup()
    GPIO.cleanup()
