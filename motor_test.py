import RPi.GPIO as GPIO
import time

# 핀 번호 설정 (BCM 모드)
EN = 18
IN1 = 17
IN2 = 27

# GPIO 모드 설정
GPIO.setmode(GPIO.BCM)

# 모터 핀 설정
GPIO.setup(EN, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# PWM 설정: EN 핀에 PWM 신호를 보냄
pwm = GPIO.PWM(EN, 100)  # 100 Hz 주파수
pwm.start(0)  # PWM 시작, 0% 듀티 사이클

def motor_forward(speed):
    pwm.ChangeDutyCycle(speed)  # 모터 속도 설정
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    print("모터 전진")

def motor_backward(speed):
    pwm.ChangeDutyCycle(speed)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    print("모터 후진")

def motor_stop():
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    print("모터 정지")

try:
    # 테스트 순서
    motor_forward(50)  # 50% 속도로 전진
    time.sleep(2)      # 2초간 동작
    motor_backward(50) # 50% 속도로 후진
    time.sleep(2)
    motor_stop()       # 모터 정지

except KeyboardInterrupt:
    pass

finally:
    pwm.stop()         # PWM 정지
    GPIO.cleanup()     # GPIO 핀 설정 초기화
