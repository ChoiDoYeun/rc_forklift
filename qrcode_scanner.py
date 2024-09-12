import time
import cv2
import RPi.GPIO as GPIO
from pyzbar import pyzbar
from adafruit_servokit import ServoKit

# PCA9685 모듈 초기화 (16채널 서보 컨트롤러 사용)
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 0
kit.servo[1].angle = 60
kit.servo[2].angle = 0
kit.servo[3].angle = 0
kit.servo[4].angle = 0
kit.servo[5].angle = 80
kit.servo[6].angle = 120

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

# GPIO 핀 설정
GPIO.setmode(GPIO.BCM)

motor1 = MotorController(18, 17, 27)
motor2 = MotorController(22, 23, 24)
motor3 = MotorController(12, 5, 6)
motor4 = MotorController(16, 13, 26)

def stop_motors():
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()

def go_forward():
    motor1.forward(50)  # 모터 속도 설정
    motor2.forward(50)
    motor3.forward(50)
    motor4.forward(50)

def turn_left():
    motor1.forward(50)
    motor2.backward(50)
    motor3.forward(50)
    motor4.backward(50)

# QR 코드 인식 및 동작 수행 함수
def decode_qr_code(frame):
    qr_codes = pyzbar.decode(frame)
    for qr_code in qr_codes:
        qr_data = qr_code.data.decode('utf-8')
        print(f"QR 코드 인식됨: {qr_data}")
        
        if qr_data == "go":
            go_forward()
        elif qr_data == "left":
            turn_left()
        elif qr_data == "stop":
            stop_motors()

# 카메라를 통해 실시간 QR 코드 인식 및 동작 제어
def main():
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # QR 코드 인식 및 동작 수행
            decode_qr_code(frame)

            # 화면에 QR 코드가 인식된 부분 표시
            for qr_code in pyzbar.decode(frame):
                (x, y, w, h) = qr_code.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 결과 화면 출력
            cv2.imshow("QR Code Scanner", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 종료 시 자원 정리
        cap.release()
        cv2.destroyAllWindows()
        stop_motors()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
