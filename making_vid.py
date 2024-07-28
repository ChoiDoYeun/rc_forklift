import time
from adafruit_servokit import ServoKit
import cv2
import RPi.GPIO as GPIO
import pygame

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

def main():
    for i in range(32):  # 최대 32개의 비디오 장치를 시도
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully opened /dev/video{i}")
            break
    else:
        print("Error: Could not open any camera.")
        return

    # 비디오 저장을 위한 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    try:
        print("W: Forward, X: Backward, A: Rotate left, D: Rotate right, S: Stop, Q: Quit")
        
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # 화면에 프레임을 출력
            cv2.imshow('Camera Stream', frame)
            out.write(frame)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        motor1.forward(70)
                        motor2.forward(70)
                        motor3.forward(70)
                        motor4.forward(70)
                    elif event.key == pygame.K_x:
                        motor1.backward(70)
                        motor2.backward(70)
                        motor3.backward(70)
                        motor4.backward(70)
                    elif event.key == pygame.K_a:
                        motor1.stop()  # 좌측 앞 모터 속도 낮춤
                        motor2.forward(40)  # 우측 앞 모터 속도 높임
                        motor3.stop()  # 좌측 뒤 모터 속도 낮춤
                        motor4.forward(40)  # 우측 뒤 모터 속도 높임
                    elif event.key == pygame.K_d:
                        motor1.forward(40)  # 좌측 앞 모터 속도 높임
                        motor2.stop()      # 우측 앞 모터 멈춤
                        motor3.forward(40)  # 좌측 뒤 모터 속도 높임
                        motor4.stop()      # 우측 뒤 모터 멈춤
                    elif event.key == pygame.K_s:
                        stop_motors()
                    elif event.key == pygame.K_q:
                        running = False
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_w, pygame.K_x, pygame.K_a, pygame.K_d]:
                        stop_motors()

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        motor1.cleanup()
        motor2.cleanup()
        motor3.cleanup()
        motor4.cleanup()
        GPIO.cleanup()
        pygame.quit()

if __name__ == "__main__":
    main()
