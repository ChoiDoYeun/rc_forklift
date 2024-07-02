import time
from adafruit_servokit import ServoKit
import cv2

# PCA9685 모듈 초기화 (16채널 서보 컨트롤러 사용)
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 0
kit.servo[1].angle = 0
kit.servo[2].angle = 0
kit.servo[3].angle = 90
kit.servo[4].angle = 0

def set_servo_angle(servo_number, angle):
    if 0 <= angle <= 180:
        kit.servo[servo_number].angle = angle
    else:
        print("Angle must be between 0 and 180 degrees")

def main():
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # 그레이스케일로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 이진화 적용
        _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        # 이진화된 이미지를 화면에 출력
        cv2.imshow('Binary Camera Stream', binary_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 사용자 입력을 기다림 (비차단 방식)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            try:
                servo_number = int(input("Enter servo number (0-4): "))
                angle = int(input(f"Enter angle for servo {servo_number} (0-180): "))
                set_servo_angle(servo_number, angle)
            except ValueError:
                print("Invalid input. Please enter valid numbers.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
