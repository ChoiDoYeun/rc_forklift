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

def main():
    # GStreamer를 사용하지 않고 웹캠을 엽니다.
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 해상도를 320x240으로 설정합니다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
