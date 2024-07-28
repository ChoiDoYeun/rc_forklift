import time
from adafruit_servokit import ServoKit
import cv2

# PCA9685 모듈 초기화 (16채널 서보 컨트롤러 사용)
kit = ServoKit(channels=16)

# 서보모터 초기 설정
kit.servo[0].angle = 0
kit.servo[1].angle = 60
kit.servo[2].angle = 0
kit.servo[3].angle = 0
kit.servo[4].angle = 0
kit.servo[5].angle = 80
kit.servo[6].angle = 90

def main():
    for i in range(32):  # 최대 32개의 비디오 장치를 시도
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully opened /dev/video{i}")
            break
    else:
        print("Error: Could not open any camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # 화면에 프레임을 출력
        cv2.imshow('Camera Stream', frame)
        
        # 'q' 키를 누르면 종료, 스페이스바를 누르면 캡처하여 저장
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 스페이스바를 눌렀을 때
            timestamp = int(time.time())
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image captured and saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
