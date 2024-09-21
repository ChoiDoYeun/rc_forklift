import cv2
from adafruit_servokit import ServoKit
import os

# 서보모터 초기화
kit = ServoKit(channels=16)

# 서보모터 초기 설정 (스티어링 휠과 카메라용 서보모터)
kit.servo[0].angle = 85  # 서보모터 1 초기 각도
kit.servo[1].angle = 85  # 서보모터 2 초기 각도
kit.servo[2].angle = 155  # 서보모터 2 초기 각도

# 카메라 초기화 (OpenCV 사용)
cap = cv2.VideoCapture(0)  # 카메라 장치 선택

# 저장할 이미지 카운터
img_counter = 0

# 이미지 저장 함수
def save_image(frame, img_counter):
    img_name = f"image_{img_counter}.png"
    cv2.imwrite(img_name, frame)
    print(f"{img_name} 저장 완료!")
    img_counter += 1
    return img_counter

# 영상 출력 및 서보모터 제어
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기를 256x256으로 조정
    frame_resized = cv2.resize(frame, (256, 256))

    # 조정된 크기의 영상 출력
    cv2.imshow("Camera Feed (256x256)", frame_resized)

    # 키 입력 확인
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

    # 스페이스바를 누르면 이미지 저장
    if key == ord(' '):
        img_counter = save_image(frame_resized, img_counter)

    # 서보모터 제어 및 각도 출력
    servo1_angle, servo2_angle = control_servo(key, servo1_angle, servo2_angle)

# 종료 처리
cap.release()
cv2.destroyAllWindows()
