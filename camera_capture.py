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

# 서보모터 각도 설정
servo1_angle = 85  # 서보모터 1 좌우 각도
servo2_angle = 155  # 서보모터 2 상하 각도

# 저장할 이미지 카운터
img_counter = 0

# 키보드로 서보모터 제어하는 함수
def control_servo(key, servo1_angle, servo2_angle):
    if key == ord('d'):  # 서보모터 1 각도 증가 (우)
        servo1_angle += 5
    elif key == ord('a'):  # 서보모터 1 각도 감소 (좌)
        servo1_angle -= 5
    elif key == ord('w'):  # 서보모터 2 각도 감소 (상)
        servo2_angle -= 5
    elif key == ord('s'):  # 서보모터 2 각도 증가 (하)
        servo2_angle += 5

    # 각도 제한 설정
    servo1_angle = max(0, min(180, servo1_angle))
    servo2_angle = max(0, min(180, servo2_angle))

    # 서보모터 각도 적용
    kit.servo[1].angle = servo1_angle
    kit.servo[2].angle = servo2_angle

    # 서보모터 각도를 출력
    print(f"Servo1 (좌우): {servo1_angle}도, Servo2 (상하): {servo2_angle}도")

    return servo1_angle, servo2_angle

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
