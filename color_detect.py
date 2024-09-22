import cv2
import numpy as np

# 카메라 설정
cap = cv2.VideoCapture(0)

# 각 색상의 HSV 범위를 정의
colors = {
    "red": [(0, 120, 70), (10, 255, 255)],  # 빨간색 범위
    "blue": [(110, 50, 50), (130, 255, 255)],  # 파란색 범위
    "yellow": [(20, 100, 100), (30, 255, 255)],  # 노란색 범위
    "black": [(0, 0, 0), (180, 255, 50)]  # 검정색 범위
}

while True:
    # 카메라에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 이미지를 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower, upper) in colors.items():
        # 지정한 색상 범위로 마스크 생성
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 마스크에서 픽셀이 검출되면 그 색상을 감지했다고 판단
        if np.any(mask):
            print(f"{color_name} detected")

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 및 카메라 릴리스
cap.release()
cv2.destroyAllWindows()
