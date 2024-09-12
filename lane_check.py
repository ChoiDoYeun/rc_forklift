import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# 동영상 처리 실행
video_path = 'output.avi'  # 동영상 파일 경로

# 이진화 함수
def binarize_image(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary

# 에지 감지 함수 (Canny 에지 감지 사용)
def apply_canny_edge_detection(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]  # H 채널 추출
    canny_edges = cv2.Canny(h_channel, 50, 150)
    return binarize_image(canny_edges)

# 이전 회전 방향을 저장하는 변수
previous_turn_direction = "직진"  # 초기 값

def sliding_window_single_line(binary_warped, nwindows=9):
    margin = 50
    minpix = 50
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:], axis=0)
    base = np.argmax(histogram)  # 하나의 라인만 감지

    window_height = int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    current = base
    line_inds = []
    msk = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_x_low = current - margin
        win_x_high = current + margin

        good_line_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

        line_inds.append(good_line_inds)

        if len(good_line_inds) > minpix:
            current = int(np.mean(nonzerox[good_line_inds]))

        cv2.rectangle(msk, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

    if len(line_inds) == 0:
        return None, None, None, None, None, None

    line_inds = np.concatenate(line_inds)

    linex = nonzerox[line_inds]
    liney = nonzeroy[line_inds]

    if len(linex) == 0 or len(liney) == 0:
        return None, None, None, None, None, None

    line_fit = np.polyfit(liney, linex, 2)

    msk[liney, linex] = [0, 0, 255]
    
    # 가장 상단과 하단의 픽셀 좌표 찾기
    min_y_idx = np.argmin(liney)
    max_y_idx = np.argmax(liney)
    top = (linex[min_y_idx], liney[min_y_idx])  # 가장 상단 픽셀 (x, y)
    bottom = (linex[max_y_idx], liney[max_y_idx])  # 가장 하단 픽셀 (x, y)

    return line_fit, liney, linex, msk, bottom, top

def calculate_angle(bottom, top):
    x1, y1 = bottom
    x2, y2 = top
    if x2 - x1 == 0:
        return 90.0  # 수직선은 90도
    angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    return angle

def determine_turn(angle, threshold=15):
    if abs(angle) >= threshold:
        if angle > 0:
            return "좌회전"
        else:
            return "우회전"
    else:
        return "직진"

def main_pipeline(image):
    global previous_turn_direction  # 전역 변수를 사용하여 이전 회전 방향을 저장

    # Canny 에지 감지 및 바이너리화
    binary_canny = apply_canny_edge_detection(image)

    # 슬라이딩 윈도우를 이용한 단일 라인 감지
    line_fit, liney, linex, msk, bottom, top = sliding_window_single_line(binary_canny)

    if bottom is None or top is None:
        # 감지된 픽셀이 없을 경우 이전 회전 방향을 사용
        print(f"회전 방향: {previous_turn_direction} (이전 값)")
        return None, previous_turn_direction

    # 각도 계산
    angle = calculate_angle(bottom, top)

    # 회전 방향 결정
    turn_direction = determine_turn(angle)

    # 회전 방향을 전역 변수에 저장
    previous_turn_direction = turn_direction

    # 결과 출력
    print(f"감지된 라인의 각도: {angle}도")
    print(f"회전 방향: {turn_direction}")
    
    return angle, turn_direction

# 동영상 처리에서 각도 계산 및 회전 방향 출력
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("동영상을 열 수 없습니다.")
        return

    frame_count = 0  # 프레임 카운터

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        frame_count += 1  # 프레임 카운트 증가
        
        # 프레임마다 파이프라인 실행
        angle, turn_direction = main_pipeline(frame)

        # 프레임 번호와 회전 방향 출력
        print(f"프레임 {frame_count}: 회전 방향: {turn_direction}")

        # 결과 이미지 시각화
        cv2.imshow('Frame', frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 동영상 처리 실행
process_video(video_path)
