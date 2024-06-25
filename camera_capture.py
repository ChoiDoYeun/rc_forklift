import cv2
import datetime

def capture_frame_on_spacebar():
    # 웹캠에서 영상을 캡쳐합니다.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        # 비디오 프레임을 읽습니다.
        ret, frame = cap.read()
        
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 화면에 프레임을 표시합니다.
        cv2.imshow('Video Capture', frame)

        # 키보드 입력을 기다립니다.
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # 스페이스바를 누르면
            # 현재 시간을 파일 이름에 추가합니다.
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"{filename} 저장되었습니다.")
        
        elif key == ord('q'):  # 'q' 키를 누르면
            break

    # 모든 윈도우를 닫고 캡쳐를 중지합니다.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frame_on_spacebar()
