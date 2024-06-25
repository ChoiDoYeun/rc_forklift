import cv2

def capture_frame_on_spacebar():
    # 웹캠에서 영상을 캡쳐합니다.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 캡쳐할 때 사용할 파일 번호를 입력받습니다.
    capture_number = int(input("파일 번호를 입력하세요: "))
    
    # 해상도를 320x240으로 설정합니다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

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
            # 입력받은 파일 번호로 파일 이름을 만듭니다.
            filename = f"capture_{capture_number}.png"
            cv2.imwrite(filename, frame)
            print(f"{filename} 저장되었습니다.")
        
        elif key == ord('q'):  # 'q' 키를 누르면
            break

    # 모든 윈도우를 닫고 캡쳐를 중지합니다.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frame_on_spacebar()
