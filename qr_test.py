import torch
from PIL import Image
import numpy as np
import cv2
from pyzbar.pyzbar import decode

# 경로 설정
base_path = "/home/dodo/rc_forklift/"
model_path = base_path + 'best.pt'
cropped_img_path = base_path + 'cropped_test.png'

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
torch.save(model.state_dict(), "local_model_path.pt")
model.load_state_dict(torch.load("local_model_path.pt"))

# 감지된 객체 크롭하여 저장
def crop_object(results, frame):
    if len(results.xyxy[0]) > 0: 
        x_min, y_min, x_max, y_max = results.xyxy[0][0][:4].cpu().numpy().astype(int)
        cropped_img = frame[y_min:y_max, x_min:x_max]
        return cropped_img
    return None

# qrcode decode
def read_qr_code(frame):
    decoded_objects = decode(frame)
    return decoded_objects[0].data.decode() if decoded_objects else None

# 카메라 스트리밍 시작
cap = cv2.VideoCapture(0)  # 0은 기본 카메라

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # YOLO 모델로 객체 감지
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    # 감지된 객체 크롭
    cropped_img = crop_object(results, frame)

    if cropped_img is not None:
        # QR 코드 읽기
        qr_code_data = read_qr_code(cropped_img)
        if qr_code_data:
            print(f"QR 코드 감지됨: {qr_code_data}")
            cv2.imshow("Cropped QR", cropped_img)
    
    # 원본 화면 출력
    cv2.imshow("Camera Stream", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
