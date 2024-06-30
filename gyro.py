import smbus
import time

# I2C 버스 및 GY-291 주소 설정
bus = smbus.SMBus(1)
address = 0x53

# GY-291 초기 설정
bus.write_byte_data(address, 0x2D, 0x08)  # 측정 모드 설정

def read_accel():
    # 각 축의 가속도 데이터를 읽습니다.
    x = bus.read_byte_data(address, 0x32)
    y = bus.read_byte_data(address, 0x34)
    z = bus.read_byte_data(address, 0x36)
    return x, y, z

try:
    while True:
        x, y, z = read_accel()
        print(f"X: {x}, Y: {y}, Z: {z}")
        time.sleep(1)
except KeyboardInterrupt:
    print("종료합니다.")
