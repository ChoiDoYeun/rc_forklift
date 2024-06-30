import smbus
import time

# I2C 버스 및 GY-291 주소 설정
bus = smbus.SMBus(1)
address = 0x53

# GY-291 초기 설정
bus.write_byte_data(address, 0x2D, 0x08)  # 측정 모드 설정
bus.write_byte_data(address, 0x31, 0x0B)  # 풀 해상도 설정

def read_accel():
    # 각 축의 가속도 데이터를 읽습니다.
    data = bus.read_i2c_block_data(address, 0x32, 6)
    x = (data[1] << 8) | data[0]
    y = (data[3] << 8) | data[2]
    z = (data[5] << 8) | data[4]

    # 16비트 값을 2의 보수로 변환합니다.
    if x & (1 << 15):
        x -= (1 << 16)
    if y & (1 << 15):
        y -= (1 << 16)
    if z & (1 << 15):
        z -= (1 << 16)

    return x, y, z

try:
    while True:
        x, y, z = read_accel()
        print(f"X: {x}, Y: {y}, Z: {z}")
        time.sleep(1)
except KeyboardInterrupt:
    print("종료합니다.")
