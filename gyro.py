import smbus
import time

# I2C 버스 초기화
bus = smbus.SMBus(1)  # 라즈베리파이 모델에 따라 버스 번호가 다를 수 있음

# GY-291 주소
DEVICE_ADDRESS = 0x68

# MPU-6050 초기화
bus.write_byte_data(DEVICE_ADDRESS, 0x6B, 0)

def read_raw_data(addr):
    # 가속도계 및 자이로 데이터 읽기
    high = bus.read_byte_data(DEVICE_ADDRESS, addr)
    low = bus.read_byte_data(DEVICE_ADDRESS, addr+1)
    value = ((high << 8) | low)
    if value > 32768:
        value = value - 65536
    return value

while True:
    # 가속도계 데이터
    acc_x = read_raw_data(0x3B)
    acc_y = read_raw_data(0x3D)
    acc_z = read_raw_data(0x3F)

    # 자이로 데이터
    gyro_x = read_raw_data(0x43)
    gyro_y = read_raw_data(0x45)
    gyro_z = read_raw_data(0x47)

    # 출력
    print(f"ACCEL: X={acc_x}, Y={acc_y}, Z={acc_z}")
    print(f"GYRO: X={gyro_x}, Y={gyro_y}, Z={gyro_z}")

    time.sleep(1)
