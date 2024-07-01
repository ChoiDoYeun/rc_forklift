import smbus
import RPi.GPIO as GPIO
import time

# I2C 버스 및 GY-291 주소 설정
bus = smbus.SMBus(1)
address = 0x53

# GY-291 초기 설정
def initialize_sensor():
    try:
        bus.write_byte_data(address, 0x2D, 0x08)  # 측정 모드 설정
        bus.write_byte_data(address, 0x31, 0x0B)  # 풀 해상도 설정
        print("센서 초기화 성공")
    except Exception as e:
        print(f"초기 설정 중 오류 발생: {e}")

def read_accel():
    try:
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
    except Exception as e:
        print(f"데이터 읽기 중 오류 발생: {e}")
        return 0, 0, 0

# 모터 제어 클래스 정의
class MotorController:
    def __init__(self, en, in1, in2):
        self.en = en
        self.in1 = in1
        self.in2 = in2
        GPIO.setup(self.en, GPIO.OUT)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        self.pwm = GPIO.PWM(self.en, 100)
        self.pwm.start(0)
        self.state = "stopped"
        
    def backward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)
        self.state = f"backward at speed {speed}"

    def forward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)
        self.state = f"forward at speed {speed}"

    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        self.state = "stopped"

    def cleanup(self):
        self.pwm.stop()

GPIO.setmode(GPIO.BCM)

motor1 = MotorController(18, 17, 27)
motor2 = MotorController(22, 23, 24)
motor3 = MotorController(12, 5, 6)
motor4 = MotorController(16, 13, 26)

def log_accel_data(filename, timestamp, action, magnitude):
    with open(filename, 'a') as file:
        file.write(f"{timestamp}, {action}, {magnitude}\n")

initialize_sensor()

filename = "accel_data.txt"
start_time = time.time()

try:
    # 전진
    motor1.forward(70)
    motor2.forward(70)
    motor3.forward(70)
    motor4.forward(70)
    action = "forward"
    duration = 0.75
    end_time = time.time() + duration
    while time.time() < end_time:
        x, y, z = read_accel()
        magnitude = (x**2 + y**2 + z**2) ** 0.5
        timestamp = time.time() - start_time
        log_accel_data(filename, timestamp, action, magnitude)
        time.sleep(0.5)
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()
    time.sleep(0.01)

    # 시계 반대 방향 회전
    motor1.forward(70)
    motor2.backward(70)
    motor3.forward(70)
    motor4.backward(70)
    action = "counter-clockwise"
    duration = 0.75
    end_time = time.time() + duration
    while time.time() < end_time:
        x, y, z = read_accel()
        magnitude = (x**2 + y**2 + z**2) ** 0.5
        timestamp = time.time() - start_time
        log_accel_data(filename, timestamp, action, magnitude)
        time.sleep(0.5)
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()

    print("동작 완료. 프로그램을 종료합니다.")

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")

finally:
    motor1.cleanup()
    motor2.cleanup()
    motor3.cleanup()
    motor4.cleanup()
    GPIO.cleanup()
