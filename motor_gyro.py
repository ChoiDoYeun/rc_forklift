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

def print_motor_states_and_accel():
    x, y, z = read_accel()
    print(f"Motor1: {motor1.state}")
    print(f"Motor2: {motor2.state}")
    print(f"Motor3: {motor3.state}")
    print(f"Motor4: {motor4.state}")
    print(f"Accel X: {x}, Y: {y}, Z: {z}")

initialize_sensor()

try:
    while True:
        user_input = input("Enter 1 for forward, 2 for clockwise rotation, 3 for counter-clockwise rotation, q to quit: ")

        if user_input == '1':
            # Forward
            motor1.forward(72)
            motor2.forward(70)
            motor3.forward(70)
            motor4.forward(70)
            time.sleep(0.418)
        
        elif user_input == '2':
            # Clockwise rotation
            motor1.backward(70)
            motor2.forward(70)
            motor3.backward(70)
            motor4.forward(70)
            time.sleep(0.75)
        
        elif user_input == '3':
            # Counter-clockwise rotation
            motor1.forward(70)
            motor2.backward(70)
            motor3.forward(70)
            motor4.backward(70)
            time.sleep(0.75)
        
        elif user_input == 'q':
            print("Exiting program.")
            break
        
        else:
            print("Invalid input. Please enter 1, 2, 3, or q.")
        
        # Stop all motors
        motor1.stop()
        motor2.stop()
        motor3.stop()
        motor4.stop()

        # Print motor states and accelerometer data
        print_motor_states_and_accel()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    motor1.cleanup()
    motor2.cleanup()
    motor3.cleanup()
    motor4.cleanup()
    GPIO.cleanup()
