import time
from adafruit_servokit import ServoKit

# PCA9685 모듈은 16채널 서보 컨트롤러이지만 우리는 5개만 사용
kit = ServoKit(channels=16)

def set_servo_angle(servo_number, angle):
    if 0 <= angle <= 180:
        kit.servo[servo_number].angle = angle
    else:
        print("Angle must be between 0 and 180 degrees")

def main():
    while True:
        try:
            # 사용자 입력 받기
            servo_number = int(input("Enter servo number (0-4): "))
            if 0 <= servo_number <= 4:
                angle = int(input(f"Enter angle for servo {servo_number} (0-180): "))
                set_servo_angle(servo_number, angle)
            else:
                print("Servo number must be between 0 and 4")
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
