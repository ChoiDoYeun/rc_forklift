import pygame
import time

# Pygame 초기화
pygame.init()

# 조이스틱 초기화
pygame.joystick.init()

# 연결된 조이스틱이 있는지 확인
if pygame.joystick.get_count() == 0:
    print("연결된 조이스틱이 없습니다")
else:
    # 첫 번째 조이스틱을 선택
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"조이스틱 이름: {joystick.get_name()}")
    print(f"총 버튼 수: {joystick.get_numbuttons()}")
    print(f"총 축(axes) 수: {joystick.get_numaxes()}")
    print(f"총 해트(hats) 수: {joystick.get_numhats()}")

    # 메인 루프
    running = True
    while running:
        for event in pygame.event.get():
            # 버튼 눌림 감지
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"버튼 {event.button} 눌림")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"버튼 {event.button} 뗌")

            # 축(axes) 움직임 감지
            elif event.type == pygame.JOYAXISMOTION:
                for i in range(joystick.get_numaxes()):
                    axis_value = joystick.get_axis(i)
                    print(f"축 {i} 움직임: {axis_value}")

            # 해트(hats) 움직임 감지 (방향 패드)
            elif event.type == pygame.JOYHATMOTION:
                for i in range(joystick.get_numhats()):
                    hat_value = joystick.get_hat(i)
                    print(f"해트 {i} 움직임: {hat_value}")

            # 종료 이벤트 처리 (ESC키 또는 창 닫기)
            elif event.type == pygame.QUIT:
                running = False

    # Pygame 종료
    pygame.quit()
