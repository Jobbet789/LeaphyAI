# Version: 0.0.1

# IMPORTS
import math
import pygame

# CLASSES
from lib import Ball
from lib import Robot
from lib import Window

# CONSTANTS
WIDTH = 800
HEIGHT = 600

BALLX = 400
BALLY = 300

ROBOTX = 400
ROBOTY = 300

window = Window.Window(WIDTH, HEIGHT)
window.setup()

def main():
    screen = window.setup()
    run = True
    clock = pygame.time.Clock()

    ball = Ball.Ball(BALLX, BALLY)
    robot = Robot.Robot(ROBOTX, ROBOTY)

    while run:
        dt = clock.tick(60) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            robot.set_speed(-10*dt, 10*dt)
        elif keys[pygame.K_RIGHT]:
            robot.set_speed(10*dt, -10*dt)
        else:
            robot.set_speed(1, 1)

        robot.move()
        ball.move()

        window.draw(screen, robot, ball)
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()