# Version: 0.0.1

# IMPORTS
import math
import pygame
import random

# CLASSES
from lib import Ball
from lib import Robot
from lib import Window

# CONSTANTS
WIDTH = 800
HEIGHT = 600

window = Window.Window(WIDTH, HEIGHT)
window.setup()

def main():
    screen = window.setup()
    run = True
    clock = pygame.time.Clock()

    ball = Ball.Ball(random.randint(0, WIDTH), random.randint(0, HEIGHT))
    robot = Robot.Robot(random.randint(0, WIDTH), random.randint(0, HEIGHT))

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

        robot.move(WIDTH, HEIGHT)
        ball.move()

        robot.vision(ball)

        # check if robot is colliding with ball
        if robot.x < ball.x + ball.RADIUS and robot.x + robot.RADIUS > ball.x and robot.y < ball.y + ball.RADIUS and robot.y + robot.RADIUS > ball.y:
            ball = Ball.Ball(random.randint(0, WIDTH), random.randint(0, HEIGHT))

        window.draw(screen, robot, ball)
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()