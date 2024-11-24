import lib.Ball as Ball
import lib.Robot as Robot
import lib.Window as Window

import pygame
import random

class env:
    WIDTH = 800
    HEIGHT = 600

    def __init__(self):
        self.window = Window.Window(self.WIDTH, self.HEIGHT)
        self.ball = Ball.Ball(self.WIDTH, self.HEIGHT)
        self.robot = Robot.Robot(self.WIDTH, self.HEIGHT)

        self.action_size = 2
        self.state_size = 2

        self.ballVis = False

        self.screen = self.window.setup()
    
    def reset(self):
        self.ball = Ball.Ball(random.randint(0+Ball.Ball.RADIUS, self.WIDTH-Ball.Ball.RADIUS), random.randint(0+Ball.Ball.RADIUS, self.HEIGHT-Ball.Ball.RADIUS))
        self.robot = Robot.Robot(random.randint(0+Robot.Robot.RADIUS, self.WIDTH-Robot.Robot.RADIUS), random.randint(0+Robot.Robot.RADIUS, self.HEIGHT-Robot.Robot.RADIUS))
        return self.get_state()
    
    def get_state(self):
        return [self.robot.headingToBall, int(self.ballVis)]
    
    def step(self, action: list, time: int):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.robot.speed1 = action[0]
        self.robot.speed2 = action[1]
        self.robot.move(self.WIDTH, self.HEIGHT)
        self.ballVis = self.robot.vision(self.ball)
        reward = 0

        done = False
        distance = self.robot.distanceToBall(self.ball)

        if self.ballVis:
            reward += 1
        else:
            reward -= 1

        if distance < 10:
            reward += 50
            # more reward based on time
            reward += 20 * (20*30 - time)
            done = True

        return self.get_state(), reward, done
    
    def render(self):
        self.window.draw(self.screen, self.ball, self.robot)
        pygame.display.update()

    def close(self):
        pygame.quit()
        
