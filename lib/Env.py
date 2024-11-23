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
        self.ball = Ball.Ball(self.WIDTH, self.HEIGHT)
        self.robot = Robot.Robot(self.WIDTH, self.HEIGHT)
        return self.get_state()
    
    def get_state(self):
        return [self.robot.headingToBall, int(self.ballVis)]
    
    def step(self, action):
        self.robot.speed1 = action[0]
        self.robot.speed2 = action[1]
        self.robot.move(self.WIDTH, self.HEIGHT)
        self.ballVis = self.robot.vision(self.ball)
        reward = 0

        done = False
        distance = self.robot.distanceToBall(self.ball)
        if distance < 10:
            reward = 100
            done = True
        elif distance <= 100:
            reward = 1-distance/100
        else:
            reward = -1 

        return self.get_state(), reward, done
    
    def render(self):
        self.window.draw(self.screen, self.ball, self.robot)
        pygame.display.update()

    def close(self):
        pygame.quit()
        
