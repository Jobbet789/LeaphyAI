# Version 0.0.2
import lib.Ball as Ball
import lib.Robot as Robot
import lib.DQLAgent as DQLAgent
import lib.Window as Window

import numpy as np
import tensorflow as tf
import random
from collections import deque
import math
import pygame

# CONSTANTS
WIDTH = 800
HEIGHT = 600

window = Window.Window(WIDTH, HEIGHT)

def train_dql(episodes):
    screen = window.setup()
    clock = pygame.time.Clock()

    agent = DQLAgent.DQLAgent()
    max_steps_per_episode = 500
    for e in range(episodes):
        robot = Robot.Robot(random.randint(0, WIDTH), random.randint(0, HEIGHT))
        ball = Ball.Ball(random.randint(0, WIDTH), random.randint(0, HEIGHT))

        state = np.array(robot.inputsNN(robot.vision(ball)[0]))
        done = False
        total_reward = 0
        steps = 0

        while not done:
            #clock.tick(60)
            action = agent.act(state)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            robot.speed1 = action[0]
            robot.speed2 = action[1]

            robot.move(WIDTH, HEIGHT)

            next_state = np.array(robot.inputsNN(robot.vision(ball)[0]))

            distance = math.sqrt(((robot.x+robot.RADIUS/2) - (ball.x+ball.RADIUS/2))**2 + ((robot.y+robot.RADIUS/2) - (ball.y+ball.RADIUS/2))**2)
            reward = 50 if distance < 10 else 0
            if reward != 1:
                # give a reward based on the distance to the ball
                if distance < 50:
                    reward = 1 - distance/50
            total_reward += reward

            done = distance < 10

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            if steps % 10 == 0:
                window.draw(screen, robot, ball)

            if steps >= max_steps_per_episode:
                done = True

            if done:
                print(f"episode: {e}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon}")

        agent.replay()
        if e % 10 == 0:
            agent.update_target_model()

if __name__ == '__main__':
    train_dql(1000)
    pygame.quit()