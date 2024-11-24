# Version 0.0.2
import lib.Ball as Ball
import lib.Robot as Robot
import lib.Window as Window
import lib.Env as Env

import numpy as np
import random
from collections import deque
import math
import pygame
import tensorflow as tf

EPISODES = 1000

 
# tensorflow==2.11.0

# CONSTANTS

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.models.Sequential()
        # model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return act_values # returns action
    
    def replay(self, batch_size):
        '''
        WIP 
        '''
        pass
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)



if __name__ == "__main__":
    env = Env.env()
    agent = DQNAgent(env.state_size, env.action_size)

    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        for time in range(20*30): # 20 seconds in 30 fps
            actions = agent.act(state)
            next_state, reward, done = env.step(actions)
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, actions, reward, next_state, done)
            state = next_state

            env.render()

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)