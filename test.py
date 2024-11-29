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

# use device gpu

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
        self.target_model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.models.Sequential()
        # model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)
        act_values = self.model.predict(np.reshape(state, (1, self.state_size)))[0]
        return act_values # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        targets = []
        next_state_targets = []
        for i in range(batch_size):
            targets.append(self.model.predict(np.reshape(states[i], (1, self.state_size))))
            next_state_targets.append(self.model.predict(np.reshape(next_states[i], (1, self.state_size))))

            targets[i] = np.reshape(targets[i], (2,))
            next_state_targets[i] = np.reshape(next_state_targets[i], (2,))

        for i in range(batch_size):
            if dones[i]:
                targets[i][0] = rewards[i]
                targets[i][1] = rewards[i]
            else:
                targets[i][0] = rewards[i] + self.gamma * next_state_targets[i][0]
                targets[i][1] = rewards[i] + self.gamma * next_state_targets[i][1]

        states = np.array(states)
        targets = np.array(targets)

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)



if __name__ == "__main__":
    env = Env.env()
    agent = DQNAgent(env.state_size, env.action_size)

    done = False
    batch_size = 32
    clock = pygame.time.Clock()
    real = False

    #agent.load("model.h5")


    for e in range(EPISODES):
        state = env.reset()
        state = np.array(state)
        total_reward = 0
        for time in range(999999): # 20 seconds in 30 fps
            actions = agent.act(state)
            next_state, reward, done = env.step(actions, time)
            total_reward += reward
            next_state = np.array(next_state)
            agent.remember(state, actions, reward, next_state, done)
            state = next_state

            env.render()

            if done or time == 999999-1:
                if time < 599:
                    real = True
                print("episode: {}/{}, score: {}, e: {:.2}, r: {}".format(e, EPISODES, time, agent.epsilon, total_reward))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    

    # save
    agent.save("model.h5")
        
