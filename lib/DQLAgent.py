import numpy as np
import tensorflow as tf
import random
from collections import deque

class DQLAgent:

    STATE_SIZE = 5
    ACTION_SIZE = 2 
    GAMMA = 0.99  # Discount factor
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = 1.0
        self.gamma = self.GAMMA
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.STATE_SIZE, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.ACTION_SIZE, activation='tanh')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.ACTION_SIZE)
            return np.random.uniform(-1, 1, self.ACTION_SIZE)
        q_values = self.model.predict(state[np.newaxis])
        return q_values[0]
    
    def replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, self.BATCH_SIZE)
        states, targets = [], []


        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state[np.newaxis])
            if done:
                target[0] = reward
            else:
                t = self.target_model.predict(next_state[np.newaxis])

                target[0] = reward + self.gamma * np.amax(t[0])
            states.append(state)
            targets.append(target)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY