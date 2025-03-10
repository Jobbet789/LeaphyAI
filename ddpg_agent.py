import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from collections import deque

class Actor(nn.Module): # Known as DQN for the commented code
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Using tanh to bound actions between -1 and 1.
        return torch.tanh(self.fc4(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # The critic takes both state and action as input.
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        # Concatenate state and action along the feature dimension.
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.9995
        self.tau = 0.001  # For soft update of target parameters
        self.batch_size = 128
        
        # Replay buffer
        self.memory = deque(maxlen=1000000)

        # Add device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor and Critic Networks
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size, action_size).to(self.device)
        
        # Target networks
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, noise_scale=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0)
            action = action.cpu().numpy()
        self.actor.train()
        # Add exploration noise
        noise = noise_scale * np.random.randn(self.action_size)
        return np.clip(action + noise, -1, 1)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        minibatch = np.array(minibatch, dtype=object)

        # Move tensors to device
        states = torch.FloatTensor(np.array(minibatch[:, 0].tolist())).to(self.device)
        actions = torch.FloatTensor(np.array(minibatch[:, 1].tolist())).to(self.device)
        rewards = torch.FloatTensor(np.array(minibatch[:, 2].tolist())).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(minibatch[:, 3].tolist())).to(self.device)
        dones = torch.FloatTensor(np.array(minibatch[:, 4].tolist(), dtype=float)).unsqueeze(1).to(self.device)
        
        # ----- Update Critic -----
        # Compute target actions and Q-values
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            # Bellman target
            y = rewards + self.gamma * (1 - dones) * target_q
        
        # Current Q estimates
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ----- Update Actor -----
        # Actor loss is defined as the negative of the critic’s Q-value,
        # so that maximizing Q is equivalent to minimizing the loss.
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----- Soft Update of Target Networks -----
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
    
    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
