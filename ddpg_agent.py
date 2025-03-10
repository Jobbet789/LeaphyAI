import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple

# Define device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=(400, 300), init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        
        # Initialize the final layer with smaller weights to ensure initial outputs are near zero
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output in range (-1, 1) for continuous action space
        return torch.tanh(self.fc3(x))

# Critic Network (Value)
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=(400, 300), init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
        # Initialize the final layer with smaller weights
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

# DDPG Agent implementation
class DDPGAgent:
    def __init__(self, state_size, action_size, 
                 buffer_size=int(1e6), 
                 batch_size=64, 
                 gamma=0.99, 
                 tau=1e-3, 
                 lr_actor=1e-4, 
                 lr_critic=1e-3,
                 weight_decay=0):
        
        self.device = device 
        # Initialize parameters
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update parameter
        
        # Actor Networks (local and target)
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic Networks (local and target)
        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # Make sure target and local networks start with the same weights
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Noise process for exploration
        self.noise = OUNoise(action_size)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        # Step counter for learning frequency
        self.t_step = 0
        self.learn_every = 1  # Learn every n steps
        
    def act(self, state, noise_scale=0.1):
        """Get action based on current policy with optional noise for exploration"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Set to evaluation mode (disables dropout if any)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        # Set back to training mode
        self.actor.train()
        
        # Add noise for exploration
        if noise_scale > 0:
            noise = self.noise.sample() * noise_scale
            action += noise
            
        # Clip action to be between -1 and 1
        return np.clip(action, -1, 1)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.learn_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.replay()
    
    def replay(self):
        """Sample from replay buffer and update actor and critic networks"""
        # If not enough samples, return
        if len(self.memory) < self.batch_size:
            return
            
        # Get random batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for critic to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def hard_update(self, target_model, local_model):
        """Hard update model parameters: θ_target = θ_local"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            
    def save(self, actor_path, critic_path):
        """Save the models"""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
    def load(self, actor_path, critic_path):
        """Load the models"""
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        # Update target networks to match loaded models
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)