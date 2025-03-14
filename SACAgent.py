# IMPOOORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal 
import random
from collections import deque, namedtuple

# Use CUDA when avail
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# Replay buffer to store exp
class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_state = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_state, dones)
    
    def __len__(self):
        return len(self.buffer)


# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size_1=256, hidden_size_2=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        self.fc1 = nn.Linear(state_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)

        self.mu = nn.Linear(hidden_size_2, action_dim)
        self.log_std = nn.Linear(hidden_size_2, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std


    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()

        # Use reparameterization trick
        normal = Normal(mu, std)
        x_t = normal.rsample() # Sample reparameterization

        # Apply tanh squashing to ensure action bounds
        y_t = torch.tanh(x_t)

        # Calculate log probability, adding correction term for the tanh squashing
        log_prob = normal.log_prob(x_t)
        # Correction for tanh squashing
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return y_t, log_prob, mu

# Critic Network (Q-function) with two hidden layers (SAC)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size_1=256, hidden_size_2=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.q1 = nn.Linear(hidden_size_2, 1)

        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.fc4 = nn.Linear(hidden_size_1, hidden_size_2)
        self.q2 = nn.Linear(hidden_size_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.q1(q1))

        # Q2
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = F.relu(self.q2(q2))

        return q1, q2


    def q1_forward(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.q1(q1))

        return q1

class SACAgent:
    def __init__(self,
                state_dim,
                action_dim,
                action_high,
                hidden_size_1=256,
                hidden_size_2=256,
                buffer_size=int(1e6),
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                lr=3e-4,
                automatic_entropy_tuning=True):
    
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Init actor net
        self.actor = Actor(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Init critic nets
        self.critic = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Init target critic nets
        self.critic_target = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        # Hard copy params
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Init replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if evaluate:
            # Use deterministic action for eval
            with torch.no_grad():
                _, _, mu = self.actor.sample(state)
                return mu.cpu().numpy()[0] * self.action_high
        else:
            # Use stochastic action for training
            with torch.no_grad():
                action, _, _, = self.actor.sample(state)
                return action.cpu().numpy()[0] * self.action_high
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action / self.action_high, reward, next_state, done)
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from buffer
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Update critic nets
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Calculate critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor net
        new_actions, log_probs, _ = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)

        # Calculate actor loss
        actor_loss = (self.alpha * log_probs - min_q).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update entropy coefficient (alpha)
        if self.automatic_entropy_tuning:
            alpha_loss = -((self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean())

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target critic networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def save(self, filepath, with_buffer=False):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None
        }, filepath)

        if with_buffer:
            with open(f"{filepath}_buffer.pkl", 'wb') as f:
                pickle.dump(self.memory)
        
    def load(self, filepath, with_buffer=False):
        checkpoint = torch.load(filepath)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()

        if with_buffer:
            with open(f"{filepath}_buffer.pkl", 'rb') as f:
                self.memory = pickle.load(f)

"""
if __name__ == "__main__":
    # Example with a continuous action space environment
    state_dim = 17  # Example state dimension
    action_dim = 6  # Example action dimension
    action_high = 1.0  # Example action bounds
    
    # Initialize agent
    agent = SACAgent(state_dim=state_dim, 
                     action_dim=action_dim, 
                     action_high=action_high,
                     hidden_size_1=256,
                     hidden_size_2=256,
                     buffer_size=1000000,
                     batch_size=256,
                     gamma=0.99,
                     tau=0.005,
                     automatic_entropy_tuning=True)
    
    print(f"SAC Agent initialized on device: {device}")

"""
