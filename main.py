import socket
import threading
import json
import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from collections import deque

#from visualization import NetworkVisualizer

class Actor(nn.Module): # Known as DQN for the commented code
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Using tanh to bound actions between -1 and 1.
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # The critic takes both state and action as input.
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, state, action):
        # Concatenate state and action along the feature dimension.
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # For soft update of target parameters
        self.batch_size = 32
        
        # Replay buffer
        self.memory = deque(maxlen=2000)
        
        # Actor and Critic Networks
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        
        # Target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, noise_scale=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        self.actor.train()
        # Add exploration noise
        noise = noise_scale * np.random.randn(self.action_size)
        return np.clip(action + noise, -1, 1)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        minibatch = np.array(minibatch, dtype=object)
        states = torch.FloatTensor(np.array(minibatch[:, 0].tolist()))
        actions = torch.FloatTensor(np.array(minibatch[:, 1].tolist()))
        rewards = torch.FloatTensor(np.array(minibatch[:, 2].tolist())).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(minibatch[:, 3].tolist()))
        dones = torch.FloatTensor(np.array(minibatch[:, 4].tolist(), dtype=float)).unsqueeze(1)
        
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

# ----- Training Server (Modified to Use the DDPG Agent) -----

class TrainingServer:
    def __init__(self, use_saved_models):
        # Adjust state_size and action_size as needed.
        self.agent = DDPGAgent(state_size=2, action_size=2)
        if use_saved_models:
            self.agent.actor.load_state_dict(torch.load('actor_model.pth'))
            self.agent.critic.load_state_dict(torch.load('critic_model.pth'))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('localhost', 65432))
        self.sock.listen(1)
        print("Training server started on port 65432")
        #self.visualizer = NetworkVisualizer(self.agent.actor)
        #self.visualizer.start()
    
    def handle_client(self, conn):
        prev_state = None
        prev_action = None
        episode = 0 
        total_episodes = 100
        
        while True:
            try:
                while episode < total_episodes:
                    # Receive state from client
                    data = conn.recv(1024).decode()
                    if not data:
                        break
                    try:
                        msg = json.loads(data)
                        state = msg['state']
                        reward = msg['reward']
                        done = msg['done']
                    except (KeyError, json.JSONDecodeError):
                        continue
                    
                    # Store experience and train
                    if prev_state is not None:
                        self.agent.remember(prev_state, prev_action, reward, state, done)
                        self.agent.replay()
                    
                    # Get next action (with exploration noise)
                    action = self.agent.act(state)
                    
                    # Send action to client
                    conn.send(json.dumps(action.tolist()).encode())
                    
                    # Update previous state and action
                    prev_state = state
                    prev_action = action
                    
                    if done:
                        episode += 1
                        prev_state = None
                        prev_action = None
                        print(f"Episode {episode}/{total_episodes}")
                
                torch.save(self.agent.actor.state_dict(), 'actor_model.pth')
                torch.save(self.agent.critic.state_dict(), 'critic_model.pth')
                print("Models saved")
                episode = 0
                        
            except ConnectionResetError:
                pass

            finally:
                conn.close()
                torch.save(self.agent.actor.state_dict(), 'actor_model.pth')
                torch.save(self.agent.critic.state_dict(), 'critic_model.pth')
                print("Models saved")
    
    def run(self):
        conn, addr = self.sock.accept()
        print(f"Connected to {addr}")
        # client_thread = threading.Thread(target=self.handle_client, args=(conn,))
        # client_thread.start()
        self.handle_client(conn)
        self.sock.close()


            

if __name__ == "__main__":
    if input("> Use saved models? (y/n): ").lower() == 'y':
        server = TrainingServer(True)
    else:
        server = TrainingServer(False)
    server.run()