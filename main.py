import pygame
import sys
import time
import numpy as np
import torch
import random
from collections import deque
import os
import pickle
import argparse

# Import your game and agent
from game import Game
from ddpg_agent import DDPGAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Training hyperparameters
EPISODES = 1000000
MAX_STEPS = 200
SAVE_MODEL_EVERY = 1000  # Save model weights every N episodes
PRINT_EVERY = 10  # Print stats every N episodes
SIMULATION_SPEED = 5


class TrainingCheckpoint:
    """
    A class to handle saving and loading training checkpoints
    """
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        # Create the checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, agent, episode, noise_scale, all_rewards, moving_avg_rewards, best_reward):
        """
        Save the current training state
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pkl")
        model_path = os.path.join(self.checkpoint_dir, f"models_episode_{episode}")
        os.makedirs(model_path, exist_ok=True)
        
        # Save models (move to CPU before saving)
        torch.save(agent.actor.cpu().state_dict(), os.path.join(model_path, "actor.pth"))
        torch.save(agent.critic.cpu().state_dict(), os.path.join(model_path, "critic.pth"))
        torch.save(agent.target_actor.cpu().state_dict(), os.path.join(model_path, "target_actor.pth"))
        torch.save(agent.target_critic.cpu().state_dict(), os.path.join(model_path, "target_critic.pth"))
    
        # Make sure to move models back to GPU after saving
        agent.actor.to(agent.device)
        agent.critic.to(agent.device)
        agent.target_actor.to(agent.device)
        agent.target_critic.to(agent.device)
        
        # Save optimizer states
        torch.save(agent.actor_optimizer.state_dict(), os.path.join(model_path, "actor_optimizer.pth"))
        torch.save(agent.critic_optimizer.state_dict(), os.path.join(model_path, "critic_optimizer.pth"))
        
        # Save replay buffer
        replay_buffer = list(agent.memory)
        
        # Save training state (excluding PyTorch models which are saved separately)
        checkpoint_data = {
            'episode': episode,
            'noise_scale': noise_scale,
            'all_rewards': all_rewards,
            'moving_avg_rewards': list(moving_avg_rewards),
            'best_reward': best_reward,
            'replay_buffer': replay_buffer,
            'random_state': {
                'random': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state()
            }
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self, checkpoint_path, agent):
        """
        Load a training checkpoint
        """
        # Extract episode number from checkpoint path for loading the corresponding models
        checkpoint_file = os.path.basename(checkpoint_path)
        episode_num = int(checkpoint_file.split("_")[2].split(".")[0])
        model_path = os.path.join(self.checkpoint_dir, f"models_episode_{episode_num}")
        
        # Load the checkpoint data
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Load models directly to the device
        agent.actor.load_state_dict(torch.load(os.path.join(model_path, "actor.pth"), map_location=agent.device))
        agent.critic.load_state_dict(torch.load(os.path.join(model_path, "critic.pth"), map_location=agent.device))
        agent.target_actor.load_state_dict(torch.load(os.path.join(model_path, "target_actor.pth"), map_location=agent.device))
        agent.target_critic.load_state_dict(torch.load(os.path.join(model_path, "target_critic.pth"), map_location=agent.device))
        
        # Restore optimizer states
        agent.actor_optimizer.load_state_dict(torch.load(os.path.join(model_path, "actor_optimizer.pth")))
        agent.critic_optimizer.load_state_dict(torch.load(os.path.join(model_path, "critic_optimizer.pth")))
        
        # Restore replay buffer
        agent.memory = deque(checkpoint_data['replay_buffer'], maxlen=agent.memory.maxlen)
        
        # Restore random states for reproducibility
        random.setstate(checkpoint_data['random_state']['random'])
        np.random.set_state(checkpoint_data['random_state']['numpy'])
        torch.set_rng_state(checkpoint_data['random_state']['torch'])
        
        # Return the training state variables
        return (
            checkpoint_data['episode'],
            checkpoint_data['noise_scale'],
            checkpoint_data['all_rewards'],
            deque(checkpoint_data['moving_avg_rewards'], maxlen=100),
            checkpoint_data['best_reward']
        )
    
    def get_latest_checkpoint(self):
        """
        Get the path to the latest checkpoint file
        """
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
        
        if not checkpoint_files:
            return None
        
        # Extract episode numbers and find the latest
        episode_nums = [int(f.split("_")[2].split(".")[0]) for f in checkpoint_files]
        latest_idx = episode_nums.index(max(episode_nums))
        
        return os.path.join(self.checkpoint_dir, checkpoint_files[latest_idx])

def train(resume_training=False):
    # Initialize the checkpointing system
    checkpoint_handler = TrainingCheckpoint()
    
    # Initialize the game (non-rendered for faster training)
    game = Game(rendered=False, physics_steps=SIMULATION_SPEED)
    
    # Get state size and action size
    state = game.get_state()
    state_size = len(state)
    action_size = 2
    
    # Initialize the agent
    agent = DDPGAgent(state_size, action_size)
    
    # Initialize training variables
    episode_start = 1
    noise_scale = max(0.01, 0.5 - (0.4 * episode_start / 2000))
    all_rewards = []
    moving_avg_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    
    # Try to resume training if requested
    if resume_training:
        latest_checkpoint = checkpoint_handler.get_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            episode_start, noise_scale, all_rewards, moving_avg_rewards, best_reward = checkpoint_handler.load_checkpoint(latest_checkpoint, agent)
            episode_start += 1  # Start from the next episode
            print(f"Resumed from episode {episode_start-1}, with best reward: {best_reward:.2f}")
        else:
            print("No checkpoint found. Starting training from scratch.")
    
    print("Starting training...")
    print(f"State size: {state_size}, Action size: {action_size}")
    
    try:
        # Training loop
        for episode in range(episode_start, EPISODES + 1):
            # Reset the environment
            game.reset_simulation()
            state = game.get_state()
            
            # Decaying noise for exploration (continue from the current noise_scale)
            if episode > episode_start:
                noise_scale = max(0.01, 0.5 - (0.4 * episode / 2000))
            
            episode_reward = 0
            done = False
            step = 0
            
            start_time = time.time()

            
            # Episode loop
            while not done and step < MAX_STEPS:
                # Select an action
                action = agent.act(state, noise_scale)
                
                # Take action in the environment
                next_state, reward, done = game.action(action[0], action[1])
                
                # Store experience in replay memory
                agent.remember(state, action, reward, next_state, done)
                
                # Learn from experiences
                agent.replay()
                
                # Update current state and total reward
                state = next_state
                episode_reward += reward
                step += 1
            
            # Calculate episode duration
            episode_duration = time.time() - start_time
            
            # Track rewards
            all_rewards.append(episode_reward)
            moving_avg_rewards.append(episode_reward)
            avg_reward = sum(moving_avg_rewards) / len(moving_avg_rewards)
            
            # Save total reward to file
            with open("rewards.txt", "a") as f:
                f.write(f"{episode_reward}\n")
            
            # Print episode stats
            if episode % PRINT_EVERY == 0:
                print(f"Episode: {episode}/{EPISODES} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Moving Avg: {avg_reward:.2f} | "
                      f"Steps: {step} | "
                      f"Noise: {noise_scale:.2f} | "
                      f"Time: {episode_duration:.2f}s")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.actor.state_dict(), 'best_actor.pth')
                torch.save(agent.critic.state_dict(), 'best_critic.pth')
                print(f"New best model saved with avg reward: {best_reward:.2f}")
            
            # Save model and checkpoint periodically
            if episode % SAVE_MODEL_EVERY == 0:
                # Save checkpoint
                checkpoint_handler.save_checkpoint(
                    agent, episode, noise_scale, all_rewards, 
                    moving_avg_rewards, best_reward
                )
                print(f"Model and checkpoint saved at episode {episode}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        # Save checkpoint with current episode number
        checkpoint_handler.save_checkpoint(
            agent, episode, noise_scale, all_rewards,
            moving_avg_rewards, best_reward
        )
        print(f"Checkpoint saved at episode {episode}. Exiting...")
        raise  # Re-raise the KeyboardInterrupt to be caught by the outer try/except
    
    return agent, episode, noise_scale, all_rewards, moving_avg_rewards, best_reward

def test(model_path='best_actor.pth', episodes=10):
    """Test the trained model"""
    # Initialize the game in rendered mode
    game = Game(rendered=True)
    
    # Get state size and action size
    state = game.get_state()
    state_size = len(state)
    action_size = 2
    
    # Initialize the agent
    agent = DDPGAgent(state_size, action_size)
    
    # Load pre-trained model
    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()  # Set to evaluation mode
    
    print(f"Testing model: {model_path}")
    
    for episode in range(1, episodes + 1):
        game.reset_simulation()
        state = game.get_state()
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < MAX_STEPS * SIMULATION_SPEED:
            # Get action without exploration noise
            action = agent.act(state, noise_scale=0.0)
            
            # Take action
            next_state, reward, done = game.action(action[0], action[1])
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
            
            # Draw the game state
            game.draw()
            
            # Add a small delay to make rendering visible
            pygame.time.delay(20)
            
            # Process any quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        
        print(f"Episode {episode}/{episodes} - Reward: {episode_reward:.2f}, Steps: {step}")
    
    pygame.quit()

def play(episodes=10):
    """Play the game manually for reward function testing"""
    # Initialize the game in rendered mode
    game = Game(rendered=True)
    
    print("Manual play mode for reward function testing")
    print("Controls: W,A,S,D to move | R to reset")
    
    for episode in range(1, episodes + 1):
        game.reset_simulation()
        
        episode_reward = 0
        done = False
        step = 0
        
        # Reset total reward for this episode
        game.total_reward = 0
        
        print(f"Episode {episode}/{episodes} - Starting. Press R to reset early.")
        
        while not done and step < MAX_STEPS * SIMULATION_SPEED:
            # Process events and handle keyboard input
            game.process_events()
            
            # Check if the game is done after processing events
            done = game.done
            
            # Draw the game state
            game.draw()
            
            # Track steps
            step += 1
            
            # Process any quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("Game manually closed.")
                    return
                # Check if reset was pressed
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print(f"Manual reset after {step} steps with reward: {game.total_reward:.2f}")
                    done = True
                    break
        
        print(f"Episode {episode}/{episodes} completed - Final Reward: {game.total_reward:.2f}, Steps: {step}")
    
    pygame.quit()

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test DDPG agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'play'], 
                        help='Mode: train, test, or play!')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--model', type=str, default='best_actor.pth',
                        help='Model path for testing')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes for testing')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            train(resume_training=args.resume)
        elif args.mode == 'test':
            test(model_path=args.model, episodes=args.episodes)
        elif args.mode == 'play':
            play(episodes=args.episodes)
    except KeyboardInterrupt:
        # We just need to exit gracefully here
        print("\nExiting...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pygame.quit()
        print("Training ended.")
