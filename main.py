import argparse
import os
import sys

# Constants
RANDOM_SEED = 42
EPISODES = 1000000
MAX_STEPS = 250
SAVE_MODEL_EVERY = 1000
PRINT_EVERY = 10
SIMULATION_SPEED = 5

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train or test DDPG agent')
    parser.add_argument('mode', type=str, choices=['train', 'test', 'play', 'resume'], 
                        help='Mode to run the agent')
    parser.add_argument('--model', type=str, default='best_actor.pth', 
                        help='Model path for testing')
    parser.add_argument('--episodes', type=int, default=10, 
                        help='Number of episodes for testing')
    parser.add_argument('--clean', action='store_true', 
                        help='Clean up the training directory')
    parser.add_argument('--reset_noise', action='store_true', 
                        help='Reset noise scale when resuming training')
    parser.add_argument('--checkpoint', type=int, default=None, 
                        help='Checkpoint episode to resume training from')
    parser.add_argument('--model_only', action='store_true', 
                        help='Load only the models from the checkpoint')
    return parser.parse_args()

class CheckpointHandler:
    """Handles saving and loading training checkpoints"""
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        # Create the checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, agent, episode, noise_scale, all_rewards, moving_avg_rewards, best_reward):
        """Save the current training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pkl")
        model_path = os.path.join(self.checkpoint_dir, f"models_episode_{episode}")
        os.makedirs(model_path, exist_ok=True)
        
        # Save models (move to CPU before saving)
        torch.save(agent.actor.cpu().state_dict(), os.path.join(model_path, "actor.pth"))
        torch.save(agent.critic.cpu().state_dict(), os.path.join(model_path, "critic.pth"))
        torch.save(agent.actor_target.cpu().state_dict(), os.path.join(model_path, "actor_target.pth"))
        torch.save(agent.critic_target.cpu().state_dict(), os.path.join(model_path, "critic_target.pth"))
    
        # Make sure to move models back to GPU after saving
        agent.actor.to(agent.device)
        agent.critic.to(agent.device)
        agent.actor_target.to(agent.device)
        agent.critic_target.to(agent.device)
        
        # Save optimizer states
        torch.save(agent.actor_optimizer.state_dict(), os.path.join(model_path, "actor_optimizer.pth"))
        torch.save(agent.critic_optimizer.state_dict(), os.path.join(model_path, "critic_optimizer.pth"))
        
        # Save replay buffer
        replay_buffer = [tuple(exp) for exp in agent.memory.memory]

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
    
    def load_checkpoint(self, checkpoint_path, agent, model_only=False):
        """Load a training checkpoint"""
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
        agent.actor_target.load_state_dict(torch.load(os.path.join(model_path, "actor_target.pth"), map_location=agent.device))
        agent.critic_target.load_state_dict(torch.load(os.path.join(model_path, "critic_target.pth"), map_location=agent.device))
        
        if not model_only:
            # Restore optimizer states
            agent.actor_optimizer.load_state_dict(torch.load(os.path.join(model_path, "actor_optimizer.pth")))
            agent.critic_optimizer.load_state_dict(torch.load(os.path.join(model_path, "critic_optimizer.pth")))
            
            # Restore replay buffer
            # Convert back to deque of Experiences
            agent.memory.memory = deque([agent.memory.experience(*exp) for exp in checkpoint_data['replay_buffer']], 
                                        maxlen=agent.buffer_size)
            
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
        # If model_only, return None to indicate no training state was loaded
        return None
    
    def get_checkpoint(self, episode=-1):
        """Get the path to a specific checkpoint file or the latest one"""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                           if f.startswith("checkpoint_") and f.endswith(".pkl")]
        
        if not checkpoint_files:
            return None
        
        # Extract episode numbers and find the requested one or the latest
        episode_nums = [int(f.split("_")[2].split(".")[0]) for f in checkpoint_files]
        if episode == -1:
            idx = episode_nums.index(max(episode_nums))
        else:
            try:
                idx = episode_nums.index(episode)
            except ValueError:
                return None
        
        return os.path.join(self.checkpoint_dir, checkpoint_files[idx])
    
    def load_training_state(self, checkpoint_episode, agent, model_only=False):
        """Load training state from a checkpoint"""
        checkpoint_path = self.get_checkpoint(checkpoint_episode)
        if checkpoint_path:
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            return self.load_checkpoint(checkpoint_path, agent, model_only)
        return None

class TrainingManager:
    """Manages the training process for the DDPG agent"""
    def __init__(self):
        # Set seeds for reproducibility
        self._set_seeds()
        self.checkpoint_handler = CheckpointHandler()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
    
    def train(self, checkpoint=None, reset_noise=False, model_only=False):
        """Train the agent"""
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
        noise_scale = max(0.01, 0.5 - (0.0008 * episode_start))
        all_rewards = []
        moving_avg_rewards = deque(maxlen=100)
        best_reward = float('-inf')
        
        # Try to resume training if requested
        if checkpoint is not None:
            training_data = self.checkpoint_handler.load_training_state(
                checkpoint, agent, model_only)
            
            if training_data:
                episode_start, noise_scale, all_rewards, moving_avg_rewards, best_reward = training_data
                
                if reset_noise:
                    noise_scale = 0.5
                    episode_start = 0
                
                episode_start += 1  # Start from the next episode
                print(f"Resumed from episode {episode_start-1}, with best reward: {best_reward:.2f}")
            else:
                print("No checkpoint found. Starting training from scratch.")
        
        print("Starting training...")
        print(f"State size: {state_size}, Action size: {action_size}")
        
        try:
            # Training loop
            self._run_training_loop(
                agent, game, episode_start, noise_scale, all_rewards, 
                moving_avg_rewards, best_reward
            )
            
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            # Current episode is the last one we were working on
            current_episode = episode_start
            self.checkpoint_handler.save_checkpoint(
                agent, current_episode, noise_scale, all_rewards,
                moving_avg_rewards, best_reward
            )
            print(f"Checkpoint saved at episode {current_episode}. Exiting...")
            raise
        
        return agent, episode_start, noise_scale, all_rewards, moving_avg_rewards, best_reward
    
    def _run_training_loop(self, agent, game, episode_start, noise_scale, 
                          all_rewards, moving_avg_rewards, best_reward):
        """Execute the training loop"""
        for episode in range(episode_start, EPISODES + 1):
            # Reset the environment
            game.reset_simulation()
            state = game.get_state()
            
            # Decaying noise for exploration
            if episode > episode_start:
                noise_scale = max(0.01, 0.5 - (0.0008 * episode))
            
            episode_reward = 0
            done = False
            step = 0
            
            start_time = time.time()
            
            # Episode loop
            while not done and step < MAX_STEPS:
                # Select and take action
                action = agent.act(np.array(state), noise_scale)[0]
                next_state, reward, done = game.action(action[0], action[1])
                
                # Store experience and learn
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                # Update current state and total reward
                state = next_state
                episode_reward += reward
                step += 1
            
            # Process episode results
            self._process_episode_results(
                episode, episode_reward, step, noise_scale, start_time,
                all_rewards, moving_avg_rewards, best_reward, agent
            )
            
            # Save checkpoint periodically
            if episode % SAVE_MODEL_EVERY == 0:
                self.checkpoint_handler.save_checkpoint(
                    agent, episode, noise_scale, all_rewards, 
                    moving_avg_rewards, best_reward
                )
                print(f"Model and checkpoint saved at episode {episode}")
    
    def _process_episode_results(self, episode, episode_reward, step, noise_scale, 
                                start_time, all_rewards, moving_avg_rewards, 
                                best_reward, agent):
        """Process the results of an episode"""
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
    
    def test(self, model_path='best_actor.pth', episodes=10):
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
        agent.actor.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.actor.eval()  # Set to evaluation mode
        
        print(f"Testing model: {model_path}")
        
        for episode in range(1, episodes + 1):
            self._run_test_episode(game, agent, episode, episodes)
        
        pygame.quit()
    
    def _run_test_episode(self, game, agent, episode, total_episodes):
        """Run a single test episode"""
        game.reset_simulation()
        state = game.get_state()
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < MAX_STEPS * SIMULATION_SPEED:
            # Get action without exploration noise
            action = agent.act(np.array(state), noise_scale=0.0)[0]
            
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
        
        print(f"Episode {episode}/{total_episodes} - Reward: {episode_reward:.2f}, Steps: {step}")
    
    def play(self, episodes=10):
        """Play the game manually for reward function testing"""
        # Initialize the game in rendered mode
        game = Game(rendered=True)
        
        print("Manual play mode for reward function testing")
        print("Controls: W,A,S,D to move | R to reset")
        
        for episode in range(1, episodes + 1):
            self._run_play_episode(game, episode, episodes)
        
        pygame.quit()
    
    def _run_play_episode(self, game, episode, total_episodes):
        """Run a single play episode"""
        game.reset_simulation()
        
        episode_reward = 0
        done = False
        step = 0
        
        # Reset total reward for this episode
        game.total_reward = 0
        
        print(f"Episode {episode}/{total_episodes} - Starting. Press R to reset early.")
        
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
        
        print(f"Episode {episode}/{total_episodes} completed - Final Reward: {game.total_reward:.2f}, Steps: {step}")

def clean_training_dir():
    """Clean up the training directory"""
    # Remove checkpoints
    if os.path.exists("checkpoints"):
        for f in os.listdir("checkpoints"):
            # If it's a file, remove it, if it's a directory remove it and its contents 
            if os.path.isfile(os.path.join("checkpoints", f)):
                os.remove(os.path.join("checkpoints", f))
            else:
                for ff in os.listdir(os.path.join("checkpoints", f)):
                    os.remove(os.path.join("checkpoints", f, ff))
                os.rmdir(os.path.join("checkpoints", f))
        
    # Remove models, 'best_critic.pth' and 'best_actor.pth'
    if os.path.exists("best_critic.pth"):
        os.remove("best_critic.pth")
    if os.path.exists("best_actor.pth"):
        os.remove("best_actor.pth")
        
    # Remove rewards file
    if os.path.exists("rewards.txt"):
        os.remove("rewards.txt")
        
    print("Training directory cleaned.")

if __name__ == "__main__":
    # Parse command line arguments first, before importing heavy modules
    args = parse_arguments()
    
    # Now import the heavy modules
    import pygame
    import time
    import numpy as np
    import torch
    import random
    from collections import deque
    import pickle
    
    # Import your game and agent
    from game import Game
    from ddpg_agent import DDPGAgent
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize pygame
    pygame.init()
    
    try:
        # Create training manager
        manager = TrainingManager()
        
        # Execute based on command line arguments
        if args.clean:
            clean_training_dir()
            
        if args.mode == 'train':
            manager.train(checkpoint=args.checkpoint, reset_noise=args.reset_noise, model_only=args.model_only)
        elif args.mode == 'resume':
            manager.train(checkpoint=-1, reset_noise=args.reset_noise, model_only=args.model_only)
        elif args.mode == 'test':
            manager.test(args.model, args.episodes)
        elif args.mode == 'play':
            manager.play(args.episodes)
        
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        pygame.quit()
        print("Training ended.")
