import argparse
import os
import sys

# Constants
RANDOM_SEED = 42
EPISODES = int(1e6)
MAX_STEPS = 300
PRINT_EVERY = 10
SIMULATION_SPEED = 5

def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Soft Actor-Critic (SAC) algorithm")
    parser.add_argument("mode", type=str, choices=['train', 'test'], help="Select the mode of the algorithm")
    parser.add_argument("--resume", action="store_true", help="Resume training from a saved model", default=False)

    return parser.parse_args()

class TrainingManager:
    # Manages the training process for the agent
    def __init__(self):
        # Set seeds for reproducibility
        self._set_seeds()
        self.best_reward = float('-inf')

    def _set_seeds(self):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    def train(self, continue_training=False):
        # Train the agent
        # Initialize the game 
        game = Game(rendered=False, physics_steps=SIMULATION_SPEED)

        # Get state size and action size
        state = game.get_state()
        state_size = len(state)
        action_size = 2 # I'm lazy

        action_high = 1.0 # Temp

        # Initialize the agent
        agent = SACAgent(state_size,
                         action_size,
                         action_high)

        if continue_training:
            agent.load("best_model")

        episode_start = 1
        all_rewards = []
        moving_avg_rewards = deque(maxlen=100)
        best_reward = float('-inf')

        print("Starting training...")
        print(f"State size: {state_size}, Action size: {action_size}")

        try:
            # Training loop
            self._run_training_loop(
                    agent, game, episode_start,
                    all_rewards, moving_avg_rewards
            )

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

    def _run_training_loop(self, agent, game, episode_start, all_rewards, 
                           moving_avg_rewards):
        ball_count = 1
        print(f"Starting with {ball_count} ball{'s' if ball_count > 1 else ''}...")

        # Run the training loop
        for episode in range(episode_start, EPISODES + 1):
            # Reset the env
            game.reset_simulation(ball_count=ball_count)
            state = game.get_state()

            episode_reward = 0
            done = False
            step = 0

            start_time = time.time()

            # Episode loop
            while not done and step < MAX_STEPS:
                # Select and take action
                action = agent.act(np.array(state))
                next_state, reward, done = game.action(action[0], action[1])

                # Store experience and learn
                agent.remember(state, action, reward, next_state, done)
                agent.update()

                # Update current stae and total reward
                state = next_state
                episode_reward += reward
                step += 1

            if self._process_episode(
                    episode, episode_reward, step, start_time,
                    all_rewards, moving_avg_rewards, agent, ball_count):
                if ball_count <= 3:
                    ball_count += 1
                    # remove replay buffer
                    agent.memory.buffer.clear()
                    print(f"Adding a ball. Total balls: {ball_count}")


    def _process_episode(self, episode, episode_reward, step, start_time,
                         all_rewards, moving_avg_rewards, agent, ball_count):
        episode_duration = time.time() - start_time

        all_rewards.append(episode_reward)
        moving_avg_rewards.append(episode_reward)
        avg_reward = sum(moving_avg_rewards) / len(moving_avg_rewards)

        with open("rewards.txt", "a") as f:
            f.write(f"{episode_reward}\n")

        if episode % PRINT_EVERY == 0:
            print(f"Episode: {episode}/{EPISODES} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Moving Average: {avg_reward:.2f} | "
                  f"Steps: {step} | "
                  f"Time: {episode_duration:.2f}s")

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            agent.save("best_model")
            print(f"New best model saved with avg reward: {self.best_reward:.2f}")

            return avg_reward >= (ball_count * 200 + 300)


class TestingManager:
    @staticmethod
    def test(model_path="best_model"):
        # Test the agent
        game = Game(rendered=True, physics_steps=SIMULATION_SPEED)

        state = game.get_state()
        state_size = len(state)
        action_size = 2
        action_high = 1.0

        agent = SACAgent(state_size, action_size, action_high, hidden_size_1=512, hidden_size_2=512)
        agent.load(model_path)

        print("Testing the agent...")

        try:
            TestingManager._run_testing_loop(game, agent)
        except KeyboardInterrupt:
            print("\nTesting interrupted.")

    @staticmethod
    def _run_testing_loop(game, agent):
        while True:
            game.reset_simulation(ball_count=2)
            state = game.get_state()

            done = False
            step = 0 
            episode_reward = 0

            while not done and step < MAX_STEPS:
                action = agent.act(np.array(state), evaluate=True)
                next_state, reward, done = game.action(action[0], action[1])

                state = next_state
                episode_reward += reward
                step += 1

                game.draw()

                pygame.time.delay(20)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            print(f"Episode reward: {episode_reward:.2f}")

if __name__ == '__main__':
    args = parse_arguments()

    # Now import the heavy modules
    import pygame
    import time
    import numpy as np
    import torch
    import random
    from collections import deque

    # Import local packages
    from game import Game
    from SACAgent import SACAgent

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init pygame
    pygame.init()

    try:
        manager = TrainingManager()

        if args.mode == 'train':
            manager.train(continue_training=args.resume)
        elif args.mode == 'test':
            TestingManager.test()

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        pygame.quit()
        print("Program terminated.")
