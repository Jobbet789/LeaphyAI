import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(filename):
    """
    Load and plot reward data from a text file.
    
    Args:
        filename (str): Path to the text file containing rewards (one float per line)
    """
    try:
        # Load rewards from file
        with open(filename, 'r') as f:
            rewards = [float(line.strip()) for line in f if line.strip()]
        
        episodes = np.arange(1, len(rewards) + 1)
        
        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('ML Training Reward Analysis', fontsize=16)
        
        # Plot 1: Raw rewards over episodes
        ax1.plot(episodes, rewards, 'b-', alpha=0.6)
        ax1.plot(episodes, rewards, 'bo', markersize=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Raw Reward per Episode')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Moving average to show trend
        window_size = max(1, len(rewards) // 20)  # Adaptive window size
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ma_episodes = episodes[window_size-1:]
        
        ax2.plot(ma_episodes, moving_avg, 'r-')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward (Moving Average)')
        ax2.set_title(f'Moving Average Reward (Window Size: {window_size})')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add statistics as text
        stats_text = (
            f"Total Episodes: {len(rewards)}\n"
            f"Min Reward: {min(rewards):.2f}\n"
            f"Max Reward: {max(rewards):.2f}\n"
            f"Mean Reward: {np.mean(rewards):.2f}\n"
            f"Final Reward: {rewards[-1]:.2f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        # plt.savefig('reward_analysis.png', dpi=300)
        # print(f"Analysis complete. Plot saved as 'reward_analysis.png'")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Default filename
    filename = "rewards.txt"
    plot_rewards(filename)
