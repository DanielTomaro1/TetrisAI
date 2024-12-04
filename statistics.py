import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
from datetime import datetime

class GameStatistics:
    def __init__(self, window_size=100):
        # Game performance metrics
        self.scores = []
        self.lines_cleared = []
        self.avg_heights = []
        self.holes = []
        
        # Learning metrics
        self.rewards = []
        self.epsilon_values = []
        self.losses = []
        self.q_values = []
        
        # Episode information
        self.moves = 0
        self.game_duration = 0
        self.start_time = datetime.now()
        
        # Rolling averages
        self.window_size = window_size
        self.score_window = deque(maxlen=window_size)
        self.lines_window = deque(maxlen=window_size)
        self.reward_window = deque(maxlen=window_size)
        self.q_value_window = deque(maxlen=window_size)
        self.loss_window = deque(maxlen=window_size)

    def update(self, score, lines, avg_height, holes, reward=0, epsilon=0, loss=0, q_value=0):
        # Update game metrics
        self.scores.append(score)
        self.lines_cleared.append(lines)
        self.avg_heights.append(avg_height)
        self.holes.append(holes)
        
        # Update learning metrics
        self.rewards.append(reward)
        self.epsilon_values.append(epsilon)
        self.losses.append(loss)
        self.q_values.append(q_value)
        
        # Update rolling averages
        self.score_window.append(score)
        self.lines_window.append(lines)
        self.reward_window.append(reward)
        self.q_value_window.append(q_value)
        if loss != 0:
            self.loss_window.append(loss)
        
        self.moves += 1

    def generate_plots(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(20, 15))
        
        # 1. Game Performance Metrics
        plt.subplot(3, 2, 1)
        plt.plot(self.scores, label='Score', alpha=0.6)
        plt.plot(self.get_rolling_average(self.scores), label='Avg Score', linewidth=2)
        plt.title('Score Progress')
        plt.xlabel('Moves')
        plt.ylabel('Score')
        plt.legend()
        
        # 2. Learning Progress
        plt.subplot(3, 2, 2)
        plt.plot(self.rewards, label='Reward', alpha=0.6)
        plt.plot(self.get_rolling_average(self.rewards), label='Avg Reward', linewidth=2)
        plt.title('Learning Progress (Rewards)')
        plt.xlabel('Moves')
        plt.ylabel('Reward')
        plt.legend()
        
        # 3. Exploration vs Exploitation
        plt.subplot(3, 2, 3)
        plt.plot(self.epsilon_values)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Moves')
        plt.ylabel('Epsilon')
        
        # 4. Q-Values
        plt.subplot(3, 2, 4)
        plt.plot(self.q_values, label='Q-Value', alpha=0.6)
        plt.plot(self.get_rolling_average(self.q_values), label='Avg Q-Value', linewidth=2)
        plt.title('Q-Value Evolution')
        plt.xlabel('Moves')
        plt.ylabel('Q-Value')
        plt.legend()
        
        # 5. Loss Function
        if self.losses:
            plt.subplot(3, 2, 5)
            plt.plot(self.losses, label='Loss', alpha=0.6)
            plt.plot(self.get_rolling_average(self.losses), label='Avg Loss', linewidth=2)
            plt.title('Training Loss')
            plt.xlabel('Moves')
            plt.ylabel('Loss')
            plt.legend()
        
        # 6. Game State Analysis
        plt.subplot(3, 2, 6)
        plt.plot(self.avg_heights, label='Avg Height', alpha=0.6)
        plt.plot(self.holes, label='Holes', alpha=0.6)
        plt.plot(self.lines_cleared, label='Lines Cleared', alpha=0.6)
        plt.title('Game State Analysis')
        plt.xlabel('Moves')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'tetris_learning_stats_{timestamp}.png')
        plt.close()

    def get_rolling_average(self, data, window=100):
        ret = np.cumsum(data, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        return ret[window - 1:] / window

    def save_statistics(self):
        self.game_duration = (datetime.now() - self.start_time).total_seconds()
        
        stats = {
            'game_stats': {
                'total_moves': self.moves,
                'final_score': self.scores[-1] if self.scores else 0,
                'total_lines_cleared': sum(self.lines_cleared),
                'average_score': np.mean(self.scores),
                'average_lines_per_move': np.mean(self.lines_cleared),
                'game_duration_seconds': self.game_duration,
                'moves_per_second': self.moves / self.game_duration if self.game_duration > 0 else 0
            },
            'learning_stats': {
                'final_epsilon': self.epsilon_values[-1] if self.epsilon_values else 1.0,
                'average_reward': np.mean(self.rewards),
                'average_q_value': np.mean(self.q_values),
                'average_loss': np.mean(self.losses) if self.losses else 0,
                'total_training_steps': len(self.losses)
            },
            'performance_metrics': {
                'average_height': np.mean(self.avg_heights),
                'average_holes': np.mean(self.holes),
                'max_score': max(self.scores) if self.scores else 0,
                'max_lines_cleared_at_once': max(self.lines_cleared) if self.lines_cleared else 0
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'tetris_learning_stats_{timestamp}.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        return stats

    def print_training_summary(self):
        print("\nTraining Summary:")
        print("=" * 50)
        print(f"Total Moves: {self.moves}")
        print(f"Final Score: {self.scores[-1] if self.scores else 0}")
        print(f"Final Epsilon: {self.epsilon_values[-1]:.4f}")
        print(f"Average Reward: {np.mean(self.rewards):.2f}")
        print(f"Average Q-Value: {np.mean(self.q_values):.2f}")
        if self.losses:
            print(f"Final Loss: {self.losses[-1]:.4f}")
        print(f"Total Lines Cleared: {sum(self.lines_cleared)}")
        print("=" * 50)