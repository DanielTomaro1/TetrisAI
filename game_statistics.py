import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import json
from datetime import datetime
import pandas as pd
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class TrainingMetrics:
    episode: int
    score: float
    lines_cleared: int
    avg_height: float
    holes: int
    reward: float = 0.0
    epsilon: float = 0.0
    loss: float = 0.0
    q_value: float = 0.0
    generation: int = 0
    member: int = 0
    well_count: int = 0
    surface_smoothness: float = 0.0
    deep_holes: float = 0.0

class GameStatistics:
    def __init__(self, window_size: int = 100):
        # Create output directories
        os.makedirs('outputs/plots', exist_ok=True)
        os.makedirs('outputs/stats', exist_ok=True)
        os.makedirs('outputs/plots/animated', exist_ok=True)
        os.makedirs('outputs/reports', exist_ok=True)
        
        # Enhanced performance metrics
        self.scores: List[float] = []
        self.lines_cleared: List[int] = []
        self.avg_heights: List[float] = []
        self.holes: List[int] = []
        self.deep_holes: List[float] = []
        self.well_formations: List[int] = []
        self.surface_smoothness: List[float] = []
        
        # Learning metrics
        self.rewards: List[float] = []
        self.epsilon_values: List[float] = []
        self.losses: List[float] = []
        self.q_values: List[float] = []
        
        # Enhanced genetic metrics
        self.generation_stats: List[Dict] = []
        self.member_stats: List[Dict] = []
        self.population_diversity: List[float] = []
        self.mutation_effects: List[Dict] = []
        self.generation_metrics: List[Dict] = []
        
        # Generation tracking
        self.current_generation: int = 0
        self.generation_scores: Dict[int, List[float]] = {}
        self.generation_lines: Dict[int, List[int]] = {}
        self.generation_fitness: Dict[int, List[float]] = {}
        
        # Episode information
        self.moves: int = 0
        self.game_duration: float = 0
        self.start_time = datetime.now()
        
        # Rolling averages
        self.window_size = window_size
        self.score_window = deque(maxlen=window_size)
        self.lines_window = deque(maxlen=window_size)
        self.reward_window = deque(maxlen=window_size)
        self.q_value_window = deque(maxlen=window_size)
        self.loss_window = deque(maxlen=window_size)
        
        # Performance tracking
        self.best_scores_per_generation: List[float] = []
        self.avg_scores_per_generation: List[float] = []
        self.worst_scores_per_generation: List[float] = []
        
        # Session identification
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join('outputs/reports', self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)

    def update(self, metrics: TrainingMetrics):
        """Update all statistics with new metrics"""
        # Update game metrics
        self.scores.append(metrics.score)
        self.lines_cleared.append(metrics.lines_cleared)
        self.avg_heights.append(metrics.avg_height)
        self.holes.append(metrics.holes)
        self.deep_holes.append(metrics.deep_holes)
        self.well_formations.append(metrics.well_count)
        self.surface_smoothness.append(metrics.surface_smoothness)
        
        # Update generation tracking
        if metrics.generation != self.current_generation:
            if self.current_generation in self.generation_scores:
                # Save generation statistics
                self.generation_stats.append({
                    'generation': self.current_generation,
                    'avg_score': np.mean(self.generation_scores[self.current_generation]),
                    'max_score': max(self.generation_scores[self.current_generation]),
                    'min_score': min(self.generation_scores[self.current_generation]),
                    'avg_lines': np.mean(self.generation_lines[self.current_generation]),
                    'avg_fitness': np.mean(self.generation_fitness[self.current_generation])
                })
            self.current_generation = metrics.generation
            
        # Update generation dictionaries
        if metrics.generation not in self.generation_scores:
            self.generation_scores[metrics.generation] = []
            self.generation_lines[metrics.generation] = []
            self.generation_fitness[metrics.generation] = []
            
        self.generation_scores[metrics.generation].append(metrics.score)
        self.generation_lines[metrics.generation].append(metrics.lines_cleared)
        self.generation_fitness[metrics.generation].append(metrics.reward)
        
        # Update member stats
        self.member_stats.append({
            'generation': metrics.generation,
            'member': metrics.member,
            'score': metrics.score,
            'lines': metrics.lines_cleared,
            'fitness': metrics.reward
        })
        
        # Update learning metrics
        self.rewards.append(metrics.reward)
        self.epsilon_values.append(metrics.epsilon)
        self.losses.append(metrics.loss)
        self.q_values.append(metrics.q_value)
        
        # Update rolling averages
        self.score_window.append(metrics.score)
        self.lines_window.append(metrics.lines_cleared)
        self.reward_window.append(metrics.reward)
        self.q_value_window.append(metrics.q_value)
        if metrics.loss != 0:
            self.loss_window.append(metrics.loss)
        
        self.moves += 1

    def get_rolling_average(self, data: List[float], window: int = 100) -> np.ndarray:
        """Calculate rolling average with specified window size"""
        if not data:
            return np.array([])
        
        data_array = np.array(data)
        if len(data_array) < window:
            return data_array
            
        ret = np.cumsum(data_array, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        return ret[window - 1:] / window
    def generate_enhanced_plots(self):
        """Generate comprehensive visualization of all metrics"""
        plot_dir = os.path.join(self.session_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Generate all plots
        self._create_overview_dashboard(plot_dir)
        self._create_performance_analysis(plot_dir)
        self._create_learning_dynamics_plot(plot_dir)
        self._create_diversity_analysis(plot_dir)
        
        # Generate HTML report
        self._generate_html_report()

    def _create_overview_dashboard(self, output_dir: str):
        """Create main dashboard with key metrics"""
        plt.figure(figsize=(20, 15))
        
        # Score progress with confidence interval
        plt.subplot(2, 2, 1)
        if len(self.scores) > 0:
            x = np.arange(len(self.scores))
            scores_array = np.array(self.scores)
            window = min(50, len(scores_array))
            if len(scores_array) >= window:
                rolling_mean = np.convolve(scores_array, np.ones(window)/window, mode='valid')
                rolling_std = np.array([np.std(scores_array[max(0, i-window):i]) 
                                    for i in range(window, len(scores_array)+1)])
                plt.plot(x[window-1:], rolling_mean, label='Moving Average')
                plt.fill_between(x[window-1:], 
                               rolling_mean - rolling_std,
                               rolling_mean + rolling_std,
                               alpha=0.3)
            else:
                plt.plot(x, scores_array, label='Scores')
            
            plt.title('Score Progress with Confidence Interval')
            plt.xlabel('Episodes')
            plt.ylabel('Score')
            plt.legend()
        
        # Correlation heatmap
        plt.subplot(2, 2, 2)
        if len(self.scores) > 0:
            metrics_df = pd.DataFrame({
                'Score': self.scores,
                'Lines': self.lines_cleared,
                'Height': self.avg_heights,
                'Holes': self.holes,
                'Deep Holes': self.deep_holes,
                'Wells': self.well_formations,
                'Smoothness': self.surface_smoothness
            })
            sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm')
            plt.title('Metrics Correlation')
        
        # Generation performance
        plt.subplot(2, 2, 3)
        if self.generation_stats:
            gen_df = pd.DataFrame(self.generation_stats)
            plt.plot(gen_df['generation'], gen_df['avg_score'], label='Avg Score')
            plt.plot(gen_df['generation'], gen_df['max_score'], label='Max Score')
            plt.fill_between(gen_df['generation'], 
                           gen_df['min_score'], gen_df['max_score'], 
                           alpha=0.2)
            plt.title('Generation Performance')
            plt.xlabel('Generation')
            plt.ylabel('Score')
            plt.legend()
        
        # Learning progress
        plt.subplot(2, 2, 4)
        if self.rewards:
            plt.plot(self.get_rolling_average(self.rewards), label='Reward')
        plt.title('Learning Progress')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overview_dashboard.png'))
        plt.close()
    
    def _create_performance_analysis(self, output_dir: str):
        """Create detailed performance analysis plots"""
        plt.figure(figsize=(15, 10))
        
        # Performance metrics over generations
        plt.subplot(2, 1, 1)
        if self.generation_stats:
            gen_df = pd.DataFrame(self.generation_stats)
            plt.plot(gen_df['generation'], gen_df['avg_fitness'], label='Avg Fitness')
            plt.plot(gen_df['generation'], gen_df['avg_lines'], label='Avg Lines')
            plt.title('Performance Metrics per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Value')
            plt.legend()
        
        # Member performance distribution
        plt.subplot(2, 1, 2)
        if self.member_stats:
            member_df = pd.DataFrame(self.member_stats)
            sns.boxplot(x='generation', y='fitness', data=member_df)
            plt.title('Member Fitness Distribution per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'))
        plt.close()

    def _create_learning_dynamics_plot(self, output_dir: str):
        """Create plots showing learning dynamics"""
        plt.figure(figsize=(15, 10))
        
        # Genetic algorithm metrics
        plt.subplot(2, 1, 1)
        if self.generation_stats:
            gen_df = pd.DataFrame(self.generation_stats)
            plt.plot(gen_df['generation'], 
                    (gen_df['max_score'] - gen_df['min_score']) / gen_df['avg_score'],
                    label='Score Spread')
            plt.title('Population Score Distribution')
            plt.xlabel('Generation')
            plt.ylabel('Score Spread')
        
        # Board state metrics
        plt.subplot(2, 1, 2)
        if len(self.holes) > 0:
            plt.plot(self.get_rolling_average(self.holes), label='Holes')
            plt.plot(self.get_rolling_average(self.deep_holes), label='Deep Holes')
            plt.plot(self.get_rolling_average(self.surface_smoothness), label='Surface Smoothness')
            plt.title('Board State Metrics')
            plt.xlabel('Episodes')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_dynamics.png'))
        plt.close()

    def _create_diversity_analysis(self, output_dir: str):
        """Create population diversity analysis plots"""
        plt.figure(figsize=(15, 10))
        
        # Population diversity
        plt.subplot(2, 1, 1)
        if self.population_diversity:
            plt.plot(self.population_diversity)
            plt.title('Population Diversity Over Time')
            plt.xlabel('Generation')
            plt.ylabel('Diversity Metric')
        
        # Well formations and surface metrics
        plt.subplot(2, 1, 2)
        if len(self.well_formations) > 0:
            plt.plot(self.get_rolling_average(self.well_formations), label='Wells')
            plt.plot(self.get_rolling_average(self.surface_smoothness), label='Smoothness')
            plt.title('Strategic Metrics')
            plt.xlabel('Episodes')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diversity_analysis.png'))
        plt.close()

    def _generate_html_report(self):
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        html_content = f"""
        <html>
        <head>
            <title>Tetris AI Training Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                h1, h2 {{ color: #333; }}
                img {{ max-width: 100%; height: auto; border-radius: 5px; }}
                .stat-box {{ background-color: #f8f9fa; padding: 10px; margin: 5px; border-radius: 5px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Tetris AI Training Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <div class="section">
                    <h2>Training Overview</h2>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <h3>Performance Metrics</h3>
                            <p>Total Episodes: {len(self.scores)}</p>
                            <p>Best Score: {max(self.scores) if self.scores else 0}</p>
                            <p>Average Score: {np.mean(self.scores) if self.scores else 0:.2f}</p>
                            <p>Generations: {len(self.generation_stats)}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Learning Metrics</h3>
                            <p>Total Training Steps: {self.moves}</p>
                            <p>Average Reward: {np.mean(self.rewards) if self.rewards else 0:.2f}</p>
                            <p>Latest Generation Avg: {self.generation_stats[-1]['avg_score'] if self.generation_stats else 0:.2f}</p>
                        </div>
                    </div>
                    <img src="plots/overview_dashboard.png" alt="Overview Dashboard">
                </div>
                
                <div class="section">
                    <h2>Performance Analysis</h2>
                    <img src="plots/performance_analysis.png" alt="Performance Analysis">
                </div>
                
                <div class="section">
                    <h2>Learning Dynamics</h2>
                    <img src="plots/learning_dynamics.png" alt="Learning Dynamics">
                </div>
                
                <div class="section">
                    <h2>Diversity Analysis</h2>
                    <img src="plots/diversity_analysis.png" alt="Diversity Analysis">
                </div>
            </div>
        </body>
        </html>
        """
        
        report_path = os.path.join(self.session_dir, 'training_report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
        print(f"HTML report generated at {report_path}")

    def save_statistics(self):
        """Save all statistics to JSON file"""
        self.game_duration = (datetime.now() - self.start_time).total_seconds()
        
        stats = {
            'game_stats': {
                'total_moves': self.moves,
                'final_score': self.scores[-1] if self.scores else 0,
                'total_lines_cleared': sum(self.lines_cleared),
                'average_score': np.mean(self.scores) if self.scores else 0,
                'average_lines_per_move': np.mean(self.lines_cleared) if self.lines_cleared else 0,
                'game_duration_seconds': self.game_duration,
                'moves_per_second': self.moves / self.game_duration if self.game_duration > 0 else 0
            },
            'learning_stats': {
                'average_reward': np.mean(self.rewards) if self.rewards else 0,
                'average_q_value': np.mean(self.q_values) if self.q_values else 0,
                'average_loss': np.mean(self.losses) if self.losses else 0,
                'total_training_steps': self.moves
            },
            'performance_metrics': {
                'average_height': np.mean(self.avg_heights) if self.avg_heights else 0,
                'average_holes': np.mean(self.holes) if self.holes else 0,
                'average_deep_holes': np.mean(self.deep_holes) if self.deep_holes else 0,
                'average_wells': np.mean(self.well_formations) if self.well_formations else 0,
                'average_smoothness': np.mean(self.surface_smoothness) if self.surface_smoothness else 0,
                'max_score': max(self.scores) if self.scores else 0,
                'max_lines_cleared_at_once': max(self.lines_cleared) if self.lines_cleared else 0
            },
            'genetic_metrics': {
                'total_generations': len(self.generation_stats),
                'final_generation_stats': self.generation_stats[-1] if self.generation_stats else {},
                'population_diversity': self.population_diversity[-1] if self.population_diversity else 0,
                'best_generation_score': max([g['max_score'] for g in self.generation_stats]) if self.generation_stats else 0
            }
        }
        
        stats_path = os.path.join(self.session_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"Statistics saved to {stats_path}")
        return stats

    def print_training_summary(self):
        """Print comprehensive training summary"""
        print("\nTraining Summary:")
        print("=" * 50)
        print(f"Total Episodes: {len(self.scores)}")
        print(f"Total Generations: {len(self.generation_stats)}")
        print(f"Total Moves: {self.moves}")
        print(f"Final Score: {self.scores[-1] if self.scores else 0}")
        print(f"Best Score: {max(self.scores) if self.scores else 0}")
        print(f"Average Score: {np.mean(self.scores) if self.scores else 0:.2f}")
        if self.generation_stats:
            print(f"Latest Generation Average: {self.generation_stats[-1]['avg_score']:.2f}")
            print(f"Latest Generation Best: {self.generation_stats[-1]['max_score']}")
        print(f"Total Lines Cleared: {sum(self.lines_cleared)}")
        print(f"Average Lines per Episode: {np.mean(self.lines_cleared) if self.lines_cleared else 0:.2f}")
        print(f"Training Duration: {self.game_duration:.2f} seconds")
        print("=" * 50)