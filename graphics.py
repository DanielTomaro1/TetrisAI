import pygame
import numpy as np
from collections import deque

# Existing constants...
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 660
WINDOW_WIDTH_EXTENDED = WINDOW_WIDTH + 450  # Extended for metrics panel
CELL_SIZE = 30

TETROMINO_COLORS = {
    1: (0, 240, 240),     # Cyan for I
    2: (240, 240, 0),     # Yellow for O
    3: (160, 0, 240),     # Purple for T
    4: (240, 160, 0),     # Orange for L
    5: (0, 0, 240),       # Blue for J
    6: (0, 240, 0),       # Green for S
    7: (240, 0, 0)        # Red for Z
}

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

class MetricsVisualizer:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.reward_history = deque(maxlen=max_points)
        self.q_value_history = deque(maxlen=max_points)
        self.loss_history = deque(maxlen=max_points)
        self.score_history = deque(maxlen=max_points)
        
        # Initialize with zeros
        for _ in range(max_points):
            self.reward_history.append(0)
            self.q_value_history.append(0)
            self.loss_history.append(0)
            self.score_history.append(0)

    def update(self, reward, q_value, loss, score):
        self.reward_history.append(reward)
        self.q_value_history.append(q_value)
        self.loss_history.append(loss)
        self.score_history.append(score)

    def draw_graph(self, screen, data, position, size, color, title, min_val=None, max_val=None):
        x, y = position
        width, height = size
        
        # Draw border and background
        pygame.draw.rect(screen, DARK_GRAY, (x, y, width, height))
        pygame.draw.rect(screen, BLACK, (x + 1, y + 1, width - 2, height - 2))
        
        # Draw title
        font = pygame.font.Font(None, 24)
        title_surface = font.render(title, True, WHITE)
        screen.blit(title_surface, (x + 5, y + 5))
        
        # Calculate min and max for scaling
        if min_val is None:
            min_val = min(data)
        if max_val is None:
            max_val = max(data) if max(data) != 0 else 1
            
        # Draw graph
        points = []
        for i, value in enumerate(data):
            point_x = x + (i * (width - 20) // self.max_points) + 10
            try:
                point_y = y + height - 20 - ((value - min_val) * (height - 40) // (max_val - min_val))
            except ZeroDivisionError:
                point_y = y + height - 20
            points.append((point_x, point_y))
            
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)

def draw_metrics_panel(screen, metrics_visualizer, epsilon, episode_count, current_stats):
    panel_x = WINDOW_WIDTH + 20
    panel_width = WINDOW_WIDTH_EXTENDED - WINDOW_WIDTH - 40
    
    # Draw episode info
    font = pygame.font.Font(None, 30)
    episode_text = font.render(f"Episode: {episode_count}", True, WHITE)
    epsilon_text = font.render(f"Epsilon: {epsilon:.3f}", True, WHITE)
    screen.blit(episode_text, (panel_x, 20))
    screen.blit(epsilon_text, (panel_x, 50))
    
    # Draw current stats
    stats_y = 90
    stats_font = pygame.font.Font(None, 24)
    stats_text = [
        f"Score: {current_stats.get('score', 0)}",
        f"Lines: {current_stats.get('lines', 0)}",
        f"Holes: {current_stats.get('holes', 0)}",
        f"Height: {current_stats.get('height', 0):.1f}"
    ]
    
    for i, text in enumerate(stats_text):
        text_surface = stats_font.render(text, True, WHITE)
        screen.blit(text_surface, (panel_x, stats_y + i * 25))
    
    # Draw graphs
    graph_height = 100
    graph_width = panel_width
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.reward_history,
        (panel_x, 200),
        (graph_width, graph_height),
        GREEN,
        "Reward"
    )
    
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.q_value_history,
        (panel_x, 320),
        (graph_width, graph_height),
        BLUE,
        "Q-Value"
    )
    
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.loss_history,
        (panel_x, 440),
        (graph_width, graph_height),
        RED,
        "Loss"
    )
    
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.score_history,
        (panel_x, 560),
        (graph_width, graph_height),
        YELLOW,
        "Score"
    )

# Keep existing drawing functions...
def draw_grid(screen, grid):
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, DARK_GRAY, rect, 1)
            if grid[y][x] != 0:
                pygame.draw.rect(screen, TETROMINO_COLORS.get(grid[y][x], WHITE), rect)
                pygame.draw.rect(screen, GRAY, rect, 1)

def draw_next_tetromino(screen, next_tetromino):
    next_x = WINDOW_WIDTH + 20
    next_y = 100
    
    font = pygame.font.Font(None, 36)
    label = font.render("NEXT", True, WHITE)
    screen.blit(label, (next_x, next_y - 30))
    
    for i, row in enumerate(next_tetromino):
        for j, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    next_x + j * CELL_SIZE,
                    next_y + i * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(screen, TETROMINO_COLORS.get(cell, WHITE), rect)
                pygame.draw.rect(screen, GRAY, rect, 1)

def draw_score(screen, score):
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

def draw_game_over(screen):
    font = pygame.font.Font(None, 48)
    text = font.render("Game Over!", True, WHITE)
    text_rect = text.get_rect(center=(WINDOW_WIDTH_EXTENDED//2, WINDOW_HEIGHT//2))
    screen.blit(text, text_rect)