import pygame
import numpy as np
from collections import deque

# Constants
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 660
WINDOW_WIDTH_EXTENDED = WINDOW_WIDTH + 450
CELL_SIZE = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

TETROMINO_COLORS = {
    1: (0, 240, 240),     # Cyan for I
    2: (240, 240, 0),     # Yellow for O
    3: (160, 0, 240),     # Purple for T
    4: (240, 160, 0),     # Orange for L
    5: (0, 0, 240),       # Blue for J
    6: (0, 240, 0),       # Green for S
    7: (240, 0, 0)        # Red for Z
}

class MetricsVisualizer:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.score_history = deque(maxlen=max_points)
        self.fitness_history = deque(maxlen=max_points)
        self.lines_history = deque(maxlen=max_points)
        self.height_history = deque(maxlen=max_points)
        
        # Initialize with zeros
        for _ in range(max_points):
            self.score_history.append(0)
            self.fitness_history.append(0)
            self.lines_history.append(0)
            self.height_history.append(0)

    def update(self, score, fitness, lines, height):
        self.score_history.append(score)
        self.fitness_history.append(fitness)
        self.lines_history.append(lines)
        self.height_history.append(height)

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

def draw_metrics_panel(screen, metrics_visualizer, generation, member, population_size, current_stats):
    panel_x = WINDOW_WIDTH + 20
    panel_width = WINDOW_WIDTH_EXTENDED - WINDOW_WIDTH - 40
    
    # Draw generation and member info
    font = pygame.font.Font(None, 30)
    gen_text = font.render(f"Generation: {generation}", True, WHITE)
    member_text = font.render(f"Member: {member}/{population_size}", True, WHITE)
    screen.blit(gen_text, (panel_x, 20))
    screen.blit(member_text, (panel_x, 50))
    
    # Draw current stats
    stats_y = 90
    stats_font = pygame.font.Font(None, 24)
    stats_text = [
        f"Score: {current_stats.get('score', 0)}",
        f"Lines: {current_stats.get('lines', 0)}",
        f"Holes: {current_stats.get('holes', 0)}",
        f"Height: {current_stats.get('height', 0):.1f}",
        f"Fitness: {current_stats.get('fitness', 0):.1f}"
    ]
    
    for i, text in enumerate(stats_text):
        text_surface = stats_font.render(text, True, WHITE)
        screen.blit(text_surface, (panel_x, stats_y + i * 25))
    
    # Draw graphs
    graph_height = 100
    graph_width = panel_width
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.score_history,
        (panel_x, 200),
        (graph_width, graph_height),
        GREEN,
        "Score"
    )
    
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.fitness_history,
        (panel_x, 320),
        (graph_width, graph_height),
        BLUE,
        "Fitness"
    )
    
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.lines_history,
        (panel_x, 440),
        (graph_width, graph_height),
        RED,
        "Lines"
    )
    
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.height_history,
        (panel_x, 560),
        (graph_width, graph_height),
        YELLOW,
        "Height"
    )

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