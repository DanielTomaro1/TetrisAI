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
        self.holes_history = deque(maxlen=max_points)
        
        # Initialize with zeros
        for _ in range(max_points):
            self.score_history.append(0)
            self.fitness_history.append(0)
            self.lines_history.append(0)
            self.height_history.append(0)
            self.holes_history.append(0)

    def update(self, score, fitness, lines, height, holes=0):
        self.score_history.append(score)
        self.fitness_history.append(fitness)
        self.lines_history.append(lines)
        self.height_history.append(height)
        self.holes_history.append(holes)

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
    
    # Draw generation and member info with background
    info_bg = pygame.Rect(panel_x, 10, panel_width, 80)
    pygame.draw.rect(screen, DARK_GRAY, info_bg)
    
    font = pygame.font.Font(None, 30)
    gen_text = font.render(f"Generation: {generation}", True, WHITE)
    member_text = font.render(f"Member: {member}/{population_size}", True, WHITE)
    screen.blit(gen_text, (panel_x + 10, 20))
    screen.blit(member_text, (panel_x + 10, 50))
    
    # Draw current stats with background
    stats_y = 90
    stats_bg = pygame.Rect(panel_x, stats_y, panel_width, 100)
    pygame.draw.rect(screen, DARK_GRAY, stats_bg)
    
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
        screen.blit(text_surface, (panel_x + 10, stats_y + 10 + i * 18))
    
    # Draw graphs
    graph_height = 100
    graph_width = panel_width
    
    # Score graph
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.score_history,
        (panel_x, 200),
        (graph_width, graph_height),
        GREEN,
        "Score History"
    )
    
    # Fitness graph
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.fitness_history,
        (panel_x, 320),
        (graph_width, graph_height),
        BLUE,
        "Fitness History"
    )
    
    # Lines graph
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.lines_history,
        (panel_x, 440),
        (graph_width, graph_height),
        RED,
        "Lines Cleared History"
    )
    
    # Bottom graph background and title
    pygame.draw.rect(screen, DARK_GRAY, (panel_x, 560, panel_width, graph_height))
    pygame.draw.rect(screen, BLACK, (panel_x + 1, 561, panel_width - 2, graph_height - 2))
    
    # Draw title for bottom graph
    title_font = pygame.font.Font(None, 24)
    title = title_font.render("Stack Metrics History", True, WHITE)
    screen.blit(title, (panel_x + 5, 565))
    
    # Draw legend for bottom graph
    legend_y = 585
    legend_font = pygame.font.Font(None, 20)
    
    # Draw the actual graphs in the correct order (holes first, then height)
    # Draw holes graph in red
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.holes_history,
        (panel_x, 560),
        (graph_width, graph_height),
        (255, 50, 50),  # Bright red
        ""
    )
    
    # Draw height graph in yellow (on top)
    metrics_visualizer.draw_graph(
        screen, 
        metrics_visualizer.height_history,
        (panel_x, 560),
        (graph_width, graph_height),
        (255, 255, 0),  # Bright yellow
        ""
    )
    
    # Height legend (Yellow) - drawn second to match graph order
    pygame.draw.line(screen, (255, 255, 0), (panel_x + 10, legend_y), (panel_x + 30, legend_y), 3)
    height_text = legend_font.render("Height", True, (255, 255, 0))
    screen.blit(height_text, (panel_x + 35, legend_y - 7))
    
    # Holes legend (Red) - drawn first to match graph order
    pygame.draw.line(screen, (255, 50, 50), (panel_x + 100, legend_y), (panel_x + 120, legend_y), 3)
    holes_text = legend_font.render("Holes", True, (255, 50, 50))
    screen.blit(holes_text, (panel_x + 125, legend_y - 7))

def draw_grid(screen, grid):
    # Draw background for the grid
    grid_bg = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, DARK_GRAY, grid_bg, 1)
    
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, DARK_GRAY, rect, 1)
            if grid[y][x] != 0:
                pygame.draw.rect(screen, TETROMINO_COLORS.get(grid[y][x], WHITE), rect)
                pygame.draw.rect(screen, GRAY, rect, 1)

def draw_next_tetromino(screen, next_tetromino):
    # Move the next piece preview further right
    next_x = WINDOW_WIDTH + 150
    next_y = 100
    
    # Draw background for next piece
    preview_bg = pygame.Rect(next_x - 10, next_y - 40, CELL_SIZE * 6, CELL_SIZE * 6)
    pygame.draw.rect(screen, DARK_GRAY, preview_bg)
    
    # Draw label
    font = pygame.font.Font(None, 36)
    label = font.render("NEXT PIECE", True, WHITE)
    label_rect = label.get_rect(center=(next_x + CELL_SIZE * 2, next_y - 20))
    screen.blit(label, label_rect)
    
    # Calculate offset to center the piece
    offset_x = (4 - len(next_tetromino[0])) * CELL_SIZE // 2
    offset_y = (4 - len(next_tetromino)) * CELL_SIZE // 2
    
    for i, row in enumerate(next_tetromino):
        for j, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    next_x + j * CELL_SIZE + offset_x,
                    next_y + i * CELL_SIZE + offset_y,
                    CELL_SIZE,
                    CELL_SIZE
                )
                pygame.draw.rect(screen, TETROMINO_COLORS.get(cell, WHITE), rect)
                pygame.draw.rect(screen, GRAY, rect, 1)

def draw_score(screen, score):
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    score_bg = pygame.Rect(5, 5, score_text.get_width() + 10, 40)
    pygame.draw.rect(screen, DARK_GRAY, score_bg)
    screen.blit(score_text, (10, 10))

def draw_game_over(screen):
    font = pygame.font.Font(None, 48)
    text = font.render("Game Over!", True, WHITE)
    text_rect = text.get_rect(center=(WINDOW_WIDTH_EXTENDED//2, WINDOW_HEIGHT//2))
    
    # Draw background for game over text
    bg_rect = text_rect.copy()
    bg_rect.inflate_ip(20, 20)
    pygame.draw.rect(screen, DARK_GRAY, bg_rect)
    screen.blit(text, text_rect)