import numpy as np

GRID_WIDTH = 10
GRID_HEIGHT = 22

def create_grid():
    return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

def clear_tetromino_from_grid(grid, tetromino, position):
    x, y = position
    for i, row in enumerate(tetromino):
        for j, cell in enumerate(row):
            if cell and 0 <= y + i < GRID_HEIGHT and 0 <= x + j < GRID_WIDTH:
                grid[y + i][x + j] = 0

def place_tetromino_on_grid(grid, tetromino, position):
    x, y = position
    for i, row in enumerate(tetromino):
        for j, cell in enumerate(row):
            if cell and 0 <= y + i < GRID_HEIGHT and 0 <= x + j < GRID_WIDTH:
                grid[y + i][x + j] = cell

def is_valid_position(grid, tetromino, position):
    x, y = position
    for i, row in enumerate(tetromino):
        for j, cell in enumerate(row):
            if cell:
                new_x, new_y = x + j, y + i
                if new_x < 0 or new_x >= GRID_WIDTH or new_y >= GRID_HEIGHT:
                    return False
                if new_y >= 0 and grid[new_y][new_x] != 0:
                    return False
    return True

def check_completed_rows(grid):
    completed_rows = []
    for i in range(GRID_HEIGHT):
        if all(cell != 0 for cell in grid[i]):
            completed_rows.append(i)
    return completed_rows

def clear_rows(grid, completed_rows):
    for row_index in sorted(completed_rows, reverse=True):
        # Remove the completed row
        grid = np.delete(grid, row_index, axis=0)
        # Add a new empty row at the top
        grid = np.vstack([np.zeros(GRID_WIDTH, dtype=int), grid])
    return grid