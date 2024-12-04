import numpy as np
from grid import is_valid_position, place_tetromino_on_grid, clear_rows, check_completed_rows
from tetromino import TETROMINOES

def get_aggregate_height(grid):
    heights = []
    for x in range(grid.shape[1]):
        col = grid[:, x]
        for y in range(len(col)):
            if col[y] != 0:
                heights.append(grid.shape[0] - y)
                break
        if len(heights) < x + 1:
            heights.append(0)
    return sum(heights)

def get_holes(grid):
    holes = 0
    for x in range(grid.shape[1]):
        col = grid[:, x]
        block_found = False
        for y in range(len(col)):
            if col[y] != 0:
                block_found = True
            elif block_found and col[y] == 0:
                holes += 1
    return holes

def get_bumpiness(grid):
    heights = []
    for x in range(grid.shape[1]):
        col = grid[:, x]
        for y in range(len(col)):
            if col[y] != 0:
                heights.append(grid.shape[0] - y)
                break
        if len(heights) < x + 1:
            heights.append(0)
    
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness

def calculate_score(grid):
    aggregate_height = get_aggregate_height(grid)
    completed_lines = len(check_completed_rows(grid))
    holes = get_holes(grid)
    bumpiness = get_bumpiness(grid)
    
    # Weights found through genetic algorithm
    return (-0.510066 * aggregate_height +
            0.760666 * completed_lines +
            -0.35663 * holes +
            -0.184483 * bumpiness)

def simulate_move(grid, tetromino, position):
    grid_copy = grid.copy()
    place_tetromino_on_grid(grid_copy, tetromino, position)
    completed_rows = check_completed_rows(grid_copy)
    if completed_rows:
        grid_copy = clear_rows(grid_copy, completed_rows)
    return grid_copy

def get_possible_moves(grid, tetromino_key):
    possible_moves = []
    rotations = TETROMINOES[tetromino_key]
    
    for rotation_index in range(len(rotations)):
        tetromino = rotations[rotation_index]
        for x in range(-2, grid.shape[1] + 2):
            position = (x, 0)
            if not is_valid_position(grid, tetromino, position):
                continue
                
            # Drop the tetromino
            while is_valid_position(grid, tetromino, (position[0], position[1] + 1)):
                position = (position[0], position[1] + 1)
            
            possible_moves.append((rotation_index, position))
    
    return possible_moves

def choose_best_move(grid, tetromino_key, next_tetromino_key):
    best_score = float('-inf')
    best_move = None
    
    for rotation_index, position in get_possible_moves(grid, tetromino_key):
        tetromino = TETROMINOES[tetromino_key][rotation_index]
        grid_after_move = simulate_move(grid, tetromino, position)
        
        # Look ahead one piece
        future_score = float('-inf')
        for next_rotation_index, next_position in get_possible_moves(grid_after_move, next_tetromino_key):
            next_tetromino = TETROMINOES[next_tetromino_key][next_rotation_index]
            future_grid = simulate_move(grid_after_move, next_tetromino, next_position)
            score = calculate_score(future_grid)
            future_score = max(future_score, score)
        
        if future_score > best_score:
            best_score = future_score
            best_move = (rotation_index, position)
    
    return best_move

def calculate_score_with_details(grid):
    # Get all heuristics
    aggregate_height = get_aggregate_height(grid)
    completed_lines = len(check_completed_rows(grid))
    holes = get_holes(grid)
    bumpiness = get_bumpiness(grid)
    
    # Weights (unchanged)
    weights = [-0.510066, 0.760666, -0.35663, -0.184483]
    
    # Calculate total score
    score = (weights[0] * aggregate_height +
            weights[1] * completed_lines +
            weights[2] * holes +
            weights[3] * bumpiness)
    
    # Return both the score and the components
    heuristics = [aggregate_height, completed_lines, holes, bumpiness]
    return score, heuristics, weights