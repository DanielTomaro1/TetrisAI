import os
import pygame
import numpy as np
import torch
import time
from grid import create_grid, is_valid_position, clear_tetromino_from_grid, place_tetromino_on_grid, check_completed_rows, clear_rows
from tetromino import spawn_tetromino, TETROMINOES
from graphics import (
    WINDOW_WIDTH_EXTENDED, WINDOW_HEIGHT, BLACK, 
    draw_grid, draw_next_tetromino, draw_score, draw_game_over,
    MetricsVisualizer, draw_metrics_panel
)
from neural_ai import GeneticTetrisAI, NetworkConfig, GeneticConfig, LearningConfig
from game_statistics import GameStatistics, TrainingMetrics

def check_board_state(grid, tetromino, position):
    """Debug function to check the state of the board"""
    print("\nBoard State Check:")
    print(f"Grid Shape: {grid.shape}")
    print(f"Current Position: {position}")
    print(f"Is Position Valid: {is_valid_position(grid, tetromino, position)}")
    print(f"Piece Matrix:")
    for row in tetromino:
        print(row)

def get_possible_moves(grid, tetromino_key, tetromino):
    possible_moves = []
    rotations = TETROMINOES[tetromino_key]
    
    for rotation_index in range(len(rotations)):
        current_tetromino = rotations[rotation_index]
        tetromino_width = len(current_tetromino[0])
        
        for x in range(-2, grid.shape[1] - tetromino_width + 3):
            y = 0
            while y < grid.shape[0] - 1:
                if is_valid_position(grid, current_tetromino, (x, y + 1)):
                    y += 1
                else:
                    break
            
            position = (x, y)
            if is_valid_position(grid, current_tetromino, position):
                possible_moves.append((rotation_index, position))
    
    print(f"Found {len(possible_moves)} possible moves for piece {tetromino_key}")
    return possible_moves

def game_loop():
    """Main game loop with improved state management and debugging"""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH_EXTENDED, WINDOW_HEIGHT))
    pygame.display.set_caption("Tetris Genetic AI")
    clock = pygame.time.Clock()
    FPS = 60

    # Initialize configurations
    network_config = NetworkConfig()
    genetic_config = GeneticConfig()
    learning_config = LearningConfig()

    # Initialize Genetic AI
    ai = GeneticTetrisAI(
        network_config=network_config,
        genetic_config=genetic_config,
        learning_config=learning_config
    )
    
    # Try to load previous state
    try:
        ai.load_state('tetris_genetic_state.pth')
        print(f"Loaded previous state - Generation: {ai.generation}, Member: {ai.current_member}")
    except FileNotFoundError:
        print("Starting fresh training session")
        ai.save_state('tetris_genetic_state.pth')  # Create initial save
    
    # Initialize statistics and visualization
    metrics_visualizer = MetricsVisualizer()
    stats = GameStatistics()
    
    running = True
    session_start_time = time.time()
    last_save_time = session_start_time

    print(f"\nStarting training session:")
    print(f"Generation: {ai.generation}")
    print(f"Current Member: {ai.current_member + 1}/{ai.population_size}")

    while running:
        # Initialize game state for current member
        game_state = {
            'grid': create_grid(),
            'score': 0,
            'moves': 0,
            'lines_cleared': 0,
            'max_height': 0,
            'holes': 0,
            'bumpiness': 0,
            'last_move_time': pygame.time.get_ticks(),
            'piece_stats': {
                'total_pieces': 0,
                'pieces_by_type': {}
            }
        }

        # Spawn initial pieces
        current_piece = {
            'key': None,
            'tetromino': None,
            'rotation': 0,
            'position': (0, 0)
        }
        
        next_piece = {
            'key': None,
            'tetromino': None,
            'rotation': 0
        }

        # Initial piece spawn
        next_piece['key'], next_piece['tetromino'], next_piece['rotation'] = spawn_tetromino()
        current_piece['key'], current_piece['tetromino'], current_piece['rotation'] = spawn_tetromino()
        current_piece['position'] = (game_state['grid'].shape[1] // 2 - 2, 0)

        game_active = True
        move_delay = 50  # Milliseconds between moves

        while game_active and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_s:
                        ai.save_state('tetris_genetic_state.pth')
                        print("Manual state save triggered")
                    elif event.key == pygame.K_p:
                        paused = True
                        while paused and running:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif pause_event.type == pygame.KEYDOWN and pause_event.key == pygame.K_p:
                                    paused = False
                            pygame.display.flip()
                            clock.tick(FPS)

            current_time = pygame.time.get_ticks()

            # AI move timing
            if current_time - game_state['last_move_time'] > move_delay:
                game_state['moves'] += 1
                
                # Clear current piece for movement
                clear_tetromino_from_grid(game_state['grid'], current_piece['tetromino'], 
                                        current_piece['position'])
                
                # Get AI move
                state = ai.get_state(game_state['grid'], current_piece['tetromino'], 
                                   current_piece['position'])
                possible_moves = get_possible_moves(game_state['grid'], current_piece['key'], 
                                                  current_piece['tetromino'])

                if not possible_moves:
                    print(f"Game Over - No valid moves available")
                    game_active = False
                    continue

                # Get AI's chosen move
                move = ai.choose_action(state, possible_moves, game_state['grid'],
                                      current_piece['key'], current_piece['tetromino'])
                
                if move is None:
                    print(f"Game Over - AI couldn't choose valid move")
                    game_active = False
                    continue

                # Apply move
                rotation_index, new_position = move
                current_piece['tetromino'] = TETROMINOES[current_piece['key']][rotation_index]
                current_piece['position'] = new_position

                if not is_valid_position(game_state['grid'], current_piece['tetromino'], 
                                      current_piece['position']):
                    print(f"Game Over - Invalid final position")
                    game_active = False
                    continue

                # Place piece and process results
                place_tetromino_on_grid(game_state['grid'], current_piece['tetromino'], 
                                      current_piece['position'])
                
                # Check for completed rows
                completed_rows = check_completed_rows(game_state['grid'])
                if completed_rows:
                    lines_cleared = len(completed_rows)
                    game_state['lines_cleared'] += lines_cleared
                    game_state['score'] += lines_cleared * 100
                    game_state['grid'] = clear_rows(game_state['grid'], completed_rows)

                # Update piece statistics
                game_state['piece_stats']['total_pieces'] += 1
                game_state['piece_stats']['pieces_by_type'][current_piece['key']] = \
                    game_state['piece_stats']['pieces_by_type'].get(current_piece['key'], 0) + 1

                # Calculate current metrics
                heights = [0] * game_state['grid'].shape[1]
                holes = 0
                for x in range(game_state['grid'].shape[1]):
                    found_block = False
                    for y in range(game_state['grid'].shape[0]):
                        if game_state['grid'][y][x] != 0:
                            if not found_block:
                                heights[x] = game_state['grid'].shape[0] - y
                                found_block = True
                        elif found_block:
                            holes += 1

                game_state['max_height'] = max(heights)
                game_state['holes'] = holes
                game_state['bumpiness'] = sum(abs(heights[i] - heights[i+1]) 
                                            for i in range(len(heights)-1))

                # Prepare next piece
                current_piece['key'] = next_piece['key']
                current_piece['tetromino'] = next_piece['tetromino']
                current_piece['position'] = (game_state['grid'].shape[1] // 2 - 2, 0)
                next_piece['key'], next_piece['tetromino'], next_piece['rotation'] = spawn_tetromino()

                if not is_valid_position(game_state['grid'], current_piece['tetromino'], 
                                      current_piece['position']):
                    print(f"Game Over - Can't place new piece")
                    game_active = False
                    continue

                game_state['last_move_time'] = current_time

                # Update visualizer
                metrics_visualizer.update(
                    score=game_state['score'],
                    fitness=0,  # Will be calculated at game end
                    lines=game_state['lines_cleared'],
                    height=game_state['max_height'],
                    holes=game_state['holes']
                )

            # Draw game state
            screen.fill(BLACK)
            draw_grid(screen, game_state['grid'])
            draw_next_tetromino(screen, next_piece['tetromino'])
            draw_score(screen, game_state['score'])
            
            # Draw metrics panel
            draw_metrics_panel(
                screen=screen,
                metrics_visualizer=metrics_visualizer,
                generation=ai.generation,
                member=ai.current_member + 1,
                population_size=ai.population_size,
                current_stats={
                    'score': game_state['score'],
                    'lines': game_state['lines_cleared'],
                    'holes': game_state['holes'],
                    'height': game_state['max_height'],
                    'fitness': 0
                }
            )

            if not game_active:
                draw_game_over(screen)

            pygame.display.flip()
            clock.tick(FPS)

        # Game over - Process results
        if not game_active:
            # Update AI
            fitness = ai.add_fitness_score(
                score=game_state['score'],
                moves=game_state['moves'],
                lines_cleared=game_state['lines_cleared'],
                max_height=game_state['max_height'],
                holes=game_state['holes'],
                bumpiness=game_state['bumpiness']
            )

            # Update statistics
            metrics = TrainingMetrics(
                episode=game_state['moves'],
                score=game_state['score'],
                lines_cleared=game_state['lines_cleared'],
                avg_height=game_state['max_height'],
                holes=game_state['holes'],
                reward=fitness
            )
            stats.update(metrics)

            # Periodic saves and reports
            current_time = time.time()
            if current_time - last_save_time > 300:  # Save every 5 minutes
                stats.generate_enhanced_plots()
                stats.save_statistics()
                ai.save_state('tetris_genetic_state.pth')
                last_save_time = current_time

            pygame.time.wait(500)  # Brief pause between games

    # Final cleanup
    stats.generate_enhanced_plots()
    stats.save_statistics()
    stats.print_training_summary()
    ai.save_state('tetris_genetic_state.pth')
    pygame.quit()

if __name__ == "__main__":
    game_loop()