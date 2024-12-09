import pygame
import numpy as np
import torch
from grid import create_grid, is_valid_position, clear_tetromino_from_grid, place_tetromino_on_grid, check_completed_rows, clear_rows
from tetromino import spawn_tetromino, TETROMINOES
from graphics import (
    WINDOW_WIDTH_EXTENDED, WINDOW_HEIGHT, BLACK, 
    draw_grid, draw_next_tetromino, draw_score, draw_game_over,
    MetricsVisualizer, draw_metrics_panel
)
from neural_ai import GeneticTetrisAI
from statistics import GameStatistics

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
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH_EXTENDED, WINDOW_HEIGHT))
    pygame.display.set_caption("Tetris Genetic AI")
    clock = pygame.time.Clock()
    FPS = 60

    # Initialize Genetic AI
    ai = GeneticTetrisAI(population_size=30)
    try:
        ai.load_state('tetris_genetic_state.pth')
        print("Loaded previous genetic state")
    except FileNotFoundError:
        print("Starting fresh genetic training")
    
    metrics_visualizer = MetricsVisualizer()
    running = True
    generation = ai.generation

    while running:
        # Initialize new game for current member of population
        grid = create_grid()
        tetromino_key, tetromino, rotation_index = spawn_tetromino()
        next_tetromino_key, next_tetromino, _ = spawn_tetromino()
        position = (grid.shape[1] // 2 - 2, 0)
        score = 0
        moves_made = 0
        total_lines_cleared = 0
        fitness = 0
        game_over = False

        print(f"\nGeneration {generation}, Member {ai.current_member + 1}/{ai.population_size}")
        stats = GameStatistics()
        check_board_state(grid, tetromino, position)

        # Animation timing
        move_delay = 50  # Faster for genetic algorithm
        last_move_time = pygame.time.get_ticks()

        while not game_over and running:
            current_time = pygame.time.get_ticks()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_s:
                        ai.save_state('tetris_genetic_state.pth')
                        print("Genetic state saved")
                    elif event.key == pygame.K_p:  # Pause
                        paused = True
                        while paused and running:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif pause_event.type == pygame.KEYDOWN:
                                    if pause_event.key == pygame.K_p:
                                        paused = False
                            clock.tick(FPS)

            # AI move timing
            if current_time - last_move_time > move_delay:
                moves_made += 1
                
                # Clear current tetromino from grid
                clear_tetromino_from_grid(grid, tetromino, position)
                
                # Get current state and possible moves
                state = ai.get_state(grid, tetromino, position)
                possible_moves = get_possible_moves(grid, tetromino_key, tetromino)
                
                if not possible_moves:
                    print(f"No valid moves found for piece {tetromino_key}")
                    game_over = True
                    continue

                # Get AI's chosen move
                move = ai.choose_action(state, possible_moves, grid, tetromino_key, tetromino)
                if move is None:
                    print("AI couldn't choose a valid move")
                    game_over = True
                    continue
                    
                rotation_index, target_position = move
                tetromino = TETROMINOES[tetromino_key][rotation_index]
                
                # Verify and make the move
                if not is_valid_position(grid, tetromino, target_position):
                    print(f"Invalid move: {move}")
                    game_over = True
                    continue
                
                position = target_position
                place_tetromino_on_grid(grid, tetromino, position)
                
                # Process completed rows
                completed_rows = check_completed_rows(grid)
                lines_cleared = len(completed_rows) if completed_rows else 0
                total_lines_cleared += lines_cleared
                if completed_rows:
                    score += lines_cleared * 100
                    grid = clear_rows(grid, completed_rows)

                # Calculate metrics
                heights = []
                for x in range(grid.shape[1]):
                    col = grid[:, x]
                    for y in range(len(col)):
                        if col[y] != 0:
                            heights.append(grid.shape[0] - y)
                            break
                    if len(heights) < x + 1:
                        heights.append(0)
                
                current_height = max(heights)
                
                # Count holes
                current_holes = 0
                for x in range(grid.shape[1]):
                    col = grid[:, x]
                    block_found = False
                    for y in range(len(col)):
                        if col[y] != 0:
                            block_found = True
                        elif block_found and col[y] == 0:
                            current_holes += 1

                # Calculate fitness
                fitness = (
                    score * 1.0 +           # Base score
                    total_lines_cleared * 100 +    # Lines cleared
                    moves_made * 0.5 -      # Survival bonus
                    current_holes * 20 -    # Holes penalty
                    current_height * 10     # Height penalty
                )

                # Update metrics visualizer
                metrics_visualizer.update(
                    score=score,
                    fitness=fitness,
                    lines=total_lines_cleared,
                    height=current_height
                )

                # Spawn new piece
                tetromino_key = next_tetromino_key
                tetromino = next_tetromino
                next_tetromino_key, next_tetromino, _ = spawn_tetromino()
                position = (grid.shape[1] // 2 - 2, 0)
                
                if not is_valid_position(grid, tetromino, position):
                    print("Game over: Can't place new piece")
                    game_over = True
                    continue

                last_move_time = current_time

            # Draw everything
            screen.fill(BLACK)
            draw_grid(screen, grid)
            draw_score(screen, score)
            draw_next_tetromino(screen, next_tetromino)
            
            # Draw metrics panel
            current_stats = {
                'score': score,
                'lines': total_lines_cleared,
                'holes': current_holes if 'current_holes' in locals() else 0,
                'height': current_height if 'current_height' in locals() else 0,
                'fitness': fitness
            }
            
            draw_metrics_panel(
                screen=screen,
                metrics_visualizer=metrics_visualizer,
                generation=generation,
                member=ai.current_member + 1,
                population_size=ai.population_size,
                current_stats=current_stats
            )
            
            if game_over:
                draw_game_over(screen)
            
            pygame.display.flip()
            clock.tick(FPS)

        # Game over for current member
        if running:
            # Calculate final fitness for this member
            fitness = ai.add_fitness_score(
                score=score,
                moves=moves_made,
                lines_cleared=total_lines_cleared,
                max_height=current_height if 'current_height' in locals() else 22
            )
            
            print(f"Member {ai.current_member}/{ai.population_size} finished:")
            print(f"Score: {score}, Moves: {moves_made}, Lines: {total_lines_cleared}")
            print(f"Fitness: {fitness}")
            
            # Generate plots for this generation if it's complete
            if ai.current_member == 0:  # Just evolved
                generation = ai.generation
                stats.generate_plots()
                
                print(f"\nGeneration {generation} Statistics:")
                print(f"Best Fitness: {max(ai.generation_stats['best_fitness'])}")
                print(f"Average Fitness: {ai.generation_stats['avg_fitness'][-1]}")
                
            pygame.time.wait(100)  # Short delay between members

    # Final cleanup
    print("\nTraining Summary:")
    print(f"Total Generations: {generation}")
    print(f"Best Ever Fitness: {ai.best_fitness}")
    ai.save_state('tetris_genetic_state.pth')
    pygame.quit()

if __name__ == "__main__":
    game_loop()