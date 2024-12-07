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
from neural_ai import TetrisAI
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
    """Get possible moves, prioritizing piece connections and gap filling."""
    possible_moves = []
    scored_moves = []  # Will store moves with their connection scores
    rotations = TETROMINOES[tetromino_key]
    
    # Check each rotation
    for rotation_index in range(len(rotations)):
        current_tetromino = rotations[rotation_index]
        tetromino_width = len(current_tetromino[0])
        
        # Try each x position
        for x in range(-2, grid.shape[1] - tetromino_width + 3):
            # Find the lowest valid position for this x
            y = 0
            while y < grid.shape[0] - 1:
                if is_valid_position(grid, current_tetromino, (x, y + 1)):
                    y += 1
                else:
                    break
            
            position = (x, y)
            if is_valid_position(grid, current_tetromino, position):
                # Calculate a score for this move based on connections
                connections = count_connections(grid, position, current_tetromino)
                gaps_filled = count_gaps_filled(grid, position, current_tetromino)
                
                # Calculate move score (higher is better)
                move_score = (
                    connections * 2 +  # Weight for connections
                    gaps_filled * 3    # Weight for filling gaps
                )
                
                # Add to scored moves
                scored_moves.append((move_score, rotation_index, position))
    
    # Sort moves by score (highest first)
    scored_moves.sort(reverse=True)
    
    # Convert to regular moves list
    possible_moves = [(rot_idx, pos) for _, rot_idx, pos in scored_moves]
    
    print(f"Found {len(possible_moves)} possible moves for piece {tetromino_key}")
    # Print top 3 moves and their scores for debugging
    if scored_moves:
        print("Top 3 moves:")
        for score, rot_idx, pos in scored_moves[:3]:
            print(f"Score: {score}, Rotation: {rot_idx}, Position: {pos}")
    
    return possible_moves

def calculate_reward(lines_cleared, holes, height, bumpiness, game_over, grid=None, position=None, tetromino=None):
    """Calculate reward with optional piece fitting evaluation."""
    if game_over:
        return -50
    
    reward = 0
    
    # Line clearing rewards
    if lines_cleared > 0:
        reward += {
            1: 100,    # Single line
            2: 200,   # Double line
            3: 600,   # Triple line
            4: 2400   # Tetris
        }.get(lines_cleared, 0)
    
    # Structural penalties
    reward -= holes * 20       # Penalty for holes
    reward -= height * 1.5     # Smaller penalty for height
    reward -= bumpiness * 2    # Penalty for uneven surface
    
    # If we have piece fitting information, use it
    if grid is not None and position is not None and tetromino is not None:
        connections = count_connections(grid, position, tetromino)
        reward += connections * 8
        
        gaps_filled = count_gaps_filled(grid, position, tetromino)
        reward += gaps_filled * 15
    
    return reward
                
def count_connections(grid, position, tetromino):
    """Count how many sides of the tetromino connect with existing pieces."""
    connections = 0
    x, y = position
    
    # Check each cell of the tetromino
    for i in range(len(tetromino)):
        for j in range(len(tetromino[i])):
            if tetromino[i][j] != 0:
                # Check all adjacent cells
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    new_x, new_y = x + j + dx, y + i + dy
                    
                    # Count connections with existing pieces
                    if (0 <= new_x < grid.shape[1] and 
                        0 <= new_y < grid.shape[0] and 
                        grid[new_y][new_x] != 0):
                        connections += 1
    
    return connections

def count_gaps_filled(grid, position, tetromino):
    """Count how many gaps the tetromino fills."""
    gaps_filled = 0
    x, y = position
    
    # Check each cell of the tetromino
    for i in range(len(tetromino)):
        for j in range(len(tetromino[i])):
            if tetromino[i][j] != 0:
                # Check if this piece fills a gap
                if is_filling_gap(grid, x + j, y + i):
                    gaps_filled += 1
    
    return gaps_filled

def is_filling_gap(grid, x, y):
    """Check if a position represents a gap that should be filled."""
    if y >= grid.shape[0] - 1:
        return False
        
    # Count adjacent blocks
    adjacent_blocks = 0
    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < grid.shape[1] and 
            0 <= new_y < grid.shape[0] and 
            grid[new_y][new_x] != 0):
            adjacent_blocks += 1
    
    # Position is considered a gap if it has multiple adjacent blocks
    return adjacent_blocks >= 2

def game_loop():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH_EXTENDED, WINDOW_HEIGHT))
    pygame.display.set_caption("Tetris Neural AI")
    clock = pygame.time.Clock()
    FPS = 60

    # Initialize AI and load previous model if it exists
    ai = TetrisAI()
    best_score = 0  # Initialize best_score
    try:
        checkpoint = torch.load('tetris_model_final.pth')
        ai.load_model('tetris_model_final.pth')
        # Load previous best score if available
        best_score = checkpoint.get('best_score', 0)
        print(f"Loaded previous model successfully! Previous best score: {best_score}")
    except FileNotFoundError:
        print("No previous model found, starting fresh")
    
    metrics_visualizer = MetricsVisualizer()
    running = True
    total_episodes = ai.episode_count  # Start from last saved episode count

    # Add training history initialization
    training_history = {
        'scores': [],
        'episodes': [],
        'best_scores': []
    }    

    while running:
        # Initialize new episode
        grid = create_grid()
        tetromino_key, tetromino, rotation_index = spawn_tetromino()
        next_tetromino_key, next_tetromino, _ = spawn_tetromino()
        position = (grid.shape[1] // 2 - 2, 0)
        score = 0
        moves_made = 0
        game_over = False
        total_episodes += 1

        print(f"\nStarting Episode {total_episodes}")
        stats = GameStatistics()
        check_board_state(grid, tetromino, position)

        # Animation timing
        move_delay = 100  # milliseconds
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
                        ai.save_model(f'tetris_model_episode_{total_episodes}.pth', best_score)
                        print(f"Model saved at episode {total_episodes}")
                    elif event.key == pygame.K_p:  # Pause AI
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
                    check_board_state(grid, tetromino, position)
                    game_over = True
                    continue

                # Get AI's chosen move
                # In the game loop, update the AI move selection:
                move = ai.choose_action(state, possible_moves, grid, tetromino_key, tetromino)
                if move is None:
                    print("AI couldn't choose a valid move")
                    game_over = True
                    continue
                    
                rotation_index, target_position = move
                tetromino = TETROMINOES[tetromino_key][rotation_index]
                
                # Verify the move is still valid
                if not is_valid_position(grid, tetromino, target_position):
                    print(f"Warning: Chosen move became invalid: {move}")
                    check_board_state(grid, tetromino, target_position)
                    game_over = True
                    continue
                
                # Make the move
                position = target_position
                place_tetromino_on_grid(grid, tetromino, position)
                
                # Process completed rows
                completed_rows = check_completed_rows(grid)
                lines_cleared = len(completed_rows) if completed_rows else 0
                if completed_rows:
                    score += lines_cleared * 100
                    grid = clear_rows(grid, completed_rows)

                # Calculate metrics
                heights = ai._get_heights(grid)
                current_height = heights.mean()
                current_holes = ai._get_holes_per_column(grid).sum()
                current_bumpiness = ai._get_bumpiness(heights).sum()
                
                # Calculate reward
                reward = calculate_reward(
                        lines_cleared, 
                        current_holes,
                        current_height, 
                        current_bumpiness, 
                        False,
                        grid,
                        position,
                        tetromino
                    )                
                # Get new state and train AI
                next_state = ai.get_state(grid, tetromino, position)
                action_index = possible_moves.index((rotation_index, position))
                ai.train(state, action_index, reward, next_state, False)

                # Update statistics and metrics
                stats.update(
                    score=score,
                    lines=lines_cleared,
                    avg_height=current_height,
                    holes=current_holes,
                    reward=reward,
                    epsilon=ai.epsilon,
                    loss=ai.get_latest_loss(),
                    q_value=ai.get_latest_q_value()
                )

                # Update metrics visualizer
                metrics_visualizer.update(
                    reward=reward,
                    q_value=ai.get_latest_q_value(),
                    loss=ai.get_latest_loss(),
                    score=score
                )

                # Spawn new piece
                tetromino_key = next_tetromino_key
                tetromino = next_tetromino
                next_tetromino_key, next_tetromino, _ = spawn_tetromino()
                position = (grid.shape[1] // 2 - 2, 0)
                
                # Check if new piece can be placed
                if not is_valid_position(grid, tetromino, position):
                    print("Game over: Can't place new piece")
                    check_board_state(grid, tetromino, position)
                    game_over = True
                    continue

                last_move_time = current_time

            # Draw everything
            screen.fill(BLACK)
            draw_grid(screen, grid)
            draw_score(screen, score)
            draw_next_tetromino(screen, next_tetromino)
            
            # Draw metrics panel with current stats
            current_stats = {
                'score': score,
                'lines': lines_cleared if 'lines_cleared' in locals() else 0,
                'holes': current_holes if 'current_holes' in locals() else 0,
                'height': current_height if 'current_height' in locals() else 0
            }
            
            draw_metrics_panel(
                screen,
                metrics_visualizer,
                ai.epsilon,
                total_episodes,
                current_stats
            )
            
            if game_over:
                draw_game_over(screen)
            
            pygame.display.flip()
            clock.tick(FPS)

        # Episode ended
        if running:
            # Update best score
            best_score = max(best_score, score)
            
            # Final state processing
            if 'current_height' in locals() and 'current_holes' in locals() and 'current_bumpiness' in locals():
                            final_reward = calculate_reward(
                                0,  # lines_cleared
                                current_holes, 
                                current_height, 
                                current_bumpiness, 
                                True,  # game_over
                                grid,
                                position,
                                tetromino
                            )
            ai.train(state, action_index, final_reward, next_state, True)
            
            # Log episode statistics
            ai.log_episode(score, moves_made)
            
            # Generate and save statistics
            stats.generate_plots()
            final_stats = stats.save_statistics()
            print(f"\nEpisode {total_episodes} Complete:")
            print(f"Score: {score}")
            print(f"Best Score: {best_score}")
            print(f"Moves Made: {moves_made}")
            print("Final Stats:", final_stats)

            # Update target network
            ai.update_target_network()
            
            # Save model periodically
            if total_episodes % 10 == 0:
                ai.save_model(f'tetris_model_episode_{total_episodes}.pth', best_score)

            # Small delay before next episode
            pygame.time.wait(1000)

    # Final cleanup when quitting
    print("\nTraining Summary:")
    print(f"Total Episodes: {total_episodes}")
    print(f"Best Score: {best_score}")
    ai.save_model('tetris_model_final.pth', best_score)
    pygame.quit()

if __name__ == "__main__":
    game_loop()
