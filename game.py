import pygame
import numpy as np
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
    possible_moves = []
    rotations = TETROMINOES[tetromino_key]
    center_x = grid.shape[1] // 2
    
    for rotation_index in range(len(rotations)):
        current_tetromino = rotations[rotation_index]
        # Prioritize central positions
        for x_offset in range(grid.shape[1]):
            # Try positions from center outward
            for sign in [0, 1, -1]:
                x = center_x + (sign * x_offset)
                if x < 0 or x >= grid.shape[1]:
                    continue
                    
                position = (x, 0)
                if is_valid_position(grid, current_tetromino, position):
                    # Find landing position
                    y = 0
                    while y < grid.shape[0] - 1 and \
                          is_valid_position(grid, current_tetromino, (x, y + 1)):
                        y += 1
                    
                    if is_valid_position(grid, current_tetromino, (x, y)):
                        possible_moves.append((rotation_index, (x, y)))
    
    print(f"Found {len(possible_moves)} possible moves for piece {tetromino_key}")
    return possible_moves

def calculate_reward(grid, lines_cleared, previous_height, current_height, holes):
    """
    Reward function to incentivize good play in Tetris.
    
    Args:
        grid (numpy.ndarray): The current board grid.
        lines_cleared (int): Number of lines cleared in the current move.
        previous_height (int): The height of the highest column before the move.
        current_height (int): The height of the highest column after the move.
        holes (int): The number of holes in the grid after the move.
    
    Returns:
        float: The calculated reward.
    """
    # Base reward for clearing lines
    line_clear_reward = 100 * lines_cleared ** 2  # Quadratic scaling for multiple lines
    
    # Penalize for creating holes
    hole_penalty = -10 * holes
    
    # Penalize for increasing the maximum height
    height_penalty = -5 * (current_height - previous_height)
    if current_height > 20:  # Heavily penalize if stack gets dangerously high
        height_penalty -= 50
    
    # Encourage maintaining low heights
    height_incentive = -2 * current_height  # Lower height means smaller penalty
    
    # Total reward
    reward = line_clear_reward + hole_penalty + height_penalty + height_incentive
    
    return reward

def apply_action(grid, tetromino_key, action, get_holes_per_column):
    """
    Simulates the AI's chosen action and returns the updated grid, lines cleared, and holes.

    Args:
        grid (numpy.ndarray): The current game grid.
        tetromino_key (str): The key of the current tetromino (e.g., "T", "I").
        action (tuple): The AI's chosen action (rotation_index, target_position).
        get_holes_per_column (function): Function to calculate holes per column.

    Returns:
        next_grid (numpy.ndarray): The updated grid after applying the action.
        lines_cleared (int): Number of lines cleared as a result of the action.
        holes (int): Total number of holes in the updated grid.
    """
    rotation_index, target_position = action

    # Get the rotated tetromino
    tetromino = TETROMINOES[tetromino_key][rotation_index]

    # Simulate tetromino placement
    next_grid = grid.copy()
    place_tetromino_on_grid(next_grid, tetromino, target_position)

    # Check for completed rows
    completed_rows = check_completed_rows(next_grid)
    lines_cleared = len(completed_rows) if completed_rows else 0
    if completed_rows:
        next_grid = clear_rows(next_grid, completed_rows)

    # Calculate holes
    holes = get_holes_per_column(next_grid).sum()

    return next_grid, lines_cleared, holes

def game_loop():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH_EXTENDED, WINDOW_HEIGHT))
    pygame.display.set_caption("Tetris Neural AI")
    clock = pygame.time.Clock()
    FPS = 60
    previous_max_height = 0  # Track the previous max height

    # Initialize AI and Metrics
    ai = TetrisAI()
    metrics_visualizer = MetricsVisualizer()
    running = True
    episode = 0
    total_score = 0
    best_score = 0

    while running:
        # Reset game state for new episode
        grid = create_grid()
        tetromino_key, tetromino, rotation_index = spawn_tetromino()
        next_tetromino_key, next_tetromino, _ = spawn_tetromino()
        position = (grid.shape[1] // 2 - 2, 0)
        score = 0
        game_over = False
        episode += 1
        print(f"\nStarting Episode {episode}")

        # Initialize statistics for this episode
        stats = GameStatistics()

        # Animation timing
        move_delay = 100
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
                        ai.save_model(f'tetris_model_episode_{episode}.pth')
                        print(f"Model saved at episode {episode}!")
                    elif event.key == pygame.K_p:
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
                move = ai.choose_action(state, possible_moves)
                if move is None:
                    print("AI couldn't choose a valid move")
                    game_over = True
                    continue

                rotation_index, target_position = move
                tetromino = TETROMINOES[tetromino_key][rotation_index]

                if not is_valid_position(grid, tetromino, target_position):
                    print(f"Warning: Chosen move became invalid: {move}")
                    game_over = True
                    continue

                # Make the move
                position = target_position
                place_tetromino_on_grid(grid, tetromino, position)

                # Simulate the action and get the next grid state
                next_grid, lines_cleared, holes = apply_action(
                    grid, tetromino_key, move, ai._get_holes_per_column
                )

                # Initialize metrics with defaults
                current_max_height = 0
                current_holes = 0
                current_height = 0

                # Calculate metrics
                if next_grid is not None:  # Ensure `next_grid` was computed successfully
                    heights = ai._get_heights(next_grid)
                    current_max_height = np.max(heights) if len(heights) > 0 else 0
                    current_holes = ai._get_holes_per_column(next_grid).sum()
                    current_height = np.mean(heights) if len(heights) > 0 else 0

                # Update game state
                grid = next_grid
                previous_max_height = current_max_height

                # Calculate the reward
                reward = calculate_reward(
                    grid=grid,
                    lines_cleared=lines_cleared,
                    previous_height=previous_max_height,
                    current_height=current_max_height,
                    holes=holes,
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
                    q_value=ai.get_latest_q_value(),
                )

                # Update metrics visualizer
                metrics_visualizer.update(
                    reward=reward,
                    q_value=ai.get_latest_q_value(),
                    loss=ai.get_latest_loss(),
                    score=score,
                )

                # Spawn new piece
                tetromino_key = next_tetromino_key
                tetromino = next_tetromino
                next_tetromino_key, next_tetromino, _ = spawn_tetromino()
                position = (grid.shape[1] // 2 - 2, 0)

                # Check if new piece can be placed
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

            # Draw metrics panel with current stats
            current_stats = {
                "score": score,
                "lines": lines_cleared if "lines_cleared" in locals() else 0,
                "holes": current_holes if "current_holes" in locals() else 0,
                "height": current_height if "current_height" in locals() else 0,
            }

            draw_metrics_panel(
                screen,
                metrics_visualizer,
                ai.epsilon,
                episode,
                current_stats,
            )

            if game_over:
                draw_game_over(screen)

            pygame.display.flip()
            clock.tick(FPS)

        # Episode ended
        if running:
            total_score += score
            best_score = max(best_score, score)

            # Final training update for this episode
            final_reward = calculate_reward(
                grid=grid,
                lines_cleared=0,
                previous_height=previous_max_height,
                current_height=current_height,
                holes=current_holes
            )
            ai.train(state, action_index, final_reward, next_state, True)

            # Generate and save statistics
            stats.generate_plots()
            final_stats = stats.save_statistics()
            print(f"\nEpisode {episode} Statistics:")
            print(f"Score: {score}")
            print(f"Best Score: {best_score}")
            print(f"Average Score: {total_score / episode:.2f}")
            for key, value in final_stats.items():
                print(f"{key}: {value}")

            # Update target network and save model periodically
            ai.update_target_network()
            if episode % 10 == 0:
                ai.save_model(f"tetris_model_episode_{episode}.pth")

            pygame.time.wait(1000)

    # Final cleanup when quitting
    print("\nFinal Training Statistics:")
    print(f"Total Episodes: {episode}")
    print(f"Best Score: {best_score}")
    print(f"Average Score: {total_score / episode:.2f}")

    ai.save_model("tetris_model_final.pth")
    pygame.quit()


if __name__ == "__main__":
    game_loop()
