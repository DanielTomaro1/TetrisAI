import os
import torch
import torch.nn as nn
import numpy as np
import random
import copy
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from tetromino import TETROMINOES

@dataclass
class NetworkConfig:
    input_size: int = 54
    hidden_layers: List[int] = (512, 256, 128)  # Deeper network
    use_batch_norm: bool = True
    dropout_rate: float = 0.1          # Reduced dropout
    activation: str = 'leaky_relu'     # Changed activation function

@dataclass
class LearningConfig:
    # Simplified reward structure focusing on immediate board state
    score_weight: float = 10.0         
    lines_weight: float = 100.0        
    moves_weight: float = 0.0          # Removed survival reward
    
    # Board state penalties
    height_penalty: float = 2.0        # Per-column height penalty
    holes_penalty: float = 5.0         # Per-hole penalty
    bumpiness_penalty: float = 1.0     # Reduced bumpiness impact
    
    # Added new rewards
    tetris_bonus: float = 800.0        # Big bonus for 4-line clear
    well_bonus: float = 2.0            # Reward for maintaining a well
    surface_bonus: float = 1.0         # Reward for flat surface
    
    # Move selection weights
    network_weight: float = 1.0        # Only use network evaluation
    connection_weight: float = 0.0     # Removed heuristic influence
    gap_fill_weight: float = 0.0       # Removed heuristic influence

@dataclass
class GeneticConfig:
    population_size: int = 100         # Much larger population
    mutation_rate: float = 0.01        # Very rare mutations
    mutation_strength: float = 0.01    # Very small mutations
    elite_size: int = 15               # Keep more top performers
    tournament_size: int = 10          # Larger tournaments
    crossover_rate: float = 0.95       # Almost always crossover
    
class TetrisNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super(TetrisNet, self).__init__()
        self.config = config
        
        # Build layers dynamically based on config
        layers = []
        batch_norms = []
        
        prev_size = config.input_size
        for size in config.hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            if config.use_batch_norm:
                batch_norms.append(nn.BatchNorm1d(size))
            prev_size = size
        
        self.layers = nn.ModuleList(layers)
        self.batch_norms = nn.ModuleList(batch_norms)
        
        # Set activation function
        if config.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif config.activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif config.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with verified randomization"""
        print("\nInitializing network weights:")
        for i, layer in enumerate(self.layers):
            # Use normal distribution instead of Xavier
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.zeros_(layer.bias)
            
            # Verify randomization
            weight_stats = {
                'mean': layer.weight.mean().item(),
                'std': layer.weight.std().item(),
                'min': layer.weight.min().item(),
                'max': layer.weight.max().item()
            }
            print(f"Layer {i} weight stats: {weight_stats}")

    def forward(self, x):
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if self.config.use_batch_norm and x.size(0) > 1:
                x = batch_norm(x)
            x = self.activation(x)
            if i < len(self.layers) - 1:  # No dropout on last layer
                x = self.dropout(x)
        return x

    def clone(self) -> 'TetrisNet':
        """Create a deep copy of the network"""
        cloned = TetrisNet(self.config)
        cloned.load_state_dict(self.state_dict())
        return cloned
class GeneticTetrisAI:
    def __init__(self, network_config: Optional[NetworkConfig] = None,
                 genetic_config: Optional[GeneticConfig] = None,
                 learning_config: Optional[LearningConfig] = None):
        
        # Initialize or load configurations
        self.network_config = network_config or NetworkConfig()
        self.genetic_config = genetic_config or GeneticConfig()
        self.learning_config = learning_config or LearningConfig()
        
        # Setup directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('configs', exist_ok=True)
        
        # Save initial configs
        self._save_configs()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize state variables
        self.generation = 0
        self.current_member = 0
        self.population_size = self.genetic_config.population_size
        self.fitness_scores: List[float] = []
        self.best_fitness = float('-inf')
        self.best_network: Optional[TetrisNet] = None
        self.last_save_generation = 0
        self.current_grid = None  # Track current grid state
        
        # Initialize population
        self.population = [TetrisNet(self.network_config).to(self.device) 
                          for _ in range(self.population_size)]
        
        # Initialize statistics tracking
        self.generation_stats = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'population_diversity': [],
            'mutation_effects': [],
            'generation_best_network': None
        }
        
        self._init_state_tracking()
        
        print(f"Genetic AI initialized with population size: {self.population_size}")
        print(f"Network parameters: {sum(p.numel() for p in self.population[0].parameters())}")

    def _init_state_tracking(self):
        """Initialize state tracking variables"""
        self.state_history = {
            'generations': [],
            'member_progress': [],
            'fitness_history': [],
            'evolution_points': []
        }
        
    def _save_configs(self):
        """Save configurations to YAML files"""
        configs = {
            'network_config': self.network_config,
            'genetic_config': self.genetic_config,
            'learning_config': self.learning_config
        }
        
        for name, config in configs.items():
            filepath = os.path.join('configs', f'{name}.yaml')
            with open(filepath, 'w') as f:
                yaml.dump(config.__dict__, f, default_flow_style=False)

    def evaluate_board_state(self, grid, lines_cleared, move_made):
        """Evaluate board state after a move"""
        # Calculate column heights
        heights = self._get_heights(grid)
        max_height = max(heights)
        avg_height = sum(heights) / len(heights)
        
        # Count holes with weighted depth penalty
        holes = 0
        deep_hole_penalty = 0
        for x in range(grid.shape[1]):
            found_block = False
            hole_depth = 0
            for y in range(grid.shape[0]):
                if grid[y][x] != 0:
                    found_block = True
                elif found_block:
                    holes += 1
                    hole_depth += 1
                    deep_hole_penalty += hole_depth * 0.5  # Deeper holes are worse
        
        # Check for well formation (good for Tetris)
        well_count = 0
        for x in range(1, grid.shape[1]-1):
            if heights[x] < heights[x-1] - 2 and heights[x] < heights[x+1] - 2:
                well_count += 1
        
        # Calculate surface smoothness (ignoring wells)
        bumpiness = 0
        surface_breaks = 0
        for i in range(len(heights)-1):
            diff = abs(heights[i] - heights[i+1])
            bumpiness += diff
            if diff > 1:
                surface_breaks += 1
        
        # Calculate immediate rewards
        line_reward = lines_cleared * self.learning_config.lines_weight
        if lines_cleared == 4:
            line_reward *= 2  # Double reward for Tetris
        
        # Calculate penalties
        height_penalty = max_height * self.learning_config.height_penalty
        holes_penalty = (holes + deep_hole_penalty) * self.learning_config.holes_penalty
        surface_penalty = (bumpiness + surface_breaks) * self.learning_config.bumpiness_penalty
        
        # Calculate bonuses
        well_bonus = well_count * self.learning_config.well_bonus
        surface_bonus = (10 - surface_breaks) * self.learning_config.surface_bonus
        
        # Combined reward with debug output
        reward = (line_reward + well_bonus + surface_bonus - 
                height_penalty - holes_penalty - surface_penalty)
        
        if move_made:  # Only print debug when actually making a move
            print(f"\nBoard State Evaluation:")
            print(f"Heights - Max: {max_height}, Avg: {avg_height:.1f}")
            print(f"Holes: {holes} (Depth Penalty: {deep_hole_penalty:.1f})")
            print(f"Surface - Bumpiness: {bumpiness}, Breaks: {surface_breaks}")
            print(f"Well formation: {well_count}")
            print(f"Final reward: {reward:.2f}")
        
        return reward

    def get_state(self, grid, tetromino, position) -> torch.Tensor:
        """Extract state features for the neural network"""
        heights = self._get_heights(grid)
        holes = self._get_holes_per_column(grid)
        lines = self._get_lines_per_row(grid)
        bumpiness = self._get_bumpiness(heights)
        
        # Calculate aggregate metrics
        max_height = np.max(heights)
        avg_height = np.mean(heights)
        total_holes = np.sum(holes)
        
        # Normalize features
        heights = heights / grid.shape[0]
        holes = holes / max(1, max_height)
        lines = lines / grid.shape[1]
        bumpiness = bumpiness / max(1, max_height)
        
        # Combine features
        state_features = np.concatenate([
            heights,           # 10 values
            holes,            # 10 values
            lines,            # 22 values
            bumpiness,        # 9 values
            [max_height / grid.shape[0],
             avg_height / grid.shape[0],
             total_holes / (grid.shape[0] * grid.shape[1])]
        ])
        
        return torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
    def _get_heights(self, grid) -> np.ndarray:
        """Calculate height of each column"""
        heights = []
        for x in range(grid.shape[1]):
            col = grid[:, x]
            for y in range(len(col)):
                if col[y] != 0:
                    heights.append(grid.shape[0] - y)
                    break
            if len(heights) < x + 1:
                heights.append(0)
        return np.array(heights)

    def _get_holes_per_column(self, grid) -> np.ndarray:
        """Count holes in each column"""
        holes = []
        for x in range(grid.shape[1]):
            col = grid[:, x]
            col_holes = 0
            block_found = False
            for y in range(len(col)):
                if col[y] != 0:
                    block_found = True
                elif block_found and col[y] == 0:
                    col_holes += 1
            holes.append(col_holes)
        return np.array(holes)

    def _get_lines_per_row(self, grid) -> np.ndarray:
        """Calculate fill percentage of each row"""
        lines = np.zeros(grid.shape[0])
        for y in range(grid.shape[0]):
            lines[y] = sum(1 for x in grid[y] if x != 0) / grid.shape[1]
        return lines

    def _get_bumpiness(self, heights) -> np.ndarray:
        """Calculate height differences between adjacent columns"""
        return np.array([abs(heights[i] - heights[i+1]) 
                        for i in range(len(heights)-1)])

    def choose_action(self, state: torch.Tensor, possible_moves: List[Tuple], 
                    grid: np.ndarray, tetromino_key: str, tetromino: np.ndarray) -> Optional[Tuple]:
        """Choose the best action with balanced network and heuristic influence"""
        if not possible_moves:
            return None
            
        current_network = self.population[self.current_member]
        print(f"\nEvaluating {len(possible_moves)} possible moves...")
        
        with torch.no_grad():
            network_outputs = current_network(state)
            
            # Evaluate each possible move
            move_scores = []
            for i, (rotation_index, position) in enumerate(possible_moves):
                current_tetromino = TETROMINOES[tetromino_key][rotation_index]
                
                # Get raw network score (bounded by size of output)
                network_score = network_outputs[0][i].item() if i < network_outputs.size(1) else 0
                
                # Simulate move
                temp_grid = grid.copy()
                valid_placement = True
                for ti, row in enumerate(current_tetromino):
                    for tj, cell in enumerate(row):
                        if cell:
                            new_y, new_x = position[1] + ti, position[0] + tj
                            if 0 <= new_y < grid.shape[0] and 0 <= new_x < grid.shape[1]:
                                temp_grid[new_y][new_x] = cell
                            else:
                                valid_placement = False
                                break
                
                if not valid_placement:
                    continue
                
                # Get heuristic score
                heuristic_score = self.evaluate_board_state(temp_grid, 0, True)
                
                # Early in training, use more randomization
                if self.generation < 5:
                    network_weight = 0.2  # Reduce network influence initially
                    random_weight = 0.3   # Add randomization
                    heuristic_weight = 0.5
                    total_score = (
                        network_score * network_weight +
                        heuristic_score * heuristic_weight +
                        random.random() * random_weight
                    )
                else:
                    # Gradually increase network influence
                    network_weight = min(0.8, 0.2 + (self.generation / 20))
                    heuristic_weight = 1.0 - network_weight
                    total_score = (
                        network_score * network_weight +
                        heuristic_score * heuristic_weight
                    )
                
                move_scores.append((total_score, rotation_index, position))
                
                if i < 3:  # Print first few moves for debugging
                    print(f"Move {i}: rot={rotation_index}, pos={position}")
                    print(f"  Network score: {network_score:.2f}")
                    print(f"  Heuristic score: {heuristic_score:.2f}")
                    print(f"  Total score: {total_score:.2f}")
            
            if not move_scores:
                return None
                
            # Choose best move
            best_score, rotation_index, position = max(move_scores)
            print(f"\nChosen move: rot={rotation_index}, pos={position}, score={best_score:.2f}")
            return (rotation_index, position)
        
    def add_fitness_score(self, score: int, moves: int, lines_cleared: int, 
                         max_height: float, holes: int, bumpiness: float) -> float:
        """Calculate and add fitness score when a game is complete"""
        # Get the current grid state evaluation
        fitness = self.evaluate_board_state(self.current_grid, lines_cleared, True)
        
        # Add score component
        fitness += score * self.learning_config.score_weight
        
        print(f"\n=== Generation {self.generation}, Member {self.current_member + 1}/{self.population_size} ===")
        print(f"Score: {score}, Lines: {lines_cleared}, Moves: {moves}")
        print(f"Fitness: {fitness:.2f}")
        
        # Add fitness score and update tracking
        self.fitness_scores.append(fitness)
        self.state_history['fitness_history'].append(fitness)
        self.state_history['member_progress'].append(self.current_member)
        
        # Update member counter
        self.current_member += 1
        
        # Check for evolution
        if self.current_member >= self.population_size:
            print("\n=== Evolving to Next Generation ===")
            self._evolve_to_next_generation()
            self.save_state()  # Save state after evolution
        
        return fitness

    def _evolve_to_next_generation(self):
        """Handle generation transition"""
        diversity = self._calculate_population_diversity()
        self.generation_stats['population_diversity'].append(diversity)
        
        self.state_history['evolution_points'].append({
            'generation': self.generation,
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': sum(self.fitness_scores) / len(self.fitness_scores)
        })
        
        self.evolve_population()
        
        # Reset counters
        self.current_member = 0
        self.fitness_scores = []

    def evolve_population(self):
        """Create next generation using genetic algorithm"""
        self.generation += 1
        
        # Calculate statistics
        best_fitness = max(self.fitness_scores)
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        worst_fitness = min(self.fitness_scores)
        
        # Update best network if improved
        best_idx = self.fitness_scores.index(best_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_network = self.population[best_idx].clone()
            self.generation_stats['generation_best_network'] = self.population[best_idx].clone()
        
        # Sort population by fitness
        population_fitness = list(zip(self.population, self.fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Create new population
        new_population = []
        
        # Keep elite individuals
        for i in range(self.genetic_config.elite_size):
            new_population.append(population_fitness[i][0].clone())
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            if random.random() < self.genetic_config.crossover_rate:
                parent1 = self._tournament_select(population_fitness)
                parent2 = self._tournament_select(population_fitness)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            else:
                parent = self._tournament_select(population_fitness)
                child = parent.clone()
                child = self._mutate(child)
                new_population.append(child)
        
        # Update population and stats
        self.population = new_population
        self.generation_stats['best_fitness'].append(best_fitness)
        self.generation_stats['avg_fitness'].append(avg_fitness)
        self.generation_stats['worst_fitness'].append(worst_fitness)
        
        print(f"\nGeneration {self.generation} Stats:")
        print(f"Best Fitness: {best_fitness:.2f}")
        print(f"Average Fitness: {avg_fitness:.2f}")
        print(f"Population Size: {len(self.population)}")

    def _tournament_select(self, population_fitness: List[Tuple[TetrisNet, float]]) -> TetrisNet:
        """Select parent using tournament selection"""
        tournament = random.sample(population_fitness, self.genetic_config.tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, parent1: TetrisNet, parent2: TetrisNet) -> TetrisNet:
        """Create child network through crossover"""
        child = TetrisNet(self.network_config).to(self.device)
        for (name1, param1), (name2, param2) in zip(parent1.named_parameters(), 
                                                   parent2.named_parameters()):
            if random.random() < 0.5:
                child.state_dict()[name1].copy_(param1)
            else:
                child.state_dict()[name1].copy_(param2)
        return child

    def _mutate(self, network: TetrisNet) -> TetrisNet:
        """Apply random mutations to network parameters"""
        for param in network.parameters():
            if random.random() < self.genetic_config.mutation_rate:
                mutation = torch.randn_like(param) * self.genetic_config.mutation_strength
                param.data += mutation
        return network

    def save_state(self, path='tetris_genetic_state.pth'):
        """Save current state"""
        save_path = os.path.join('models', path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        state_dict = {
            'generation': self.generation,
            'current_member': self.current_member,
            'population': [net.state_dict() for net in self.population],
            'best_network': self.best_network.state_dict() if self.best_network else None,
            'best_fitness': self.best_fitness,
            'fitness_scores': self.fitness_scores,
            'generation_stats': self.generation_stats,
            'state_history': self.state_history,
            'network_config': self.network_config.__dict__,
            'genetic_config': self.genetic_config.__dict__,
            'learning_config': self.learning_config.__dict__
        }
        
        torch.save(state_dict, save_path)
        print(f"State saved to: {save_path}")

    def load_state(self, path='tetris_genetic_state.pth'):
        """Load saved state"""
        load_path = os.path.join('models', path)
        print(f"Loading state from: {load_path}")
        
        try:
            checkpoint = torch.load(load_path)
            
            # Load configurations
            self.network_config = NetworkConfig(**checkpoint['network_config'])
            self.genetic_config = GeneticConfig(**checkpoint['genetic_config'])
            self.learning_config = LearningConfig(**checkpoint['learning_config'])
            
            # Load core state
            self.generation = checkpoint.get('generation', 0)
            self.current_member = checkpoint.get('current_member', 0)
            self.fitness_scores = checkpoint.get('fitness_scores', [])
            self.best_fitness = checkpoint.get('best_fitness', float('-inf'))
            self.state_history = checkpoint.get('state_history', {
                'generations': [],
                'member_progress': [],
                'fitness_history': [],
                'evolution_points': []
            })
            
            # Load population
            self.population = []
            for state_dict in checkpoint['population']:
                net = TetrisNet(self.network_config).to(self.device)
                net.load_state_dict(state_dict)
                self.population.append(net)
            
            # Load best network if it exists
            if checkpoint.get('best_network'):
                self.best_network = TetrisNet(self.network_config).to(self.device)
                self.best_network.load_state_dict(checkpoint['best_network'])
            
            print(f"State loaded successfully - Generation: {self.generation}, Member: {self.current_member}")
            
        except Exception as e:
            print(f"Error loading state: {e}")
            self._init_state_tracking()