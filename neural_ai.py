import torch
import torch.nn as nn
import numpy as np
import random
import copy
from tetromino import TETROMINOES

class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        
        # State dimensionality
        input_size = 10 + 10 + 22 + 9 + 3  # 54 features total
        
        # Network architecture
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 80)  # Output size for possible moves
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        return x

class GeneticTetrisAI:
    def __init__(self, population_size=30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Population parameters
        self.population_size = population_size
        self.generation = 0
        self.population = [TetrisNet().to(self.device) for _ in range(population_size)]
        self.current_member = 0
        
        # Genetic parameters
        self.mutation_rate = 0.1
        self.mutation_strength = 0.2
        self.elite_size = 2
        
        # Fitness tracking
        self.fitness_scores = []
        self.best_fitness = float('-inf')
        self.best_network = None
        
        # Statistics
        self.generation_stats = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'generation_best_network': None
        }
        
        print(f"Genetic AI initialized with population size: {population_size}")
        print(f"Network parameters: ", sum(p.numel() for p in self.population[0].parameters()))

    def get_state(self, grid, tetromino, position):
        heights = self._get_heights(grid)
        holes = self._get_holes_per_column(grid)
        lines = self._get_lines_per_row(grid)
        bumpiness = self._get_bumpiness(heights)
        
        max_height = np.max(heights)
        avg_height = np.mean(heights)
        total_holes = np.sum(holes)
        
        # Normalize features
        heights = heights / grid.shape[0]
        holes = holes / max(1, max_height)
        lines = lines / grid.shape[1]
        bumpiness = bumpiness / max(1, max_height)
        
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
    
    def _get_heights(self, grid):
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

    def _get_holes_per_column(self, grid):
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

    def _get_lines_per_row(self, grid):
        lines = np.zeros(grid.shape[0])
        for y in range(grid.shape[0]):
            lines[y] = sum(1 for x in grid[y] if x != 0) / grid.shape[1]
        return lines

    def _get_bumpiness(self, heights):
        return np.array([abs(heights[i] - heights[i+1]) 
                        for i in range(len(heights)-1)])

    @staticmethod
    def count_connections(grid, position, tetromino):
        """Count how many sides of the tetromino connect with existing pieces."""
        connections = 0
        x, y = position
        
        for i in range(len(tetromino)):
            for j in range(len(tetromino[i])):
                if tetromino[i][j] != 0:
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        new_x, new_y = x + j + dx, y + i + dy
                        if (0 <= new_x < grid.shape[1] and 
                            0 <= new_y < grid.shape[0] and 
                            grid[new_y][new_x] != 0):
                            connections += 1
        return connections

    @staticmethod
    def count_gaps_filled(grid, position, tetromino):
        """Count how many gaps the tetromino fills."""
        gaps_filled = 0
        x, y = position
        
        for i in range(len(tetromino)):
            for j in range(len(tetromino[i])):
                if tetromino[i][j] != 0:
                    if GeneticTetrisAI.is_filling_gap(grid, x + j, y + i):
                        gaps_filled += 1
        return gaps_filled

    @staticmethod
    def is_filling_gap(grid, x, y):
        """Check if a position represents a gap that should be filled."""
        if y >= grid.shape[0] - 1:
            return False
            
        adjacent_blocks = 0
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < grid.shape[1] and 
                0 <= new_y < grid.shape[0] and 
                grid[new_y][new_x] != 0):
                adjacent_blocks += 1
        
        return adjacent_blocks >= 2

    def choose_action(self, state, possible_moves, grid, tetromino_key, tetromino):
        """Choose action using current network in population."""
        if not possible_moves:
            return None
        
        current_network = self.population[self.current_member]
        with torch.no_grad():
            q_values = current_network(state)
            
            # Evaluate each possible move
            move_scores = []
            for i, (rotation_index, position) in enumerate(possible_moves):
                current_tetromino = TETROMINOES[tetromino_key][rotation_index]
                
                # Get base score from network
                q_value = q_values[0][i].item() if i < q_values.size(1) else 0
                
                # Add heuristic components
                connections = self.count_connections(grid, position, current_tetromino)
                gaps_filled = self.count_gaps_filled(grid, position, current_tetromino)
                
                # Combine scores
                total_score = (q_value * 0.6 + 
                             connections * 0.2 + 
                             gaps_filled * 0.2)
                
                move_scores.append((total_score, rotation_index, position))
            
            # Choose best move
            best_score, rotation_index, position = max(move_scores)
            print(f"Move chosen: rot={rotation_index}, pos={position}, score={best_score:.2f}")
            return (rotation_index, position)

    def evolve_population(self):
        """Create next generation based on fitness scores."""
        self.generation += 1
        print(f"\nEvolving Generation {self.generation}")
        
        # Calculate statistics
        avg_fitness = np.mean(self.fitness_scores)
        best_gen_fitness = max(self.fitness_scores)
        worst_gen_fitness = min(self.fitness_scores)
        
        # Update best network if we found a better one
        best_idx = np.argmax(self.fitness_scores)
        if best_gen_fitness > self.best_fitness:
            self.best_fitness = best_gen_fitness
            self.best_network = copy.deepcopy(self.population[best_idx])
            self.generation_stats['generation_best_network'] = copy.deepcopy(self.population[best_idx])
        
        # Store generation statistics
        self.generation_stats['best_fitness'].append(best_gen_fitness)
        self.generation_stats['avg_fitness'].append(avg_fitness)
        self.generation_stats['worst_fitness'].append(worst_gen_fitness)
        
        # Sort population by fitness
        population_fitness = list(zip(self.population, self.fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Create new population
        new_population = []
        
        # Keep elite individuals
        for i in range(self.elite_size):
            new_population.append(copy.deepcopy(population_fitness[i][0]))
        
        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(population_fitness)
            parent2 = self._tournament_select(population_fitness)
            
            # Create and mutate offspring
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        # Update population
        self.population = new_population
        self.fitness_scores = []
        self.current_member = 0
        
        print(f"Generation {self.generation} Stats:")
        print(f"Best Fitness: {best_gen_fitness:.2f}")
        print(f"Average Fitness: {avg_fitness:.2f}")
        print(f"Worst Fitness: {worst_gen_fitness:.2f}")
        print(f"All-time Best Fitness: {self.best_fitness:.2f}")

    def _tournament_select(self, population_fitness, tournament_size=3):
        """Select parent using tournament selection."""
        tournament = random.sample(population_fitness, tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, parent1, parent2):
        """Create child network through crossover."""
        child = TetrisNet().to(self.device)
        
        # Uniform crossover for each parameter
        for (name1, param1), (name2, param2) in zip(parent1.named_parameters(), 
                                                   parent2.named_parameters()):
            if random.random() < 0.5:
                child.state_dict()[name1].copy_(param1)
            else:
                child.state_dict()[name1].copy_(param2)
        return child

    def _mutate(self, network):
        """Apply random mutations to network parameters."""
        for param in network.parameters():
            if random.random() < self.mutation_rate:
                mutation = torch.randn_like(param) * self.mutation_strength
                param.data += mutation
        return network

    def add_fitness_score(self, score, moves, lines_cleared, max_height):
        """Calculate and add fitness score for current member."""
        fitness = (
            score * 1.0 +           # Base score
            lines_cleared * 100 +    # Lines cleared bonus
            moves * 0.5 -           # Survival bonus
            max_height * 10         # Height penalty
        )
        self.fitness_scores.append(fitness)
        
        # Move to next member of population
        self.current_member += 1
        
        # If we've evaluated all members, evolve to next generation
        if self.current_member >= self.population_size:
            self.evolve_population()
        
        return fitness

    def save_state(self, path):
        """Save the current state of the genetic algorithm."""
        torch.save({
            'generation': self.generation,
            'population': [net.state_dict() for net in self.population],
            'best_network': self.best_network.state_dict() if self.best_network else None,
            'best_fitness': self.best_fitness,
            'generation_stats': self.generation_stats,
            'current_member': self.current_member,
            'fitness_scores': self.fitness_scores
        }, path)
        print(f"Genetic state saved to {path}")

    def load_state(self, path):
        """Load a previously saved state."""
        checkpoint = torch.load(path)
        
        self.generation = checkpoint['generation']
        self.current_member = checkpoint['current_member']
        self.fitness_scores = checkpoint['fitness_scores']
        self.best_fitness = checkpoint['best_fitness']
        self.generation_stats = checkpoint['generation_stats']
        
        # Load population
        self.population = []
        for state_dict in checkpoint['population']:
            net = TetrisNet().to(self.device)
            net.load_state_dict(state_dict)
            self.population.append(net)
            
        # Load best network if it exists
        if checkpoint['best_network']:
            self.best_network = TetrisNet().to(self.device)
            self.best_network.load_state_dict(checkpoint['best_network'])
            
        print(f"Genetic state loaded from {path}")