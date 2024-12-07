import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tetromino import TETROMINOES

class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        
        # State dimensionality: heights + holes + lines + bumpiness + max_height + avg_height + total_holes
        input_size = 10 + 10 + 22 + 9 + 3  # 54 features total
        
        # Match the architecture of the saved model
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 80)
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.01
            
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        return x

class TetrisAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = TetrisNet().to(self.device)
        self.target_model = TetrisNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.HuberLoss()
        
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.gamma = 0.99
        
        self.target_update = 5
        self.episode_count = 0
        self.latest_loss = 0
        self.latest_q_value = 0
        
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_values': [],
            'losses': [],
            'epsilons': []
        }
        
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001
        
        print("Neural network initialized with trainable parameters:",
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))

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
                    if TetrisAI.is_filling_gap(grid, x + j, y + i):
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

    def get_state(self, grid, tetromino, position):
        heights = self._get_heights(grid)
        holes = self._get_holes_per_column(grid)
        lines = self._get_lines_per_row(grid)
        bumpiness = self._get_bumpiness(heights)
        
        max_height = np.max(heights)
        avg_height = np.mean(heights)
        total_holes = np.sum(holes)
        
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

    def choose_action(self, state, possible_moves, grid, tetromino_key, tetromino):
        if not possible_moves:
            return None
        
        if random.random() < self.epsilon:
            move_scores = []
            for rotation_index, position in possible_moves:
                current_tetromino = TETROMINOES[tetromino_key][rotation_index]
                connections = self.count_connections(grid, position, current_tetromino)
                gaps_filled = self.count_gaps_filled(grid, position, current_tetromino)
                score = connections * 2 + gaps_filled * 3
                move_scores.append((score, rotation_index, position))
            
            top_moves = sorted(move_scores, reverse=True)[:3]
            _, rotation_index, position = random.choice(top_moves)
            print(f"Exploration move chosen: rot={rotation_index}, pos={position}")
            return (rotation_index, position)
        
        with torch.no_grad():
            q_values = self.model(state)
            self.latest_q_value = q_values.max().item()
            
            combined_scores = []
            for i, (rotation_index, position) in enumerate(possible_moves):
                current_tetromino = TETROMINOES[tetromino_key][rotation_index]
                connections = self.count_connections(grid, position, current_tetromino)
                gaps_filled = self.count_gaps_filled(grid, position, current_tetromino)
                
                q_value = q_values[0][i].item() if i < q_values.size(1) else 0
                connection_score = connections * 2 + gaps_filled * 3
                combined_score = q_value * 0.7 + connection_score * 0.3
                
                combined_scores.append((combined_score, rotation_index, position))
            
            best_score, rotation_index, position = max(combined_scores)
            print(f"Exploitation move chosen: rot={rotation_index}, pos={position}, score={best_score:.2f}")
            return (rotation_index, position)

    def train(self, state, action, reward, next_state, done):
        action = min(action, self.model.fc4.out_features - 1)
        priority = 1.0
        self.memory.append((state, action, reward, next_state, done, priority))
        
        if len(self.memory) < self.batch_size:
            return
            
        priorities = np.array([exp[5] for exp in self.memory])
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        self.model.train()
        current_q = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            self.target_model.eval()
            next_q = self.target_model(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        self.latest_loss = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        with torch.no_grad():
            td_errors = abs(current_q.squeeze() - target_q).cpu().numpy()
            for idx, error in zip(indices, td_errors):
                self.memory[idx] = (*self.memory[idx][:-1], error)
        
        if len(self.training_stats['losses']) % 100 == 0:
            self.print_training_stats()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

    def update_target_network(self):
        self.episode_count += 1
        if self.episode_count % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"Target network updated at episode {self.episode_count}")

    def save_model(self, path, best_score=0):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'memory': list(self.memory),
            'training_stats': self.training_stats,
            'best_score': best_score
        }, path)
        print(f"Model saved to {path} with best score: {best_score}")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.memory = deque(checkpoint.get('memory', []), maxlen=50000)
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"Model loaded from {path}")

    def get_latest_loss(self):
        return self.latest_loss

    def get_latest_q_value(self):
        return self.latest_q_value
    
    def log_episode(self, total_reward, episode_length):
        """Log statistics for the completed episode."""
        self.training_stats['episode_rewards'].append(total_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        self.training_stats['q_values'].append(self.latest_q_value)
        self.training_stats['losses'].append(self.latest_loss)
        self.training_stats['epsilons'].append(self.epsilon)
        
        # Print progress every 10 episodes
        if len(self.training_stats['episode_rewards']) % 10 == 0:
            recent_rewards = self.training_stats['episode_rewards'][-10:]
            recent_lengths = self.training_stats['episode_lengths'][-10:]
            print("\nLast 10 Episodes Statistics:")
            print(f"Average Reward: {np.mean(recent_rewards):.2f}")
            print(f"Average Length: {np.mean(recent_lengths):.2f}")
            print(f"Epsilon: {self.epsilon:.3f}")
            print(f"Latest Loss: {self.latest_loss:.6f}")
            print(f"Latest Q-Value: {self.latest_q_value:.6f}")