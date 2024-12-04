import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        
        # State dimensionality: heights + holes + lines + bumpiness + max_height + avg_height + total_holes
        input_size = 10 + 10 + 22 + 9 + 3  # 54 features total
        
        # Improved network architecture
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 77)  # Replace 77 with the maximum number of possible moves for your game
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Reduced dropout rate
        
    def forward(self, x):
        # Add small noise to input for better generalization
        if self.training:
            x = x + torch.randn_like(x) * 0.01
            
        # Forward pass with batch normalization
        x = self.fc1(x)
        if x.size(0) > 1:  # Only apply batch norm for batch size > 1
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
        
        # Initialize networks
        self.model = TetrisNet().to(self.device)
        self.target_model = TetrisNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # Reduced learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', 
                                                            factor=0.5, patience=5, verbose=True)
        self.criterion = nn.HuberLoss()  # Use Huber loss for better stability
        
        # Experience replay
        self.memory = deque(maxlen=50000)  # Increased memory size
        self.batch_size = 32
        
        # Exploration parameters
        self.epsilon = 0.95        # Start with high exploration
        self.epsilon_min = 0.05    # Minimum exploration
        self.epsilon_decay = 0.99  # Slower decay
        self.gamma = 0.95         # Discount factor
        
        # Training metrics
        self.target_update = 5    # Update target network more frequently
        self.episode_count = 0
        self.latest_loss = 0
        self.latest_q_value = 0
        
        # Priority replay parameters
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001

    def get_state(self, grid, tetromino, position):
        # Calculate basic features
        heights = self._get_heights(grid)
        holes = self._get_holes_per_column(grid)
        lines = self._get_lines_per_row(grid)
        bumpiness = self._get_bumpiness(heights)
        
        # Additional features
        max_height = np.max(heights)
        avg_height = np.mean(heights)
        total_holes = np.sum(holes)
        
        # Normalize features
        heights = heights / grid.shape[0]
        holes = holes / max(1, max_height)
        lines = lines / grid.shape[1]
        bumpiness = bumpiness / max(1, max_height)
        
        # Combine all features
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

    def choose_action(self, state, possible_moves):
        if not possible_moves:
            return None
            
        if random.random() < self.epsilon:
            # During exploration, prefer moves towards the center
            center_x = 5  # Assuming grid width of 10
            sorted_moves = sorted(possible_moves, 
                                key=lambda m: abs(m[1][0] - center_x))
            move = sorted_moves[0] if random.random() < 0.7 else random.choice(possible_moves)
            print(f"Random move chosen: {move}")
            return move
            
        with torch.no_grad():
            q_values = self.model(state)
            self.latest_q_value = q_values.max().item()
            
            # Create a mask for possible moves
            valid_moves = torch.zeros_like(q_values)
            for i, move in enumerate(possible_moves):
                if i < valid_moves.size(1):
                    valid_moves[0][i] = 1
            
            # Set q_values of invalid moves to negative infinity
            q_values = q_values.masked_fill(valid_moves == 0, float('-inf'))
            
            # Choose best valid move
            move_idx = q_values.argmax().item()
            if move_idx < len(possible_moves):
                print(f"Q-value based move chosen: {possible_moves[move_idx]}")
                return possible_moves[move_idx]
            
            # Fallback to random move if something goes wrong
            return random.choice(possible_moves)

    def train(self, state, action, reward, next_state, done):
        # Store experience with priority
        priority = 1.0  # Max priority for new experiences
        self.memory.append((state, action, reward, next_state, done, priority))
        
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch with priorities
        priorities = np.array([exp[5] for exp in self.memory])
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.priority_beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Update beta
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        
        # Current Q values
        self.model.train()
        current_q = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            self.target_model.eval()
            next_q = self.target_model(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss with importance sampling weights
        loss = self.criterion(current_q.squeeze(), target_q)
        loss = (loss * weights).mean()
        
        self.latest_loss = loss.item()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = abs(current_q.squeeze() - target_q).cpu().numpy()
            for idx, error in zip(indices, td_errors):
                self.memory[idx] = (*self.memory[idx][:-1], error)
        
        # Decay epsilon
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

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'memory': list(self.memory)
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.memory = deque(checkpoint['memory'], maxlen=50000)
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"Model loaded from {path}")

    def get_latest_loss(self):
        return self.latest_loss

    def get_latest_q_value(self):
        return self.latest_q_value