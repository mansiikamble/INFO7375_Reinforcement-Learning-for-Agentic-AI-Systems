import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, Any, Tuple, List

from .networks import DQN  # Import DQN from networks.py

class DQNAgent:
    """DQN Agent implementation"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Device
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=config.get('learning_rate', 0.001))
        self.memory = deque(maxlen=config.get('buffer_size', 10000))
        self.epsilon = config.get('epsilon_start', 1.0)
        
        # Counters
        self.steps = 0
        self.update_counter = 0
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def learn(self) -> Dict[str, float]:
        """Update Q-network"""
        if len(self.memory) < self.config.get('batch_size', 32):
            return {}
            
        batch = random.sample(self.memory, self.config.get('batch_size', 32))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.get('gamma', 0.99) * next_q_values
            
        # Loss and optimization
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.config.get('target_update_frequency', 100) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Update epsilon
        self.epsilon = max(self.config.get('epsilon_end', 0.01), 
                          self.epsilon * self.config.get('epsilon_decay', 0.995))
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}