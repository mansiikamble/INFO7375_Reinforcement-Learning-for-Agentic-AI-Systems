import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)