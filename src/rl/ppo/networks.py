import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class PPOActor(nn.Module):
    """Actor network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = False):
        super(PPOActor, self).__init__()
        self.continuous = continuous
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        if continuous:
            self.mean = nn.Linear(128, action_dim)
            self.log_std = nn.Linear(128, action_dim)
        else:
            self.action_head = nn.Linear(128, action_dim)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.continuous:
            mean = self.mean(x)
            log_std = self.log_std(x).clamp(-20, 2)
            std = log_std.exp()
            return Normal(mean, std)
        else:
            logits = self.action_head(x)
            return Categorical(logits=logits)

class PPOCritic(nn.Module):
    """Critic network for PPO"""
    
    def __init__(self, state_dim: int):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value_head(x)

