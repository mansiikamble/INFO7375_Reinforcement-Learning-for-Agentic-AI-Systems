import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

from .networks import PPOActor, PPOCritic

class PPOAgent:
    """PPO Agent implementation"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Device
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Networks
        self.actor = PPOActor(state_dim, action_dim, config.get('continuous', False))
        self.critic = PPOCritic(state_dim)
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                         lr=config.get('learning_rate', 0.0003))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                          lr=config.get('learning_rate', 0.0003))
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Handle continuous actions (multi-dimensional)
        if self.config.get('continuous', False):
            # For continuous actions, return the action as numpy array
            return action.squeeze(0).cpu().numpy(), log_prob.sum().item(), value.item()
        else:
            # For discrete actions, return scalar
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition for batch update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_gae(self, next_value: float) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = self.values[t + 1]
                
            delta = self.rewards[t] + self.config.get('gamma', 0.99) * next_value_t * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.config.get('gamma', 0.99) * self.config.get('gae_lambda', 0.95) * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.FloatTensor(advantages).to(self.device)
    
    def learn(self) -> Dict[str, float]:
        """Update policy using PPO"""
        if len(self.states) == 0:
            return {}
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        
        # Handle different action types
        if self.config.get('continuous', False):
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        else:
            actions = torch.LongTensor(self.actions).to(self.device)
            
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            next_value = self.critic(states[-1].unsqueeze(0)).item()
        advantages = self.compute_gae(next_value)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.config.get('n_epochs', 10)):
            # Actor loss
            dist = self.actor(states)
            new_log_probs = dist.log_prob(actions)
            
            # For continuous actions, sum log probs across dimensions
            if self.config.get('continuous', False) and len(new_log_probs.shape) > 1:
                new_log_probs = new_log_probs.sum(dim=1)
                
            ratio = (new_log_probs - old_log_probs).exp()
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 
                               1 - self.config.get('clip_range', 0.2), 
                               1 + self.config.get('clip_range', 0.2)) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses)
        }