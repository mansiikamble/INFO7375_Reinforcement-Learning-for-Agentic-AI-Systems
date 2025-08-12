from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.memory = []
        self.state_dim = config.get('state_dim', 128)
        self.action_dim = config.get('action_dim', 4)
        
    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """Select an action given a state"""
        pass
    
    @abstractmethod
    def learn(self, experiences: List[Tuple]) -> Dict[str, float]:
        """Learn from experiences"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent model"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent model"""
        pass

class BaseEnvironment(ABC):
    """Base environment for paper generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = None
        self.done = False
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment"""
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render current state"""
        pass