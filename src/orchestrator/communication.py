# src/orchestrator/communication.py
from queue import PriorityQueue
from typing import Dict, Any, List
import threading
import time

class Message:
    """Message structure for inter-agent communication"""
    def __init__(self, sender: str, receiver: str, content: Any, priority: int = 5):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.priority = priority
        self.timestamp = time.time()
    
    def __lt__(self, other):
        return self.priority < other.priority

class SharedMemorySystem:
    """Thread-safe shared memory for agents"""
    
    def __init__(self):
        self._memory = {}
        self._lock = threading.RLock()
        
    def set(self, key: str, value: Any):
        """Set a value in shared memory"""
        with self._lock:
            self._memory[key] = value
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared memory"""
        with self._lock:
            return self._memory.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple values"""
        with self._lock:
            self._memory.update(updates)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all memory contents"""
        with self._lock:
            return self._memory.copy()