# src/orchestrator/workflow.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

@dataclass
class Task:
    """Task structure for workflow"""
    task_id: str
    task_type: str
    agent_name: str
    dependencies: List[str]
    parameters: Dict[str, Any]
    status: str = 'pending'
    result: Optional[Any] = None

class WorkflowEngine:
    """Manages paper generation workflow"""
    
    def __init__(self):
        self.tasks = {}
        self.completed_tasks = set()
        self.current_phase = 'initialization'
        
        self.phases = [
            'initialization',
            'literature_review', 
            'methodology_design',
            'analysis',
            'writing',
            'revision',
            'finalization'
        ]
        
    def add_task(self, task: Task):
        """Add a task to the workflow"""
        self.tasks[task.task_id] = task
        
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute"""
        ready = []
        for task in self.tasks.values():
            if task.status == 'pending':
                if all(dep in self.completed_tasks for dep in task.dependencies):
                    ready.append(task)
        return ready
    
    def complete_task(self, task_id: str, result: Any):
        """Mark a task as completed"""
        if task_id in self.tasks:
            self.tasks[task_id].status = 'completed'
            self.tasks[task_id].result = result
            self.completed_tasks.add(task_id)
    
    def get_progress(self) -> float:
        """Get overall workflow progress"""
        if not self.tasks:
            return 0.0
        return len(self.completed_tasks) / len(self.tasks)

class TaskScheduler:
    """Schedules tasks for agents"""
    
    def __init__(self):
        self.agent_queues = {
            'literature': [],
            'methodology': [],
            'analysis': [],
            'writing': []
        }
        self.time_budget = 100
        self.time_spent = 0
        self.start_time = None
        
    def reset_timer(self):
        """Reset the timer for a new episode"""
        self.start_time = time.time()
        self.time_spent = 0
        
    def add_task(self, agent_name: str, task: Task):
        """Add a task to an agent's queue"""
        if agent_name in self.agent_queues:
            self.agent_queues[agent_name].append(task)
            
    def get_next_task(self, agent_name: str) -> Optional[Task]:
        """Get the next task for an agent"""
        if agent_name in self.agent_queues and self.agent_queues[agent_name]:
            if self.start_time is not None:
                self.time_spent = time.time() - self.start_time
            return self.agent_queues[agent_name].pop(0)
        return None
    
    def get_time_remaining(self) -> float:
        """Get remaining time budget"""
        return max(0, self.time_budget - self.time_spent)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time