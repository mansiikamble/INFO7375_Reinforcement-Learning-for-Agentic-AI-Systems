# src/agents/methodology/methodology_agent.py
import numpy as np
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.rl.dqn.agent import DQNAgent

class MethodologyAgent(DQNAgent):
    """Agent specialized in research methodology design"""
    
    def __init__(self, config: Dict[str, Any]):
        # Ensure device is set
        if 'device' not in config:
            config['device'] = 'cpu'
            
        # Ensure gamma is set
        if 'gamma' not in config:
            config['gamma'] = 0.99
            
        super().__init__(
            state_dim=config['state_dim'],
            action_dim=8,  # Different methodology choices
            config=config
        )
        
        self.methodology_templates = {
            'experimental': ['hypothesis', 'variables', 'controls', 'procedures'],
            'computational': ['algorithms', 'datasets', 'metrics', 'baselines'],
            'theoretical': ['definitions', 'axioms', 'theorems', 'proofs'],
            'survey': ['scope', 'selection_criteria', 'analysis_method'],
            'mixed_methods': ['quantitative', 'qualitative', 'integration']
        }
        
    def encode_state(self, research_question: str, paper_type: str, 
                     available_resources: Dict) -> np.ndarray:
        """Encode methodology selection state"""
        # Encode research question
        question_features = self.extract_question_features(research_question)
        
        # Paper type encoding
        type_encoding = self.one_hot_encode(paper_type, list(self.methodology_templates.keys()))
        
        # Resource constraints
        resource_features = np.array([
            available_resources.get('time_weeks', 12) / 52,
            available_resources.get('budget', 10000) / 100000,
            available_resources.get('team_size', 1) / 10,
            available_resources.get('compute_resources', 1) / 10
        ])
        
        # Combine all features
        state = np.concatenate([question_features, type_encoding, resource_features])
        
        # Pad or truncate to match state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
            
        return state
    
    def extract_question_features(self, question: str) -> np.ndarray:
        """Extract features from research question"""
        features = []
        question_lower = question.lower()
        
        # Question type indicators
        features.append(float('how' in question_lower))
        features.append(float('why' in question_lower))
        features.append(float('what' in question_lower))
        features.append(float('when' in question_lower))
        features.append(float('where' in question_lower))
        features.append(float('compare' in question_lower or 'versus' in question_lower))
        features.append(float('effect' in question_lower or 'impact' in question_lower))
        features.append(float('predict' in question_lower or 'forecast' in question_lower))
        
        # Domain indicators
        features.append(float('algorithm' in question_lower or 'model' in question_lower))
        features.append(float('system' in question_lower or 'framework' in question_lower))
        features.append(float('theory' in question_lower or 'hypothesis' in question_lower))
        features.append(float('empirical' in question_lower or 'experimental' in question_lower))
        features.append(float('optimize' in question_lower or 'improve' in question_lower))
        
        return np.array(features)
    
    def one_hot_encode(self, value: str, categories: List[str]) -> np.ndarray:
        """One-hot encode categorical variable"""
        encoding = np.zeros(len(categories))
        if value in categories:
            encoding[categories.index(value)] = 1
        return encoding
    
    def recommend_methodology(self, state: np.ndarray) -> Dict[str, Any]:
        """Recommend research methodology based on state"""
        action = self.act(state, training=False)
        
        methodology = {
            0: self.design_experimental_study,
            1: self.design_computational_study,
            2: self.design_theoretical_study,
            3: self.design_survey_study,
            4: self.design_case_study,
            5: self.design_mixed_methods,
            6: self.design_meta_analysis,
            7: self.design_replication_study
        }[action]()
        
        return methodology
    
    def design_experimental_study(self) -> Dict[str, Any]:
        """Design experimental methodology"""
        return {
            'type': 'experimental',
            'components': {
                'hypothesis': self.generate_hypothesis(),
                'variables': {
                    'independent': self.identify_independent_variables(),
                    'dependent': self.identify_dependent_variables(),
                    'controlled': self.identify_control_variables()
                },
                'design': self.select_experimental_design(),
                'sample_size': self.calculate_sample_size(),
                'procedures': self.outline_procedures(),
                'analysis_plan': self.plan_statistical_analysis()
            }
        }
    
    def design_computational_study(self) -> Dict[str, Any]:
        """Design computational methodology"""
        return {
            'type': 'computational',
            'components': {
                'algorithms': ['DQN', 'PPO', 'Multi-agent coordination'],
                'datasets': ['arXiv papers', 'Semantic Scholar data'],
                'metrics': ['Quality score', 'Coherence', 'Citation relevance'],
                'baselines': ['Random selection', 'Single agent', 'No coordination'],
                'implementation': {
                    'framework': 'PyTorch',
                    'hardware': 'GPU recommended',
                    'reproducibility': 'Random seeds, versioned dependencies'
                }
            }
        }
    
    def design_theoretical_study(self) -> Dict[str, Any]:
        """Design theoretical methodology"""
        return {
            'type': 'theoretical',
            'components': {
                'definitions': ['Agent coordination', 'Convergence criteria'],
                'axioms': ['Rational agent behavior', 'Reward maximization'],
                'theorems': ['Convergence theorem', 'Optimality conditions'],
                'proofs': ['Mathematical derivations', 'Formal verification']
            }
        }
    
    def design_survey_study(self) -> Dict[str, Any]:
        """Design survey methodology"""
        return {
            'type': 'survey',
            'components': {
                'scope': 'Multi-agent reinforcement learning systems',
                'selection_criteria': {
                    'time_range': '2019-2024',
                    'venues': 'Top AI conferences and journals',
                    'quality': 'Peer-reviewed publications'
                },
                'analysis_method': 'Systematic literature review with thematic analysis'
            }
        }
    
    def design_case_study(self) -> Dict[str, Any]:
        """Design case study methodology"""
        return {
            'type': 'case_study',
            'components': {
                'case_selection': 'Representative critical case',
                'data_collection': ['System logs', 'Performance metrics', 'User feedback'],
                'analysis': 'Mixed methods analysis',
                'validity': 'Triangulation of data sources'
            }
        }
    
    def design_mixed_methods(self) -> Dict[str, Any]:
        """Design mixed methods approach"""
        return {
            'type': 'mixed_methods',
            'components': {
                'quantitative': self.design_experimental_study(),
                'qualitative': {
                    'method': 'Content analysis',
                    'data': 'Generated papers',
                    'coding': 'Thematic coding'
                },
                'integration': 'Convergent parallel design'
            }
        }
    
    def design_meta_analysis(self) -> Dict[str, Any]:
        """Design meta-analysis methodology"""
        return {
            'type': 'meta_analysis',
            'components': {
                'search_strategy': 'Systematic database search',
                'inclusion_criteria': 'Empirical studies with effect sizes',
                'effect_size': 'Standardized mean difference',
                'synthesis': 'Random-effects model',
                'heterogeneity': 'I-squared statistic'
            }
        }
    
    def design_replication_study(self) -> Dict[str, Any]:
        """Design replication study"""
        return {
            'type': 'replication',
            'components': {
                'original_study': 'Reference to original work',
                'modifications': 'Direct replication with extended scope',
                'comparison': 'Statistical comparison with original',
                'contribution': 'Validation and generalization'
            }
        }
    
    # Helper methods for experimental design
    def generate_hypothesis(self) -> str:
        """Generate research hypothesis"""
        return "Multi-agent reinforcement learning with coordinated training will produce higher quality research papers than independent agent systems"
    
    def identify_independent_variables(self) -> List[str]:
        """Identify independent variables"""
        return [
            "Coordination mechanism (none, sequential, parallel)",
            "Learning algorithm (DQN, PPO)",
            "Number of agents",
            "Communication frequency"
        ]
    
    def identify_dependent_variables(self) -> List[str]:
        """Identify dependent variables"""
        return [
            "Paper quality score",
            "Generation time",
            "Coherence score",
            "Citation relevance"
        ]
    
    def identify_control_variables(self) -> List[str]:
        """Identify control variables"""
        return [
            "Random seed",
            "Hardware specifications",
            "Software versions",
            "Dataset split"
        ]
    
    def select_experimental_design(self) -> str:
        """Select experimental design"""
        return "2x2 factorial design with repeated measures"
    
    def calculate_sample_size(self) -> int:
        """Calculate required sample size"""
        # Power analysis (simplified)
        effect_size = 0.5  # Medium effect
        power = 0.8
        alpha = 0.05
        # Approximate calculation
        return 64
    
    def outline_procedures(self) -> List[str]:
        """Outline experimental procedures"""
        return [
            "1. Initialize experimental environment",
            "2. Set random seeds for reproducibility",
            "3. Train baseline models",
            "4. Train proposed multi-agent system",
            "5. Generate papers using both approaches",
            "6. Evaluate paper quality using metrics",
            "7. Statistical analysis of results",
            "8. Report findings"
        ]
    
    def plan_statistical_analysis(self) -> Dict[str, Any]:
        """Plan statistical analysis"""
        return {
            "descriptive": ["Mean", "Standard deviation", "Confidence intervals"],
            "inferential": {
                "tests": ["Paired t-test", "ANOVA", "Wilcoxon signed-rank"],
                "corrections": "Bonferroni correction for multiple comparisons",
                "effect_size": "Cohen's d"
            },
            "assumptions": ["Normality", "Homogeneity of variance"],
            "software": "Python (scipy, statsmodels)"
        }
    
    # Orchestrator integration methods
    def get_workload(self) -> float:
        """Get current workload (for orchestrator)"""
        # Simple implementation - could be based on current tasks
        return 0.5
    
    def get_performance_score(self) -> float:
        """Get performance score (for orchestrator)"""
        # Could be based on quality of methodologies designed
        return 0.8
    
    def get_coordination_score(self) -> float:
        """Get coordination score (for orchestrator)"""
        # Methodology agent is generally independent
        return 0.9
    
    def assign_task(self, task):
        """Assign a task to the agent"""
        # Handle task assignment from orchestrator
        if hasattr(task, 'parameters'):
            research_question = task.parameters.get('research_question', '')
            paper_type = task.parameters.get('paper_type', 'research')
            
            # Create state
            resources = {
                'time_weeks': 12,
                'budget': 10000,
                'team_size': 4,
                'compute_resources': 2
            }
            
            state = self.encode_state(research_question, paper_type, resources)
            
            # Get methodology recommendation
            methodology = self.recommend_methodology(state)
            
            # Store result (in real implementation, would communicate back to orchestrator)
            self._last_methodology = methodology