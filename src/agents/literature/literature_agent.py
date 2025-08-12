# src/agents/literature/literature_agent.py
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.rl.dqn.agent import DQNAgent
from src.tools.builtin.arxiv_tool import ArxivTool
from src.tools.builtin.semantic_scholar_tool import SemanticScholarTool

class LiteratureReviewAgent(DQNAgent):
    """Agent specialized in literature review using DQN"""
    
    def __init__(self, config: Dict[str, Any]):
        # Ensure device is set
        if 'device' not in config:
            config['device'] = 'cpu'
        
        # Ensure gamma is set
        if 'gamma' not in config:
            config['gamma'] = 0.99
            
        # Initialize DQN components
        super().__init__(
            state_dim=config['state_dim'],
            action_dim=6,  # Actions: cite, skip, deep_dive, find_similar, check_citations, synthesize
            config=config
        )
        
        # Literature-specific components
        self.encoder = SentenceTransformer('allenai-specter')
        self.arxiv_tool = ArxivTool()
        self.semantic_tool = SemanticScholarTool()
        
        # Memory for paper analysis
        self.paper_memory = {
            'cited_papers': [],
            'rejected_papers': [],
            'key_themes': [],
            'research_gaps': []
        }
        
    def encode_state(self, query: str, found_papers: List[Dict], 
                     current_citations: List[str]) -> np.ndarray:
        """Encode current literature review state"""
        # Encode query
        query_embedding = self.encoder.encode(query)
        
        # Encode found papers (average)
        if found_papers:
            paper_texts = [p['title'] + ' ' + p.get('abstract', '') for p in found_papers[:5]]
            papers_embedding = self.encoder.encode(paper_texts).mean(axis=0)
        else:
            papers_embedding = np.zeros(384)
            
        # Citation statistics
        citation_stats = np.array([
            len(current_citations),
            len(self.paper_memory['key_themes']),
            len(self.paper_memory['research_gaps'])
        ])
        
        # Combine into state vector
        state = np.concatenate([
            query_embedding[:128],  # Truncate to fit state_dim
            papers_embedding[:125],  # Leave room for citation stats
            citation_stats
        ])
        
        # Ensure state matches expected dimension
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
            
        return state
    
    def execute_action(self, action: int, paper: Dict) -> Tuple[float, Dict]:
        """Execute the selected action and return reward"""
        reward = 0
        info = {}
        
        if action == 0:  # Cite paper
            self.paper_memory['cited_papers'].append(paper)
            reward = self.calculate_citation_reward(paper)
            info['action'] = 'cited'
            
        elif action == 1:  # Skip paper
            self.paper_memory['rejected_papers'].append(paper.get('paperId', paper.get('title', '')))
            reward = -0.1  # Small negative reward for skipping
            info['action'] = 'skipped'
            
        elif action == 2:  # Deep dive into paper
            related = self.semantic_tool.get_references(paper.get('paperId', ''))
            reward = 0.5 * len(related) / 10  # Reward for finding related work
            info['action'] = 'deep_dive'
            info['found_references'] = len(related)
            
        elif action == 3:  # Find similar papers
            similar = self.arxiv_tool.search_similar(paper.get('title', ''))
            reward = 0.3 * len(similar) / 5
            info['action'] = 'find_similar'
            info['found_similar'] = len(similar)
            
        elif action == 4:  # Check citations of paper
            citations = self.semantic_tool.get_citations(paper.get('paperId', ''))
            reward = self.evaluate_citation_quality(citations)
            info['action'] = 'check_citations'
            
        elif action == 5:  # Synthesize theme
            theme = self.extract_theme(paper)
            if theme not in self.paper_memory['key_themes']:
                self.paper_memory['key_themes'].append(theme)
                reward = 1.0  # High reward for new theme
            info['action'] = 'synthesize'
            
        return reward, info
    
    def calculate_citation_reward(self, paper: Dict) -> float:
        """Calculate reward for citing a paper"""
        reward = 0.0
        
        # Relevance to existing citations
        if self.paper_memory['cited_papers']:
            existing_titles = [p.get('title', '') for p in self.paper_memory['cited_papers']]
            existing_embeddings = self.encoder.encode(existing_titles)
            new_embedding = self.encoder.encode(paper.get('title', ''))
            
            # Reward diversity
            max_similarity = np.max(np.dot(existing_embeddings, new_embedding))
            diversity_reward = 1.0 - max_similarity
            reward += 0.3 * diversity_reward
            
        # Quality metrics
        reward += min(paper.get('citationCount', 0) / 100, 1.0) * 0.3
        reward += min(paper.get('influentialCitationCount', 0) / 20, 1.0) * 0.2
        
        # Recency bonus
        year = paper.get('year', 0)
        if year >= 2020:
            reward += 0.2
            
        return reward
    
    def evaluate_citation_quality(self, citations: List[Dict]) -> float:
        """Evaluate quality of citations"""
        if not citations:
            return 0.0
            
        total_quality = 0.0
        for citation in citations[:10]:  # Look at top 10 citations
            # Quality based on citation count of citing papers
            cite_count = citation.get('citationCount', 0)
            total_quality += min(cite_count / 50, 1.0)
            
        return total_quality / 10
    
    def extract_theme(self, paper: Dict) -> str:
        """Extract main theme from paper"""
        # Simple theme extraction from title
        title_words = paper.get('title', '').lower().split()
        
        # Common ML/AI themes
        themes = {
            'deep learning': ['deep', 'neural', 'network', 'cnn', 'rnn', 'transformer'],
            'reinforcement learning': ['reinforcement', 'rl', 'q-learning', 'policy', 'reward'],
            'nlp': ['language', 'nlp', 'text', 'linguistic', 'bert', 'gpt'],
            'computer vision': ['vision', 'image', 'visual', 'detection', 'segmentation'],
            'optimization': ['optimization', 'gradient', 'convergence', 'loss'],
            'theory': ['theory', 'theoretical', 'analysis', 'proof', 'bound']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in title_words for keyword in keywords):
                return theme
                
        return 'general ai'
    
    def synthesize_introduction(self) -> str:
        """Create introduction for literature review"""
        intro = "This literature review examines recent advances in the field. "
        intro += f"We analyzed {len(self.paper_memory['cited_papers'])} key papers "
        intro += f"and identified {len(self.paper_memory['key_themes'])} major themes."
        return intro
    
    def organize_by_themes(self) -> Dict[str, List[Dict]]:
        """Organize papers by themes"""
        themed_papers = {theme: [] for theme in self.paper_memory['key_themes']}
        
        for paper in self.paper_memory['cited_papers']:
            theme = self.extract_theme(paper)
            if theme in themed_papers:
                themed_papers[theme].append(paper)
                
        return themed_papers
    
    def trace_development(self) -> List[Dict]:
        """Trace chronological development"""
        # Sort papers by year
        sorted_papers = sorted(
            self.paper_memory['cited_papers'],
            key=lambda p: p.get('year', 0)
        )
        
        return sorted_papers
    
    def identify_gaps(self) -> List[str]:
        """Identify research gaps"""
        gaps = []
        
        # Simple gap identification based on themes
        all_themes = set(self.paper_memory['key_themes'])
        expected_themes = {'deep learning', 'reinforcement learning', 'nlp', 'computer vision'}
        
        missing_themes = expected_themes - all_themes
        for theme in missing_themes:
            gaps.append(f"Limited research found on {theme}")
            
        # Add more sophisticated gap detection
        if len(self.paper_memory['cited_papers']) < 10:
            gaps.append("Insufficient literature coverage - more papers needed")
            
        return gaps
    
    def suggest_directions(self) -> List[str]:
        """Suggest future research directions"""
        directions = []
        
        for gap in self.identify_gaps():
            directions.append(f"Future work should explore {gap}")
            
        # Add directions based on themes
        if 'reinforcement learning' in self.paper_memory['key_themes']:
            directions.append("Explore multi-agent reinforcement learning applications")
            
        return directions
    
    def format_citations(self) -> List[str]:
        """Format citations in standard format"""
        citations = []
        
        for paper in self.paper_memory['cited_papers']:
            authors = paper.get('authors', ['Unknown'])
            if isinstance(authors, list) and len(authors) > 0:
                author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
            else:
                author_str = "Unknown"
                
            year = paper.get('year', 'n.d.')
            title = paper.get('title', 'Untitled')
            
            citation = f"{author_str} ({year}). {title}."
            citations.append(citation)
            
        return citations
    
    def generate_literature_review(self) -> Dict[str, Any]:
        """Generate structured literature review from accumulated knowledge"""
        review = {
            'introduction': self.synthesize_introduction(),
            'themes': self.organize_by_themes(),
            'chronological_development': self.trace_development(),
            'research_gaps': self.identify_gaps(),
            'future_directions': self.suggest_directions(),
            'references': self.format_citations()
        }
        return review
    
    def get_workload(self) -> float:
        """Get current workload (for orchestrator)"""
        return len(self.paper_memory['cited_papers']) / 50.0
    
    def get_performance_score(self) -> float:
        """Get performance score (for orchestrator)"""
        if not self.paper_memory['cited_papers']:
            return 0.0
        
        # Score based on diversity of themes and quality of papers
        theme_diversity = len(self.paper_memory['key_themes']) / 6.0
        avg_citations = np.mean([p.get('citationCount', 0) for p in self.paper_memory['cited_papers']])
        citation_score = min(avg_citations / 100, 1.0)
        
        return (theme_diversity + citation_score) / 2.0
    
    def get_coordination_score(self) -> float:
        """Get coordination score (for orchestrator)"""
        # Simple score based on completeness
        if len(self.paper_memory['cited_papers']) >= 20:
            return 1.0
        else:
            return len(self.paper_memory['cited_papers']) / 20.0
    
    def assign_task(self, task):
        """Assign a task to the agent"""
        # Handle task assignment from orchestrator
        if hasattr(task, 'parameters'):
            query = task.parameters.get('query', '')
            max_papers = task.parameters.get('max_papers', 50)
            
            # Search for papers
            papers = self.arxiv_tool.search_papers(query, max_papers)
            
            # Process each paper
            for paper in papers:
                state = self.encode_state(query, [paper], self.format_citations())
                action = self.act(state)
                reward, info = self.execute_action(action, paper)
                
                # Store experience
                next_state = self.encode_state(query, papers[1:], self.format_citations())
                self.remember(state, action, reward, next_state, False)
                
                # Learn periodically
                if len(self.memory) > self.config.get('batch_size', 32):
                    self.learn()