import arxiv
import json
import os
from typing import List, Dict

class PaperDataLoader:
    def __init__(self):
        # Data folder is at root level, need to go up from src/utils/
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_training_papers(self, num_papers: int = 100) -> List[Dict]:
        """Get papers for training"""
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"training_papers_{num_papers}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Fetch from arXiv
        search = arxiv.Search(
            query="cat:cs.AI OR cat:cs.LG",
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in search.results():
            papers.append({
                'id': result.entry_id,
                'title': result.title,
                'abstract': result.summary,
                'categories': result.categories,
                'published': str(result.published)
            })
        
        # Cache for future use
        with open(cache_file, 'w') as f:
            json.dump(papers, f)
        
        return papers