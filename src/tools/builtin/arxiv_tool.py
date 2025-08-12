import arxiv
from typing import List, Dict, Any

class ArxivTool:
    """Tool for interacting with arXiv API"""
    
    def __init__(self):
        self.client = arxiv.Client()
        
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on arXiv"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            papers.append({
                'paperId': result.entry_id,
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'year': result.published.year if result.published else None,
                'url': result.pdf_url,
                'categories': result.categories
            })
            
        return papers
    
    def search_similar(self, title: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for papers similar to given title"""
        # Use title keywords for similarity search
        keywords = ' '.join(title.split()[:5])  # Use first 5 words
        return self.search_papers(keywords, max_results)