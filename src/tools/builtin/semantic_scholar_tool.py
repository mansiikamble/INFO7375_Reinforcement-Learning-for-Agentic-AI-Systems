import requests
from typing import List, Dict, Any
import time

class SemanticScholarTool:
    """Tool for interacting with Semantic Scholar API"""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {'Accept': 'application/json'}
        
    def get_paper(self, paper_id: str) -> Dict[str, Any]:
        """Get paper details by ID"""
        url = f"{self.base_url}/paper/{paper_id}"
        params = {
            'fields': 'paperId,title,abstract,year,authors,citationCount,influentialCitationCount'
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            print(f"Error fetching paper: {e}")
            return {}
            
    def get_references(self, paper_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get references of a paper"""
        url = f"{self.base_url}/paper/{paper_id}/references"
        params = {
            'fields': 'paperId,title,year,authors',
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return [ref['citedPaper'] for ref in data.get('data', [])]
            else:
                return []
        except Exception as e:
            print(f"Error fetching references: {e}")
            return []
            
    def get_citations(self, paper_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that cite this paper"""
        url = f"{self.base_url}/paper/{paper_id}/citations"
        params = {
            'fields': 'paperId,title,year,authors,citationCount',
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return [cite['citingPaper'] for cite in data.get('data', [])]
            else:
                return []
        except Exception as e:
            print(f"Error fetching citations: {e}")
            return []
        
        # Add small delay to respect rate limits
        time.sleep(0.1)