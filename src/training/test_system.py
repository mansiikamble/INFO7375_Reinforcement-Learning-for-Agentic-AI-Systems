import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.orchestrator.orchestrator import ResearchPaperOrchestrator
import yaml

def test_basic_generation():
    """Test basic paper generation"""
    print("Testing basic paper generation...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = ResearchPaperOrchestrator(config)
    
    # Test requirements
    requirements = {
        'title': 'Test Paper: Multi-Agent Systems',
        'research_question': 'Can agents work together?',
        'paper_type': 'research',
        'venue': 'Conference'
    }
    
    # Generate paper
    print("Generating paper...")
    result = orchestrator.orchestrate_paper_generation(requirements)
    
    print("\nGeneration complete!")
    print(f"Total reward: {result['total_reward']}")
    print(f"Paper quality: {result['metrics']['paper_quality']}")
    print(f"Sections completed: {result['metrics']['sections_completed']}")
    
    # Print paper structure
    print("\nGenerated Paper Structure:")
    paper = result['paper']
    print(f"Title: {paper['title']}")
    print(f"Abstract: {paper['abstract'][:100]}...")
    print(f"Sections: {list(paper['sections'].keys())}")
    print(f"References: {len(paper['references'])} citations")

if __name__ == "__main__":
    test_basic_generation()