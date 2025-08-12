# src/api/service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import sys
import yaml
import json
import time
from pathlib import Path
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.orchestrator.orchestrator import ResearchPaperOrchestrator
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Research Paper Generator",
    description="AI system that generates research papers using multi-agent reinforcement learning",
    version="1.0.0"
)

# Global variables for the system
orchestrator = None
evaluator = None
config = None

# Request/Response Models
class PaperRequest(BaseModel):
    title: str
    research_question: str
    paper_type: str = "research"  # research, survey, tutorial, position
    venue: str = "Conference"     # Conference, Journal, arXiv, Workshop
    max_pages: int = 10
    difficulty: Optional[str] = "medium"  # easy, medium, hard

class PaperResponse(BaseModel):
    success: bool
    paper: Dict[str, Any] = None
    metrics: Dict[str, Any] = None
    generation_time: float = 0.0
    error_message: str = None

class EvaluationRequest(BaseModel):
    paper: Dict[str, Any]

class EvaluationResponse(BaseModel):
    success: bool
    quality_score: float = 0.0
    quality_grade: str = "N/A"
    detailed_scores: Dict[str, float] = {}
    analysis: Dict[str, Any] = {}
    suggestions: List[str] = []
    error_message: str = None

class SystemStatusResponse(BaseModel):
    system_loaded: bool
    agents_status: Dict[str, str]
    last_training_info: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global orchestrator, evaluator, config
    
    try:
        print("🚀 Initializing Multi-Agent Research Paper Generation System...")
        
        # Load configuration
        config_path = project_root / "configs" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize orchestrator
        orchestrator = ResearchPaperOrchestrator(config)
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator()
        
        print("✅ System initialized successfully!")
        
        # Try to load trained models if available
        try:
            latest_results = max(Path("experiments/results").glob("*"), key=lambda p: p.stat().st_mtime)
            checkpoint_dir = latest_results / "checkpoints"
            if checkpoint_dir.exists():
                latest_checkpoint = max(checkpoint_dir.glob("checkpoint_episode_*.pth"), 
                                      key=lambda p: int(p.stem.split('_')[-1]))
                print(f"📂 Found trained model: {latest_checkpoint}")
                # In a full implementation, you would load the checkpoint here
                print("⚠️  Model loading not implemented - using default initialized model")
        except:
            print("ℹ️  No trained models found - using freshly initialized system")
            
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        raise e

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Multi-Agent Research Paper Generation API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Generate research papers using coordinated AI agents with reinforcement learning",
        "endpoints": {
            "/generate": "Generate a research paper",
            "/evaluate": "Evaluate paper quality",
            "/status": "Get system status",
            "/demo": "Get demo examples"
        }
    }

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status"""
    global orchestrator, config
    
    if not orchestrator:
        return SystemStatusResponse(
            system_loaded=False,
            agents_status={"error": "System not initialized"}
        )
    
    # Check agent status
    agents_status = {}
    for agent_name, agent in orchestrator.agents.items():
        if agent is not None:
            agents_status[agent_name] = "loaded"
        else:
            agents_status[agent_name] = "not_loaded"
    
    # Get training information if available
    training_info = {}
    try:
        results_dirs = list(Path("experiments/results").glob("*"))
        if results_dirs:
            latest_results = max(results_dirs, key=lambda p: p.stat().st_mtime)
            metrics_file = latest_results / "comprehensive_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                training_info = {
                    "total_episodes": len(metrics.get('episode_rewards', [])),
                    "final_quality": metrics.get('paper_quality_scores', [0])[-1] if metrics.get('paper_quality_scores') else 0,
                    "final_coordination": metrics.get('agent_coordination_scores', [0])[-1] if metrics.get('agent_coordination_scores') else 0,
                    "training_date": latest_results.name
                }
    except:
        training_info = {"error": "No training data available"}
    
    return SystemStatusResponse(
        system_loaded=True,
        agents_status=agents_status,
        last_training_info=training_info
    )

@app.post("/generate", response_model=PaperResponse)
async def generate_paper(request: PaperRequest):
    """Generate a research paper using the multi-agent system"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        print(f"📝 Generating paper: {request.title}")
        start_time = time.time()
        
        # Prepare requirements for orchestrator
        requirements = {
            'title': request.title,
            'research_question': request.research_question,
            'paper_type': request.paper_type,
            'venue': request.venue,
            'max_pages': request.max_pages,
            'difficulty': request.difficulty or 'medium'
        }
        
        # Generate paper using orchestrator
        result = orchestrator.orchestrate_paper_generation(requirements)
        generation_time = time.time() - start_time
        
        # Prepare response
        return PaperResponse(
            success=True,
            paper=result['paper'],
            metrics={
                **result['metrics'],
                'generation_time': generation_time,
                'steps_taken': result.get('steps', 0),
                'total_reward': result.get('total_reward', 0)
            },
            generation_time=generation_time
        )
        
    except Exception as e:
        print(f"❌ Error generating paper: {e}")
        return PaperResponse(
            success=False,
            error_message=str(e)
        )

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_paper(request: EvaluationRequest):
    """Evaluate the quality of a generated paper"""
    global evaluator
    
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    try:
        print(f"🔍 Evaluating paper: {request.paper.get('title', 'Unknown')}")
        
        # Evaluate paper quality
        quality_eval = evaluator.evaluate_paper_quality(request.paper)
        
        return EvaluationResponse(
            success=True,
            quality_score=quality_eval['overall_score'],
            quality_grade=quality_eval.get('quality_grade', 'N/A'),
            detailed_scores=quality_eval['aspect_scores'],
            analysis=quality_eval['detailed_evaluations'],
            suggestions=quality_eval.get('improvement_suggestions', [])
        )
        
    except Exception as e:
        print(f"❌ Error evaluating paper: {e}")
        return EvaluationResponse(
            success=False,
            error_message=str(e)
        )

@app.get("/demo", response_model=dict)
async def get_demo_examples():
    """Get demo examples for testing the system"""
    return {
        "demo_requests": [
            {
                "title": "Deep Learning for Natural Language Processing",
                "research_question": "How can transformer architectures improve NLP tasks?",
                "paper_type": "survey",
                "venue": "Conference",
                "max_pages": 8,
                "difficulty": "medium"
            },
            {
                "title": "Reinforcement Learning in Robotics",
                "research_question": "How can RL improve robotic control systems?",
                "paper_type": "research", 
                "venue": "Journal",
                "max_pages": 12,
                "difficulty": "hard"
            },
            {
                "title": "Introduction to Machine Learning",
                "research_question": "What are the fundamental concepts of ML?",
                "paper_type": "tutorial",
                "venue": "Workshop", 
                "max_pages": 6,
                "difficulty": "easy"
            }
        ],
        "paper_types": ["research", "survey", "tutorial", "position"],
        "venues": ["Conference", "Journal", "arXiv", "Workshop"],
        "difficulties": ["easy", "medium", "hard"]
    }

@app.get("/training-progress", response_model=dict)
async def get_training_progress():
    """Get training progress and metrics"""
    try:
        # Find latest training results
        results_dirs = list(Path("experiments/results").glob("*"))
        if not results_dirs:
            return {"error": "No training results found"}
        
        latest_results = max(results_dirs, key=lambda p: p.stat().st_mtime)
        metrics_file = latest_results / "comprehensive_metrics.json"
        
        if not metrics_file.exists():
            return {"error": "Training metrics not found"}
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Calculate summary statistics
        return {
            "total_episodes": len(metrics.get('episode_rewards', [])),
            "final_performance": {
                "advanced_reward": metrics.get('advanced_rewards', [0])[-1] if metrics.get('advanced_rewards') else 0,
                "paper_quality": metrics.get('paper_quality_scores', [0])[-1] if metrics.get('paper_quality_scores') else 0,
                "coordination_score": metrics.get('agent_coordination_scores', [0])[-1] if metrics.get('agent_coordination_scores') else 0,
                "sections_completed": metrics.get('sections_completed', [0])[-1] if metrics.get('sections_completed') else 0
            },
            "learning_curves": {
                "episodes": list(range(1, len(metrics.get('episode_rewards', [])) + 1)),
                "advanced_rewards": metrics.get('advanced_rewards', []),
                "quality_scores": metrics.get('paper_quality_scores', []),
                "coordination_scores": metrics.get('agent_coordination_scores', []),
                "sections_completed": metrics.get('sections_completed', [])
            },
            "curriculum_progression": metrics.get('curriculum_progression', []),
            "training_directory": str(latest_results)
        }
        
    except Exception as e:
        return {"error": f"Failed to load training progress: {str(e)}"}

@app.get("/best-paper", response_model=dict)
async def get_best_paper():
    """Get the best generated paper from training"""
    try:
        # Find latest training results
        results_dirs = list(Path("experiments/results").glob("*"))
        if not results_dirs:
            return {"error": "No training results found"}
        
        latest_results = max(results_dirs, key=lambda p: p.stat().st_mtime)
        best_paper_file = latest_results / "best_paper.json"
        
        if not best_paper_file.exists():
            return {"error": "Best paper not found"}
        
        with open(best_paper_file, 'r', encoding='utf-8') as f:
            best_paper_data = json.load(f)
        
        return {
            "success": True,
            "best_paper": best_paper_data
        }
        
    except Exception as e:
        return {"error": f"Failed to load best paper: {str(e)}"}

if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Research Paper Generation API")
    print("📋 Available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "src.api.service:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )