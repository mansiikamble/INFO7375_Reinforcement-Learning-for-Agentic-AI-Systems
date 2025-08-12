# src/demo/streamlit_demo.py
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="Multi-Agent Research Paper Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_status():
    """Get system status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_training_progress():
    """Get training progress from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/training-progress")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def generate_paper_api(request_data):
    """Generate paper using API"""
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=request_data, timeout=300)
        return response.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error_message": "Request timed out (>5 minutes)"}
    except Exception as e:
        return {"success": False, "error_message": str(e)}

def evaluate_paper_api(paper_data):
    """Evaluate paper using API"""
    try:
        response = requests.post(f"{API_BASE_URL}/evaluate", json={"paper": paper_data})
        return response.json()
    except Exception as e:
        return {"success": False, "error_message": str(e)}

def get_demo_examples():
    """Get demo examples from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/demo")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    """Main Streamlit application"""
    
    # Title and Introduction
    st.markdown('<h1 class="main-header">🤖 Multi-Agent Research Paper Generator</h1>', unsafe_allow_html=True)
    st.markdown("### Reinforcement Learning for Agentic AI Systems - Final Assignment Demo")
    
    # Check API connection
    api_connected = check_api_connection()
    
    if not api_connected:
        st.markdown('<div class="error-message">❌ API Server not running. Please start the API server first:<br><code>python src/api/service.py</code></div>', unsafe_allow_html=True)
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home & System Status",
        "📝 Paper Generation Demo", 
        "📊 Training Analysis",
        "🔍 Paper Evaluation",
        "📈 Before/After Comparison",
        "⚙️ System Architecture"
    ])
    
    if page == "🏠 Home & System Status":
        show_home_page()
    elif page == "📝 Paper Generation Demo":
        show_paper_generation_demo()
    elif page == "📊 Training Analysis":
        show_training_analysis()
    elif page == "🔍 Paper Evaluation":
        show_paper_evaluation()
    elif page == "📈 Before/After Comparison":
        show_before_after_comparison()
    elif page == "⚙️ System Architecture":
        show_system_architecture()

def show_home_page():
    """Show home page with system status"""
    st.header("🏠 System Overview")
    
    # Get system status
    status = get_system_status()
    
    if status and status.get('system_loaded'):
        st.markdown('<div class="success-message">✅ Multi-Agent System is operational and ready!</div>', unsafe_allow_html=True)
        
        # System Status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Agent Status")
            agents_status = status.get('agents_status', {})
            for agent_name, agent_status in agents_status.items():
                status_icon = "✅" if agent_status == "loaded" else "❌"
                st.write(f"{status_icon} **{agent_name.title()} Agent**: {agent_status}")
        
        with col2:
            st.subheader("📊 Training Information")
            training_info = status.get('last_training_info', {})
            if 'error' not in training_info:
                st.metric("Total Episodes Trained", training_info.get('total_episodes', 0))
                st.metric("Final Quality Score", f"{training_info.get('final_quality', 0):.3f}")
                st.metric("Final Coordination", f"{training_info.get('final_coordination', 0):.3f}")
                st.write(f"**Training Date:** {training_info.get('training_date', 'Unknown')}")
            else:
                st.write("ℹ️ No training data available")
    
    else:
        st.markdown('<div class="error-message">❌ System not properly initialized</div>', unsafe_allow_html=True)
    
    # Assignment Information
    st.header("📚 Assignment Demonstration")
    st.write("""
    This demo showcases a **Multi-Agent Reinforcement Learning system** for collaborative research paper generation, 
    fulfilling the requirements for the Reinforcement Learning for Agentic AI Systems final assignment.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔬 RL Implementation:**
        - DQN for Literature & Methodology agents
        - PPO for Writing & Orchestrator agents
        - Multi-agent coordination learning
        """)
    
    with col2:
        st.markdown("""
        **🤝 Agentic System:**
        - Agent Orchestration System
        - Specialized task allocation
        - Dynamic coordination strategies
        """)
    
    with col3:
        st.markdown("""
        **📈 Key Features:**
        - Real-time paper generation
        - Quality evaluation system
        - Training progress visualization
        """)

def show_paper_generation_demo():
    """Show paper generation demo page"""
    st.header("📝 Paper Generation Demo")
    st.write("Generate research papers using our trained multi-agent system!")
    
    # Get demo examples
    demo_data = get_demo_examples()
    
    # Paper generation form
    with st.form("paper_generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Paper Title", 
                                value="Multi-Agent Reinforcement Learning Systems",
                                help="Enter the main topic of your research paper")
            
            research_question = st.text_area("Research Question", 
                                           value="How can multi-agent RL improve coordination in complex systems?",
                                           help="What specific question should the paper address?")
            
            paper_type = st.selectbox("Paper Type", 
                                    options=["research", "survey", "tutorial", "position"],
                                    help="Type of academic paper to generate")
        
        with col2:
            venue = st.selectbox("Target Venue", 
                               options=["Conference", "Journal", "arXiv", "Workshop"],
                               help="Publication venue affects writing style")
            
            max_pages = st.slider("Maximum Pages", min_value=4, max_value=15, value=8,
                                help="Approximate length of the paper")
            
            difficulty = st.selectbox("Difficulty Level", 
                                    options=["easy", "medium", "hard"],
                                    index=1,
                                    help="Complexity level affects content depth")
        
        # Demo examples
        st.subheader("📋 Quick Demo Examples")
        if demo_data and 'demo_requests' in demo_data:
            selected_demo = st.selectbox("Or choose a demo example:", 
                                       options=["Custom"] + [f"{ex['title']}" for ex in demo_data['demo_requests']])
            
            if selected_demo != "Custom":
                demo_example = next(ex for ex in demo_data['demo_requests'] if ex['title'] == selected_demo)
                st.info(f"**Example:** {demo_example['research_question']}")
        
        submitted = st.form_submit_button("🚀 Generate Paper", type="primary")
    
    if submitted:
        # Prepare request
        if selected_demo != "Custom" and demo_data:
            demo_example = next(ex for ex in demo_data['demo_requests'] if ex['title'] == selected_demo)
            request_data = demo_example
        else:
            request_data = {
                "title": title,
                "research_question": research_question,
                "paper_type": paper_type,
                "venue": venue,
                "max_pages": max_pages,
                "difficulty": difficulty
            }
        
        # Show generation progress
        with st.spinner(f"🔄 Generating paper: {request_data['title']}..."):
            start_time = time.time()
            result = generate_paper_api(request_data)
            generation_time = time.time() - start_time
        
        if result.get('success'):
            st.markdown('<div class="success-message">✅ Paper generated successfully!</div>', unsafe_allow_html=True)
            
            # Display metrics
            metrics = result.get('metrics', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Generation Time", f"{generation_time:.2f}s")
            with col2:
                st.metric("Quality Score", f"{metrics.get('paper_quality', 0):.3f}")
            with col3:
                st.metric("Sections Completed", f"{metrics.get('sections_completed', 0):.1f}/6")
            with col4:
                st.metric("Coordination Score", f"{metrics.get('agent_coordination_score', 0):.3f}")
            
            # Display paper content
            paper = result.get('paper', {})
            
            st.subheader("📄 Generated Paper")
            
            # Title and Abstract
            st.markdown(f"### {paper.get('title', 'Untitled')}")
            st.markdown("**Abstract:**")
            st.write(paper.get('abstract', 'No abstract available'))
            
            # Sections
            sections = paper.get('sections', {})
            if sections:
                st.markdown("**Sections:**")
                for section_name, content in sections.items():
                    with st.expander(f"📖 {section_name.replace('_', ' ').title()}"):
                        st.write(str(content))
            
            # References
            references = paper.get('references', [])
            if references:
                st.markdown("**References:**")
                with st.expander(f"📚 References ({len(references)} citations)"):
                    for i, ref in enumerate(references, 1):
                        st.write(f"{i}. {ref}")
            
            # Store generated paper in session state for evaluation
            st.session_state['last_generated_paper'] = paper
            
        else:
            error_msg = result.get('error_message', 'Unknown error occurred')
            st.markdown(f'<div class="error-message">❌ Generation failed: {error_msg}</div>', unsafe_allow_html=True)

def show_training_analysis():
    """Show training analysis and learning curves"""
    st.header("📊 Training Analysis")
    st.write("Analysis of the multi-agent reinforcement learning training process")
    
    # Get training progress
    progress_data = get_training_progress()
    
    if progress_data and 'error' not in progress_data:
        total_episodes = progress_data.get('total_episodes', 0)
        final_performance = progress_data.get('final_performance', {})
        learning_curves = progress_data.get('learning_curves', {})
        curriculum_progression = progress_data.get('curriculum_progression', [])
        
        # Training Summary
        st.subheader("🎯 Training Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Episodes", total_episodes)
        with col2:
            st.metric("Final Quality", f"{final_performance.get('paper_quality', 0):.3f}")
        with col3:
            st.metric("Final Coordination", f"{final_performance.get('coordination_score', 0):.3f}")
        with col4:
            st.metric("Sections Completed", f"{final_performance.get('sections_completed', 0):.1f}/6")
        
        # Learning Curves
        if learning_curves.get('episodes'):
            st.subheader("📈 Learning Curves")
            
            # Create comprehensive learning curves plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Advanced Rewards', 'Paper Quality', 'Agent Coordination', 'Sections Completed'],
                vertical_spacing=0.12
            )
            
            episodes = learning_curves['episodes']
            
            # Advanced rewards
            fig.add_trace(
                go.Scatter(x=episodes, y=learning_curves.get('advanced_rewards', []), 
                          name='Advanced Reward', line=dict(color='blue', width=3)),
                row=1, col=1
            )
            
            # Paper quality
            fig.add_trace(
                go.Scatter(x=episodes, y=learning_curves.get('quality_scores', []), 
                          name='Quality Score', line=dict(color='red', width=3)),
                row=1, col=2
            )
            
            # Coordination scores
            fig.add_trace(
                go.Scatter(x=episodes, y=learning_curves.get('coordination_scores', []), 
                          name='Coordination', line=dict(color='green', width=3)),
                row=2, col=1
            )
            
            # Sections completed
            fig.add_trace(
                go.Scatter(x=episodes, y=learning_curves.get('sections_completed', []), 
                          name='Sections', line=dict(color='purple', width=3)),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Multi-Agent Learning Progress")
            st.plotly_chart(fig, use_container_width=True)
        
        # Curriculum Progression
        if curriculum_progression:
            st.subheader("🎓 Curriculum Learning Progression")
            
            # Create curriculum progression chart
            df_curriculum = pd.DataFrame(curriculum_progression)
            
            fig_curriculum = px.scatter(df_curriculum, x='episode', y='difficulty', 
                                      hover_data=['topic'], 
                                      title="Curriculum Difficulty Progression",
                                      category_orders={"difficulty": ["easy", "medium", "hard"]})
            
            fig_curriculum.update_traces(marker=dict(size=10))
            st.plotly_chart(fig_curriculum, use_container_width=True)
            
            # Difficulty distribution
            difficulty_counts = df_curriculum['difficulty'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Difficulty Distribution")
                fig_pie = px.pie(values=difficulty_counts.values, names=difficulty_counts.index, 
                               title="Episodes by Difficulty Level")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("📚 Topics Covered")
                unique_topics = len(df_curriculum['topic'].unique())
                st.metric("Unique Topics", unique_topics)
                st.write("**Sample Topics:**")
                for topic in df_curriculum['topic'].unique()[:5]:
                    st.write(f"• {topic}")
                if unique_topics > 5:
                    st.write(f"... and {unique_topics - 5} more")
        
        # Performance Analysis
        st.subheader("🔍 Performance Analysis")
        
        if len(learning_curves.get('advanced_rewards', [])) > 1:
            initial_reward = learning_curves['advanced_rewards'][0]
            final_reward = learning_curves['advanced_rewards'][-1]
            improvement = final_reward - initial_reward
            improvement_pct = (improvement / max(abs(initial_reward), 0.001)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reward Improvement", f"{improvement:+.4f}", f"{improvement_pct:+.1f}%")
            
            if len(learning_curves.get('quality_scores', [])) > 1:
                with col2:
                    initial_quality = learning_curves['quality_scores'][0]
                    final_quality = learning_curves['quality_scores'][-1]
                    quality_improvement = final_quality - initial_quality
                    quality_pct = (quality_improvement / max(initial_quality, 0.001)) * 100
                    st.metric("Quality Improvement", f"{quality_improvement:+.3f}", f"{quality_pct:+.1f}%")
            
            with col3:
                avg_sections = sum(learning_curves.get('sections_completed', [])) / len(learning_curves.get('sections_completed', [1]))
                st.metric("Avg Sections Completed", f"{avg_sections:.1f}/6", f"{(avg_sections/6)*100:.1f}%")
    
    else:
        st.markdown('<div class="error-message">❌ No training data available</div>', unsafe_allow_html=True)

def show_paper_generation_demo():
    """Show interactive paper generation demo"""
    st.header("📝 Interactive Paper Generation")
    st.write("Experience the multi-agent system generating research papers in real-time!")
    
    # Quick examples
    demo_data = get_demo_examples()
    
    if demo_data and 'demo_requests' in demo_data:
        st.subheader("⚡ Quick Demo Examples")
        
        examples = demo_data['demo_requests']
        example_cols = st.columns(len(examples))
        
        for i, example in enumerate(examples):
            with example_cols[i]:
                if st.button(f"📄 {example['paper_type'].title()}: {example['title'][:30]}...", key=f"demo_{i}"):
                    st.session_state['demo_request'] = example
                    st.rerun()
    
    # Generation interface
    if 'demo_request' in st.session_state:
        demo_req = st.session_state['demo_request']
        st.success(f"Selected: {demo_req['title']}")
        
        if st.button("🚀 Generate This Paper", type="primary"):
            with st.spinner(f"🔄 Multi-agent system working on: {demo_req['title']}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress updates
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("📚 Literature Review Agent searching papers...")
                    elif i < 40:
                        status_text.text("🔬 Methodology Agent designing approach...")
                    elif i < 70:
                        status_text.text("✍️ Writing Agent crafting content...")
                    else:
                        status_text.text("🎯 Orchestrator coordinating final output...")
                    time.sleep(0.02)  # Small delay for visual effect
                
                result = generate_paper_api(demo_req)
                progress_bar.empty()
                status_text.empty()
            
            if result.get('success'):
                st.balloons()  # Celebration animation
                st.markdown('<div class="success-message">🎉 Paper generated successfully by multi-agent coordination!</div>', unsafe_allow_html=True)
                
                # Show detailed results
                show_paper_results(result)
                
            else:
                error_msg = result.get('error_message', 'Generation failed')
                st.markdown(f'<div class="error-message">❌ {error_msg}</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 Click on one of the demo examples above to get started!")

def show_paper_results(result):
    """Display paper generation results"""
    paper = result.get('paper', {})
    metrics = result.get('metrics', {})
    
    # Performance metrics
    st.subheader("📊 Generation Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Generation Time", f"{metrics.get('generation_time', 0):.2f}s")
    with col2:
        st.metric("Quality Score", f"{metrics.get('paper_quality', 0):.3f}")
    with col3:
        st.metric("Coordination", f"{metrics.get('agent_coordination_score', 0):.3f}")
    with col4:
        st.metric("Sections", f"{metrics.get('sections_completed', 0):.1f}/6")
    with col5:
        st.metric("Steps Taken", metrics.get('steps_taken', 0))
    
    # Agent contributions
    if 'agent_metrics' in metrics:
        st.subheader("🤖 Agent Contributions")
        agent_metrics = metrics['agent_metrics']
        
        agent_data = []
        for agent_name, data in agent_metrics.items():
            agent_data.append({
                'Agent': agent_name.title(),
                'Tasks Completed': data.get('tasks_completed', 0),
                'Average Quality': f"{data.get('average_quality', 0):.3f}",
                'Quality Trend': f"{data.get('quality_trend', 0):+.3f}"
            })
        
        if agent_data:
            df_agents = pd.DataFrame(agent_data)
            st.dataframe(df_agents, use_container_width=True)
    
    # Paper content display
    st.subheader("📄 Generated Paper Content")
    
    # Title and Abstract
    st.markdown(f"### {paper.get('title', 'Untitled')}")
    
    with st.expander("📝 Abstract", expanded=True):
        st.write(paper.get('abstract', 'No abstract available'))
    
    # Sections
    sections = paper.get('sections', {})
    for section_name, content in sections.items():
        with st.expander(f"📖 {section_name.replace('_', ' ').title()}"):
            st.write(str(content))
    
    # References
    references = paper.get('references', [])
    if references:
        with st.expander(f"📚 References ({len(references)} citations)"):
            for i, ref in enumerate(references, 1):
                st.write(f"{i}. {ref}")
    
    # Quality breakdown
    if 'quality_metrics' in paper:
        st.subheader("🔍 Quality Analysis")
        quality_metrics = paper['quality_metrics']
        
        quality_df = pd.DataFrame([
            {'Aspect': 'Overall Coherence', 'Score': quality_metrics.get('overall_coherence', 0)},
            {'Aspect': 'Citation Quality', 'Score': quality_metrics.get('citation_quality', 0)},
            {'Aspect': 'Methodology Soundness', 'Score': quality_metrics.get('methodology_soundness', 0)},
            {'Aspect': 'Writing Quality', 'Score': quality_metrics.get('writing_quality', 0)},
        ])
        
        fig_quality = px.bar(quality_df, x='Aspect', y='Score', 
                           title="Quality Metrics Breakdown",
                           color='Score', color_continuous_scale='Viridis')
        fig_quality.update_layout(height=400)
        st.plotly_chart(fig_quality, use_container_width=True)

def show_training_analysis():
    """Show detailed training analysis"""
    st.header("📈 Training Analysis")
    st.write("Comprehensive analysis of the reinforcement learning training process")
    
    progress_data = get_training_progress()
    
    if progress_data and 'error' not in progress_data:
        learning_curves = progress_data.get('learning_curves', {})
        episodes = learning_curves.get('episodes', [])
        
        if episodes:
            # Detailed learning curves
            st.subheader("📊 Detailed Learning Curves")
            
            metrics_to_plot = {
                'Advanced Rewards': learning_curves.get('advanced_rewards', []),
                'Paper Quality': learning_curves.get('quality_scores', []),
                'Agent Coordination': learning_curves.get('coordination_scores', []),
                'Sections Completed': learning_curves.get('sections_completed', [])
            }
            
            selected_metrics = st.multiselect(
                "Select metrics to display:",
                options=list(metrics_to_plot.keys()),
                default=list(metrics_to_plot.keys())
            )
            
            if selected_metrics:
                fig = go.Figure()
                
                colors = ['blue', 'red', 'green', 'purple']
                for i, metric in enumerate(selected_metrics):
                    values = metrics_to_plot[metric]
                    if values:
                        fig.add_trace(go.Scatter(
                            x=episodes, y=values,
                            name=metric, 
                            line=dict(color=colors[i % len(colors)], width=3)
                        ))
                
                fig.update_layout(
                    title="Learning Progress Over Time",
                    xaxis_title="Episode",
                    yaxis_title="Score",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Training statistics
        st.subheader("📋 Training Statistics")
        
        if learning_curves.get('advanced_rewards'):
            rewards = learning_curves['advanced_rewards']
            quality_scores = learning_curves.get('quality_scores', [])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Learning Progress:**")
                initial_reward = rewards[0] if rewards else 0
                final_reward = rewards[-1] if rewards else 0
                improvement = final_reward - initial_reward
                st.write(f"• Initial Reward: {initial_reward:.4f}")
                st.write(f"• Final Reward: {final_reward:.4f}")
                st.write(f"• Total Improvement: {improvement:+.4f}")
                st.write(f"• Relative Improvement: {(improvement/max(abs(initial_reward), 0.001)*100):+.1f}%")
            
            with col2:
                st.markdown("**Quality Analysis:**")
                if quality_scores:
                    initial_quality = quality_scores[0]
                    final_quality = quality_scores[-1]
                    best_quality = max(quality_scores)
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    
                    st.write(f"• Initial Quality: {initial_quality:.3f}")
                    st.write(f"• Final Quality: {final_quality:.3f}")
                    st.write(f"• Best Quality: {best_quality:.3f}")
                    st.write(f"• Average Quality: {avg_quality:.3f}")
        
        # Curriculum analysis
        curriculum_progression = progress_data.get('curriculum_progression', [])
        if curriculum_progression:
            st.subheader("🎓 Curriculum Learning Analysis")
            
            df_curriculum = pd.DataFrame(curriculum_progression)
            difficulty_counts = df_curriculum['difficulty'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(values=difficulty_counts.values, names=difficulty_counts.index,
                               title="Training Episodes by Difficulty")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("**Curriculum Statistics:**")
                total_episodes = len(curriculum_progression)
                for difficulty, count in difficulty_counts.items():
                    percentage = (count / total_episodes) * 100
                    st.write(f"• **{difficulty.title()}**: {count} episodes ({percentage:.1f}%)")
    
    else:
        st.markdown('<div class="error-message">❌ No training data available. Please run training first.</div>', unsafe_allow_html=True)

def show_paper_evaluation():
    """Show paper evaluation interface"""
    st.header("🔍 Paper Quality Evaluation")
    st.write("Evaluate the quality of generated papers using our comprehensive assessment framework")
    
    # Check if there's a previously generated paper
    if 'last_generated_paper' in st.session_state:
        st.success("📄 Using last generated paper for evaluation")
        paper_to_evaluate = st.session_state['last_generated_paper']
        
        if st.button("🔍 Evaluate This Paper", type="primary"):
            with st.spinner("🔄 Evaluating paper quality..."):
                evaluation_result = evaluate_paper_api(paper_to_evaluate)
            
            if evaluation_result.get('success'):
                st.markdown('<div class="success-message">✅ Evaluation completed!</div>', unsafe_allow_html=True)
                show_evaluation_results(evaluation_result)
            else:
                error_msg = evaluation_result.get('error_message', 'Evaluation failed')
                st.markdown(f'<div class="error-message">❌ {error_msg}</div>', unsafe_allow_html=True)
    
    else:
        st.info("💡 Generate a paper first (go to Paper Generation Demo) to use the evaluation feature")
    
    # Manual paper input
    st.subheader("📝 Manual Paper Input")
    with st.expander("Evaluate a custom paper"):
        custom_title = st.text_input("Paper Title")
        custom_abstract = st.text_area("Abstract")
        custom_sections = {}
        
        section_names = ['introduction', 'literature_review', 'methodology', 'results', 'discussion', 'conclusion']
        for section in section_names:
            custom_sections[section] = st.text_area(f"{section.replace('_', ' ').title()}")
        
        custom_references = st.text_area("References (one per line)").split('\n') if st.text_area("References (one per line)") else []
        
        if st.button("🔍 Evaluate Custom Paper"):
            custom_paper = {
                'title': custom_title,
                'abstract': custom_abstract,
                'sections': custom_sections,
                'references': custom_references
            }
            
            with st.spinner("🔄 Evaluating custom paper..."):
                evaluation_result = evaluate_paper_api(custom_paper)
            
            if evaluation_result.get('success'):
                st.success("✅ Custom paper evaluated!")
                show_evaluation_results(evaluation_result)

def show_evaluation_results(evaluation_result):
    """Display evaluation results"""
    quality_score = evaluation_result.get('quality_score', 0)
    quality_grade = evaluation_result.get('quality_grade', 'N/A')
    detailed_scores = evaluation_result.get('detailed_scores', {})
    suggestions = evaluation_result.get('suggestions', [])
    
    # Overall score
    st.subheader("🎯 Overall Quality Assessment")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Quality Score", f"{quality_score:.3f}")
    with col2:
        st.metric("Quality Grade", quality_grade)
    with col3:
        # Color-coded grade
        grade_color = "green" if quality_grade.startswith('A') else "orange" if quality_grade.startswith('B') else "red"
        st.markdown(f'<div style="color: {grade_color}; font-size: 1.5rem; font-weight: bold;">Grade: {quality_grade}</div>', unsafe_allow_html=True)
    
    # Detailed scores
    if detailed_scores:
        st.subheader("📊 Detailed Quality Breakdown")
        
        df_scores = pd.DataFrame([
            {'Aspect': aspect.title(), 'Score': score, 'Percentage': f"{score*100:.1f}%"}
            for aspect, score in detailed_scores.items()
        ])
        
        fig_scores = px.bar(df_scores, x='Aspect', y='Score',
                          title="Quality Aspects Analysis",
                          color='Score', color_continuous_scale='RdYlGn',
                          text='Percentage')
        fig_scores.update_traces(textposition='outside')
        fig_scores.update_layout(height=400)
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Show scores in table format too
        st.dataframe(df_scores, use_container_width=True)
    
    # Improvement suggestions
    if suggestions:
        st.subheader("💡 Improvement Suggestions")
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")

def show_before_after_comparison():
    """Show before/after training comparison"""
    st.header("📈 Before/After Training Comparison")
    st.write("Compare the system performance before and after reinforcement learning training")
    
    # Get best paper from training
    try:
        response = requests.get(f"{API_BASE_URL}/best-paper")
        if response.status_code == 200:
            best_paper_data = response.json()
            
            if best_paper_data.get('success'):
                best_paper_info = best_paper_data['best_paper']
                
                st.subheader("🏆 Best Generated Paper from Training")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Episode", best_paper_info.get('episode', 0))
                with col2:
                    st.metric("Advanced Reward", f"{best_paper_info.get('reward', 0):.4f}")
                with col3:
                    st.metric("Quality Score", f"{best_paper_info.get('quality', 0):.3f}")
                
                # Show best paper
                best_paper = best_paper_info.get('paper', {})
                if best_paper:
                    st.markdown(f"**Title:** {best_paper.get('title', 'Unknown')}")
                    
                    with st.expander("📄 Best Paper Content", expanded=True):
                        st.markdown("**Abstract:**")
                        st.write(best_paper.get('abstract', 'No abstract available'))
                        
                        sections = best_paper.get('sections', {})
                        if sections:
                            for section_name, content in sections.items():
                                st.markdown(f"**{section_name.replace('_', ' ').title()}:**")
                                st.write(str(content)[:300] + "..." if len(str(content)) > 300 else str(content))
        
    except Exception as e:
        st.error(f"Could not load best paper: {e}")
    
    # Comparison demo
    st.subheader("🔄 Generate Comparison Demo")
    st.write("Generate a paper to see current system performance:")
    
    if st.button("🎯 Generate Comparison Paper", type="primary"):
        comparison_request = {
            "title": "Multi-Agent Reinforcement Learning for Research Paper Generation",
            "research_question": "How effective is multi-agent coordination in automated scientific writing?",
            "paper_type": "research",
            "venue": "Conference",
            "max_pages": 10,
            "difficulty": "medium"
        }
        
        with st.spinner("🔄 Generating comparison paper..."):
            result = generate_paper_api(comparison_request)
        
        if result.get('success'):
            st.success("✅ Comparison paper generated!")
            
            # Show current performance
            metrics = result.get('metrics', {})
            st.subheader("📊 Current System Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Quality Score", f"{metrics.get('paper_quality', 0):.3f}")
            with col2:
                st.metric("Coordination", f"{metrics.get('agent_coordination_score', 0):.3f}")  
            with col3:
                st.metric("Sections", f"{metrics.get('sections_completed', 0):.1f}/6")
            with col4:
                st.metric("Generation Time", f"{metrics.get('generation_time', 0):.2f}s")
            
            # Compare with training progress
            progress_data = get_training_progress()
            if progress_data and 'error' not in progress_data:
                final_performance = progress_data.get('final_performance', {})
                
                st.subheader("📈 Training Impact Analysis")
                
                # Simulated "before training" performance for comparison
                before_performance = {
                    'quality': 0.45,
                    'coordination': 0.2,
                    'sections': 1.5,
                    'time': 300
                }
                
                after_performance = {
                    'quality': metrics.get('paper_quality', 0),
                    'coordination': metrics.get('agent_coordination_score', 0),
                    'sections': metrics.get('sections_completed', 0),
                    'time': metrics.get('generation_time', 0)
                }
                
                comparison_df = pd.DataFrame([
                    {
                        'Metric': 'Quality Score',
                        'Before Training': before_performance['quality'],
                        'After Training': after_performance['quality'],
                        'Improvement': after_performance['quality'] - before_performance['quality'],
                        'Improvement %': f"{((after_performance['quality'] - before_performance['quality']) / before_performance['quality'] * 100):+.1f}%"
                    },
                    {
                        'Metric': 'Coordination',
                        'Before Training': before_performance['coordination'],
                        'After Training': after_performance['coordination'],
                        'Improvement': after_performance['coordination'] - before_performance['coordination'],
                        'Improvement %': f"{((after_performance['coordination'] - before_performance['coordination']) / max(before_performance['coordination'], 0.001) * 100):+.1f}%"
                    },
                    {
                        'Metric': 'Sections Completed',
                        'Before Training': before_performance['sections'],
                        'After Training': after_performance['sections'],
                        'Improvement': after_performance['sections'] - before_performance['sections'],
                        'Improvement %': f"{((after_performance['sections'] - before_performance['sections']) / before_performance['sections'] * 100):+.1f}%"
                    }
                ])
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Improvement visualization
                fig_comparison = go.Figure()
                
                metrics_names = ['Quality', 'Coordination', 'Sections']
                before_values = [before_performance['quality'], before_performance['coordination'], before_performance['sections']/6]
                after_values = [after_performance['quality'], after_performance['coordination'], after_performance['sections']/6]
                
                fig_comparison.add_trace(go.Bar(name='Before Training', x=metrics_names, y=before_values, marker_color='lightcoral'))
                fig_comparison.add_trace(go.Bar(name='After Training', x=metrics_names, y=after_values, marker_color='lightblue'))
                
                fig_comparison.update_layout(
                    title="Before vs After Training Performance",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

def show_system_architecture():
    """Show system architecture and technical details"""
    st.header("⚙️ System Architecture")
    st.write("Technical details of the multi-agent reinforcement learning system")
    
    # Architecture overview
    st.subheader("🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Specialized Agents:**
        - **Literature Review Agent** (DQN)
          - Searches and analyzes research papers
          - Selects relevant citations
          - Identifies research gaps
        
        - **Methodology Agent** (DQN) 
          - Designs research methodologies
          - Selects appropriate approaches
          - Plans experimental procedures
        """)
    
    with col2:
        st.markdown("""
        - **Writing Agent** (PPO)
          - Generates paper sections
          - Optimizes writing style
          - Ensures academic formatting
        
        - **Orchestrator Agent** (PPO)
          - Coordinates all agents
          - Manages task allocation
          - Optimizes overall workflow
        """)
    
    # Technical details
    st.subheader("🔧 Technical Implementation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🧠 RL Algorithms:**
        - **DQN (Deep Q-Network)**
          - Literature & Methodology agents
          - Discrete action spaces
          - Experience replay learning
        
        - **PPO (Proximal Policy Optimization)**
          - Writing & Orchestrator agents  
          - Continuous action spaces
          - Policy gradient optimization
        """)
    
    with col2:
        st.markdown("""
        **🔄 Coordination System:**
        - **Shared Memory System**
          - Inter-agent communication
          - State synchronization
          - Result sharing
        
        - **Task Scheduling**
          - Dynamic task allocation
          - Dependency management
          - Parallel/sequential execution
        """)
    
    with col3:
        st.markdown("""
        **📊 Evaluation Framework:**
        - **Quality Metrics**
          - Structure completeness
          - Content coherence
          - Citation quality
          - Writing style assessment
        
        - **Learning Metrics**
          - Reward progression
          - Agent coordination
          - Efficiency measures
        """)
    
    # System statistics
    status = get_system_status()
    if status and status.get('system_loaded'):
        st.subheader("📈 System Statistics")
        
        training_info = status.get('last_training_info', {})
        if 'error' not in training_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Training Episodes", training_info.get('total_episodes', 0))
            with col2:
                st.metric("Final Quality", f"{training_info.get('final_quality', 0):.3f}")
            with col3:
                st.metric("Agent Coordination", f"{training_info.get('final_coordination', 0):.3f}")
            with col4:
                training_date = training_info.get('training_date', 'Unknown')
                st.write(f"**Trained:** {training_date}")
    
    # Architecture diagram (text-based)
    st.subheader("📋 Data Flow")
    st.code("""
    Input Request
         ↓
    Orchestrator Agent (PPO)
         ↓
    ┌─────────────┬─────────────┬─────────────┐
    ↓             ↓             ↓             ↓
Literature    Methodology   Analysis     Writing
Agent (DQN)   Agent (DQN)   Agent (DQN)  Agent (PPO)
    ↓             ↓             ↓             ↓
Citations &   Research      Data         Paper
Themes        Methods       Analysis     Sections
         ↓           ↓           ↓         ↓
         └───────────┼───────────┼─────────┘
                     ↓
              Shared Memory System
                     ↓
              Final Paper Assembly
                     ↓
              Quality Evaluation
                     ↓
              Response Output
    """)

if __name__ == "__main__":
    main()