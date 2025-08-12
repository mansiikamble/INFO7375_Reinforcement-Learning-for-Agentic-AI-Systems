import os

# Project structure
directories = [
    "src",
    "src/agents",
    "src/agents/literature",
    "src/agents/methodology",
    "src/agents/analysis",
    "src/agents/writing",
    "src/rl",
    "src/rl/dqn",
    "src/rl/ppo",
    "src/rl/common",
    "src/tools",
    "src/tools/builtin",
    "src/tools/custom",
    "src/orchestrator",
    "src/utils",
    "data",
    "data/raw",
    "data/raw/papers",
    "data/raw/citations",
    "data/raw/quality_metrics",
    "data/processed",
    "data/models",
    "tests",
    "docs",
    "experiments",
    "experiments/baselines",
    "experiments/ablations",
    "experiments/results",
    "configs",
    "notebooks"
]

# Create all directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created: {directory}")

print("\nProject structure created successfully!")