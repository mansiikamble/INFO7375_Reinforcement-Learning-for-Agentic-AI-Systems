# Create init_files.py
import os

# All directories that need __init__.py
python_dirs = [
    'src',
    'src/agents',
    'src/agents/literature',
    'src/agents/methodology',
    'src/agents/analysis',
    'src/agents/writing',
    'src/rl',
    'src/rl/dqn',
    'src/rl/ppo',
    'src/rl/common',
    'src/tools',
    'src/tools/builtin',
    'src/tools/custom',
    'src/orchestrator',
    'src/utils',
    'tests'
]

for dir_path in python_dirs:
    init_file = os.path.join(dir_path, '__init__.py')
    with open(init_file, 'w') as f:
        f.write('# This file makes the directory a Python package\n')
    print(f"Created: {init_file}")

print("\nAll __init__.py files created!")