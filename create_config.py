import os
import yaml

config = {
    'project': {
        'name': 'Collaborative Research Paper Generator',
        'version': '1.0.0'
    },
    'environment': {
        'device': 'cpu',  # Change to 'cuda' if you have GPU
        'seed': 42,
        'log_level': 'INFO'
    },
    'agents': {
        'literature': {
            'model_type': 'dqn',
            'state_dim': 256,
            'action_dim': 6,
            'learning_rate': 0.001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'batch_size': 32
        },
        'methodology': {
            'model_type': 'dqn',
            'state_dim': 128,
            'action_dim': 8,
            'learning_rate': 0.001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'batch_size': 32
        },
        'writing': {
            'model_type': 'ppo',
            'state_dim': 512,
            'action_dim': 5,
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'n_epochs': 10,
            'clip_range': 0.2,
            'gae_lambda': 0.95
        },
        'orchestrator': {
            'model_type': 'ppo',
            'state_dim': 1024,
            'action_dim': 32,
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'n_epochs': 10,
            'clip_range': 0.2,
            'gae_lambda': 0.95
        }
    },
    'training': {
        'episodes': 1000,
        'gamma': 0.99,
        'save_frequency': 100,
        'eval_frequency': 50
    }
}

# Create config directory if it doesn't exist
os.makedirs('configs', exist_ok=True)

# Save config
with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config file created at configs/config.yaml")