import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms import PPO

def main():
    """Main training script."""
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate problem instance
    print("Generating quantum problem instance...")
    generator = QuantumProblemGenerator(
        num_gates=50,      # Start small for testing
        num_qpus=3,
        num_qubits=20,
        qpu_capacity=10,
        time_horizon=50,
        seed=42
    )
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    # Create environment
    print("Creating DQC environment...")
    env_config = {
        'max_steps': 200,
        'w_alloc': 1.0,
        'w_schedule': 1.0,
        'w_progress': 0.01
    }
    env = DQCEnvironment(problem_data, env_config)
    
    # Create networks
    print("Initializing neural networks...")
    model_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_bases': 8,
        'dropout': 0.1,
        'num_heads': 4
    }
    
    # Shared R-GCN encoder
    rgcn = RGCN(model_config)
    
    # Policy and value networks
    policy_net = PolicyNetwork(rgcn, model_config)
    value_net = ValueNetwork(rgcn, model_config)
    
    # Create PPO algorithm
    print("Setting up PPO algorithm...")
    ppo_config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'rollout_length': 256,  # Smaller for testing
        'minibatch_size': 32,
        'num_epochs': 5
    }
    ppo = PPO(policy_net, value_net, ppo_config)
    
    # Training
    print("Starting training...")
    total_timesteps = 10000  # Small for testing
    ppo.train(env, total_timesteps)
    
    print("Training completed!")

if __name__ == "__main__":
    main()