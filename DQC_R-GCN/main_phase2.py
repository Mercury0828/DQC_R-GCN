"""
Phase 2 main training script with imitation learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms.ppo import PPO
from training.imitation_learning import ImitationLearningTrainer

def main():
    """Phase 2 main training with imitation learning."""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("PHASE 2: CORE TRAINING WITH IMITATION LEARNING")
    print("="*60)
    
    # Generate problem
    print("\nGenerating quantum problem...")
    generator = QuantumProblemGenerator(
        num_gates=50,
        num_qpus=3,
        num_qubits=20,
        qpu_capacity=10,
        time_horizon=50,
        seed=42
    )
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    # Create environment with enhanced rewards
    print("Creating environment...")
    env_config = {
        'max_steps': 200,
        'w_alloc': 1.0,
        'w_schedule': 1.0,
        'w_progress': 0.01
    }
    env = DQCEnvironment(problem_data, env_config)
    
    # Create networks
    print("Initializing networks...")
    model_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_bases': 8,
        'dropout': 0.1,
        'num_heads': 4
    }
    
    rgcn = RGCN(model_config)
    policy_net = PolicyNetwork(rgcn, model_config)
    value_net = ValueNetwork(rgcn, model_config)
    
    # Create PPO
    ppo_config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'rollout_length': 512,
        'minibatch_size': 64,
        'num_epochs': 10
    }
    ppo = PPO(policy_net, value_net, ppo_config)
    
    # Create imitation learning trainer
    print("Setting up imitation learning trainer...")
    il_config = {
        'bc_epochs': 10,
        'bc_batch_size': 64,
        'bc_lr': 3e-4,
        'expert_buffer_size': 10000,
        'expert_prob': 0.3,
        'reward_config': {
            'w_alloc': 1.0,
            'w_schedule': 1.0,
            'w_progress': 0.01,
            'w_expert': 0.5,
            'r_expert_match': 1.0,
            'r_completion': 10.0
        },
        'use_wandb': False  # Set to True if using wandb
    }
    
    trainer = ImitationLearningTrainer(env, policy_net, value_net, ppo, il_config)
    
    # Train
    print("\nStarting training...")
    total_timesteps = 50000
    trainer.train(total_timesteps)
    
    # Save model
    print("\nSaving trained model...")
    torch.save({
        'policy_state': policy_net.state_dict(),
        'value_state': value_net.state_dict(),
        'config': {
            'model_config': model_config,
            'ppo_config': ppo_config,
            'il_config': il_config
        }
    }, 'phase2_model.pt')
    
    print("Phase 2 training completed successfully!")
    print("Model saved to phase2_model.pt")

if __name__ == "__main__":
    main()