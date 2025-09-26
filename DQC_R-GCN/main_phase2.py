"""
Optimized Phase 2 main training script with performance monitoring.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms.ppo import PPO

# Add this import check
try:
    from training.imitation_learning import OptimizedImitationLearningTrainer
    USE_OPTIMIZED = True
except ImportError:
    print("Warning: Optimized trainer not found, using original version")
    from training.imitation_learning import ImitationLearningTrainer as OptimizedImitationLearningTrainer
    USE_OPTIMIZED = False

# Add performance monitor import check
try:
    from utils.performance_monitor import perf_monitor
    PERF_MONITOR_AVAILABLE = True
except ImportError:
    print("Warning: Performance monitor not available")
    PERF_MONITOR_AVAILABLE = False
    
    # Create a dummy perf_monitor
    class DummyPerfMonitor:
        def timer(self, name):
            from contextlib import nullcontext
            return nullcontext()
        
        def get_stats(self, name):
            return {'last': 0}
        
        def print_summary(self, top_n=10):
            pass
    
    perf_monitor = DummyPerfMonitor()

def main():
    """Optimized Phase 2 main training with performance monitoring."""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("OPTIMIZED PHASE 2: CORE TRAINING WITH IMITATION LEARNING")
    print("="*60)
    
    # Start overall timer
    overall_start = time.time()
    
    # Generate problem with configurable size
    print("\nGenerating quantum problem...")
    
    # Small scale for debugging - adjust these values
    DEBUG_MODE = True  # Set to False for full scale
    
    if DEBUG_MODE:
        print("Running in DEBUG MODE with small problem size")
        problem_config = {
            'num_gates': 10,      # Reduced from 50
            'num_qpus': 2,        # Reduced from 3
            'num_qubits': 8,      # Reduced from 20
            'qpu_capacity': 4,    # Reduced from 10
            'time_horizon': 20,   # Reduced from 50
            'seed': 42
        }
    else:
        problem_config = {
            'num_gates': 50,
            'num_qpus': 3,
            'num_qubits': 20,
            'qpu_capacity': 10,
            'time_horizon': 50,
            'seed': 42
        }
    
    with perf_monitor.timer("problem_generation"):
        generator = QuantumProblemGenerator(**problem_config)
        generator.generate_all_parameters()
        problem_data = generator.get_model_data()
    
    if PERF_MONITOR_AVAILABLE:
        print(f"Problem generation time: {perf_monitor.get_stats('problem_generation')['last']:.2f}s")
    
    # Create environment
    print("Creating environment...")
    env_config = {
        'max_steps': 100 if DEBUG_MODE else 200,
        'w_alloc': 1.0,
        'w_schedule': 1.0,
        'w_progress': 0.01
    }
    
    with perf_monitor.timer("env_creation"):
        env = DQCEnvironment(problem_data, env_config)
    
    # Create networks
    print("Initializing networks...")
    model_config = {
        'hidden_dim': 64 if DEBUG_MODE else 128,  # Smaller network for debugging
        'num_layers': 2 if DEBUG_MODE else 3,
        'num_bases': 4 if DEBUG_MODE else 8,
        'dropout': 0.1,
        'num_heads': 2 if DEBUG_MODE else 4
    }
    
    with perf_monitor.timer("network_initialization"):
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
        'rollout_length': 128 if DEBUG_MODE else 512,
        'minibatch_size': 32 if DEBUG_MODE else 64,
        'num_epochs': 5 if DEBUG_MODE else 10
    }
    
    with perf_monitor.timer("ppo_initialization"):
        ppo = PPO(policy_net, value_net, ppo_config)
    
    # Create optimized imitation learning trainer
    print("Setting up optimized imitation learning trainer...")
    il_config = {
        'bc_epochs': 5 if DEBUG_MODE else 10,
        'bc_batch_size': 32 if DEBUG_MODE else 64,
        'bc_lr': 3e-4,
        'expert_buffer_size': 5000 if DEBUG_MODE else 10000,
        'expert_prob': 0.3,
        'reward_config': {
            'w_alloc': 1.0,
            'w_schedule': 1.0,
            'w_progress': 0.01,
            'w_expert': 0.5,
            'r_expert_match': 1.0,
            'r_completion': 10.0
        },
        'use_wandb': False,
        'enable_profiling': PERF_MONITOR_AVAILABLE  # Enable only if available
    }
    
    with perf_monitor.timer("trainer_initialization"):
        trainer = OptimizedImitationLearningTrainer(env, policy_net, value_net, ppo, il_config)
        
        # Try to use optimized expert policy if available
        try:
            from expert_policy.uniq_expert import OptimizedUNIQExpertPolicy
            trainer.expert = OptimizedUNIQExpertPolicy(env)
            print("Using optimized expert policy")
        except ImportError:
            print("Using standard expert policy")
            # Keep the default expert from trainer initialization
    
    # Train
    print("\nStarting optimized training...")
    total_timesteps = 5000 if DEBUG_MODE else 50000
    
    print(f"Training configuration:")
    print(f"  - Total timesteps: {total_timesteps}")
    print(f"  - Rollout length: {ppo_config['rollout_length']}")
    print(f"  - Number of updates: {total_timesteps // ppo_config['rollout_length']}")
    print(f"  - Debug mode: {DEBUG_MODE}")
    print(f"  - Using optimized trainer: {USE_OPTIMIZED}")
    print(f"  - Performance monitoring: {PERF_MONITOR_AVAILABLE}")
    
    trainer.train(total_timesteps)
    
    # Save model
    print("\nSaving trained model...")
    with perf_monitor.timer("model_saving"):
        torch.save({
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
            'config': {
                'model_config': model_config,
                'ppo_config': ppo_config,
                'il_config': il_config,
                'problem_config': problem_config
            }
        }, 'phase2_model_optimized.pt')
    
    overall_time = time.time() - overall_start
    
    print(f"\nPhase 2 training completed successfully in {overall_time:.2f} seconds!")
    print(f"Model saved to phase2_model_optimized.pt")
    
    # Print detailed performance breakdown if available
    if PERF_MONITOR_AVAILABLE:
        print("\n" + "="*60)
        print("DETAILED PERFORMANCE BREAKDOWN")
        print("="*60)
        perf_monitor.print_summary(top_n=15)

if __name__ == "__main__":
    main()