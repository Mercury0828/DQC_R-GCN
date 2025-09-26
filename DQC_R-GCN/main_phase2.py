"""
Improved Phase 2 training script based on debugging findings.
Key improvements:
1. Stronger BC pretraining
2. Higher initial expert probability
3. Better reward shaping
4. Gradual transition from imitation to RL
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
import json
import pickle
from pathlib import Path

from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms.ppo import PPO
from training.imitation_learning import ImitationLearningTrainer
from expert_policy.uniq_expert import UNIQExpertPolicy

def get_training_config(problem_size="small"):
    """Get training configuration."""
    config = {
        'problem': {
            'debug': {
                'num_gates': 10,
                'num_qpus': 2,
                'num_qubits': 8,
                'qpu_capacity': 4,
                'time_horizon': 20
            },
            'small': {
                'num_gates': 20,
                'num_qpus': 3,
                'num_qubits': 12,
                'qpu_capacity': 5,
                'time_horizon': 30
            },
            'medium': {
                'num_gates': 50,
                'num_qpus': 3,
                'num_qubits': 20,
                'qpu_capacity': 10,
                'time_horizon': 50
            }
        },
        'model': {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_bases': 8,
            'dropout': 0.1,
            'num_heads': 4
        },
        'ppo': {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'rollout_length': 512,
            'minibatch_size': 64,
            'num_epochs': 10
        },
        'imitation_learning': {
            'bc_pretrain_epochs': 50,  # Increased
            'bc_pretrain_batch_size': 32,
            'bc_pretrain_lr': 5e-4,  # Increased
            'num_expert_episodes': 30,  # Increased
            'expert_buffer_size': 10000,
            'initial_expert_prob': 0.5,  # Increased
            'final_expert_prob': 0.1,
            'bc_epochs_per_update': 3,
            'bc_update_frequency': 5,
            'reward_weights': {
                'w_alloc': 1.0,
                'w_schedule': 1.0,
                'w_progress': 0.005,  # Reduced
                'w_expert': 1.0,  # Increased
                'r_expert_match': 2.0,  # Increased
                'r_completion': 10.0,
                'r_colocate': 0.5,
                'r_separate': 1.0,
                'r_time': 1.0,
                'r_parallel_epr': 0.5,
                'r_step': 0.005  # Reduced
            }
        },
        'training': {
            'phase1_epochs': 100,
            'phase2_timesteps': 25000,
            'phase2_expert_decay': 'linear',
            'phase3_timesteps': 25000,
            'phase3_expert_prob': 0.05,
            'total_timesteps': 50000,
            'log_frequency': 10,
            'save_frequency': 100
        },
        'paths': {
            'checkpoint_dir': 'checkpoints/',
            'log_dir': 'logs/',
            'expert_data': 'expert_demonstrations.pkl',
            'model_save': 'phase2_model_improved.pt'
        }
    }
    
    # Select problem configuration
    config['selected_problem'] = config['problem'][problem_size]
    return config

def phase1_bc_pretraining(trainer, config):
    """Phase 1: Strong BC pretraining to initialize policy."""
    print("\n" + "="*60)
    print("PHASE 1: BEHAVIORAL CLONING PRETRAINING")
    print("="*60)
    
    il_config = config['imitation_learning']
    
    # Generate expert demonstrations
    print(f"\nGenerating {il_config['num_expert_episodes']} expert demonstrations...")
    trainer.generate_expert_demonstrations(num_episodes=il_config['num_expert_episodes'])
    
    # Save expert demonstrations
    expert_data_path = config['paths']['expert_data']
    Path(expert_data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(expert_data_path, 'wb') as f:
        pickle.dump(list(trainer.expert_buffer), f)
    print(f"Expert demonstrations saved to {expert_data_path}")
    
    # BC pretraining
    print(f"\nBC Pretraining for {il_config['bc_pretrain_epochs']} epochs...")
    
    best_loss = float('inf')
    for epoch in range(il_config['bc_pretrain_epochs']):
        loss = trainer.behavioral_cloning_update()
        
        if loss < best_loss:
            best_loss = loss
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss={loss:.4f} (Best: {best_loss:.4f})")
            
            # Test expert probabilities
            if epoch % 20 == 0:
                test_expert_probabilities(trainer)
    
    print(f"\nPhase 1 complete. Final BC loss: {loss:.4f}")
    return best_loss

def test_expert_probabilities(trainer, num_samples=5):
    """Test current expert action probabilities."""
    probs = []
    
    for _ in range(num_samples):
        trainer.env.reset()
        state_repr = trainer.env.state.get_state_representation()
        valid_actions = trainer.env.get_valid_actions()
        
        if not valid_actions['map'] and not valid_actions['schedule']:
            continue
        
        expert_action = trainer.expert.get_expert_action(state_repr, valid_actions)
        
        with torch.no_grad():
            action_probs, _, action_info = trainer.policy_net.forward(state_repr, valid_actions)
        
        for idx, (act_type, act_params) in enumerate(action_info):
            if act_type == expert_action[0] and act_params == expert_action[1]:
                if isinstance(action_probs, torch.Tensor):
                    if action_probs.dim() == 0:
                        action_probs = action_probs.unsqueeze(0)
                    probs.append(action_probs[idx].item())
                break
    
    if probs:
        print(f"    Average expert action probability: {np.mean(probs):.3f}")

def phase2_mixed_training(trainer, config):
    """Phase 2: Mixed training with gradual transition."""
    print("\n" + "="*60)
    print("PHASE 2: MIXED TRAINING WITH GRADUAL TRANSITION")
    print("="*60)
    
    il_config = config['imitation_learning']
    training_config = config['training']
    
    # Set initial expert probability
    trainer.expert_prob = il_config['initial_expert_prob']
    
    timesteps = training_config['phase2_timesteps']
    num_updates = timesteps // trainer.ppo.rollout_length
    
    print(f"\nTraining for {timesteps} timesteps ({num_updates} updates)")
    print(f"Initial expert probability: {trainer.expert_prob:.2f}")
    
    for update in range(num_updates):
        # Decay expert probability
        if training_config['phase2_expert_decay'] == 'linear':
            progress = update / num_updates
            trainer.expert_prob = il_config['initial_expert_prob'] * (1 - progress * 0.8) + il_config['final_expert_prob']
        else:  # exponential
            decay_rate = 0.95
            trainer.expert_prob = max(il_config['final_expert_prob'], 
                                     il_config['initial_expert_prob'] * (decay_rate ** update))
        
        # Collect rollouts
        trainer.collect_mixed_rollouts(trainer.ppo.rollout_length)
        
        # PPO update
        if len(trainer.ppo.rollout_buffer) > 0:
            ppo_stats = trainer.ppo.update()
        
        # Periodic BC update
        if update % il_config['bc_update_frequency'] == 0 and len(trainer.expert_buffer) > 0:
            for _ in range(il_config['bc_epochs_per_update']):
                bc_loss = trainer.behavioral_cloning_update()
        
        # Logging
        if update % training_config['log_frequency'] == 0:
            log_training_progress(trainer, update, num_updates)
        
        # Save checkpoint
        if update % training_config['save_frequency'] == 0:
            save_checkpoint(trainer, config, f"phase2_checkpoint_{update}.pt")
    
    print("\nPhase 2 complete")

def phase3_finetuning(trainer, config):
    """Phase 3: Fine-tuning with minimal expert guidance."""
    print("\n" + "="*60)
    print("PHASE 3: FINE-TUNING")
    print("="*60)
    
    training_config = config['training']
    
    # Set low expert probability
    trainer.expert_prob = training_config['phase3_expert_prob']
    
    timesteps = training_config['phase3_timesteps']
    num_updates = timesteps // trainer.ppo.rollout_length
    
    print(f"\nFine-tuning for {timesteps} timesteps")
    print(f"Expert probability: {trainer.expert_prob:.2f}")
    
    for update in range(num_updates):
        # Collect rollouts
        trainer.collect_mixed_rollouts(trainer.ppo.rollout_length)
        
        # PPO update
        if len(trainer.ppo.rollout_buffer) > 0:
            ppo_stats = trainer.ppo.update()
        
        # Logging
        if update % training_config['log_frequency'] == 0:
            log_training_progress(trainer, update, num_updates, phase="Phase 3")
    
    print("\nPhase 3 complete")

def log_training_progress(trainer, update, total_updates, phase="Phase 2"):
    """Log training progress."""
    print(f"\n[{phase}] Update {update}/{total_updates} (Expert prob: {trainer.expert_prob:.2f})")
    
    # Calculate statistics
    stats = {}
    
    if trainer.stats['ppo_rewards']:
        rewards = list(trainer.stats['ppo_rewards'])[-10:]
        stats['avg_reward'] = np.mean(rewards) if rewards else 0
        print(f"  Avg Reward: {stats['avg_reward']:.2f}")
    
    if trainer.stats['completion_rate']:
        completions = list(trainer.stats['completion_rate'])[-10:]
        stats['avg_completion'] = np.mean(completions) if completions else 0
        print(f"  Avg Completion: {stats['avg_completion']:.2%}")
    
    if trainer.stats['expert_match_rate']:
        matches = list(trainer.stats['expert_match_rate'])[-10:]
        stats['avg_match'] = np.mean(matches) if matches else 0
        print(f"  Expert Match Rate: {stats['avg_match']:.2%}")
    
    # Save to log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"training_log_{phase.replace(' ', '_')}.txt"
    with open(log_file, 'a') as f:
        f.write(f"Update {update}: {stats}\n")

def save_checkpoint(trainer, config, filename):
    """Save training checkpoint."""
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    
    torch.save({
        'policy_state': trainer.policy_net.state_dict(),
        'value_state': trainer.value_net.state_dict(),
        'config': config
    }, checkpoint_path)
    
    print(f"  Checkpoint saved to {checkpoint_path}")

def main():
    """Main training with improved configuration."""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("IMPROVED PHASE 2 TRAINING")
    print("="*60)
    
    # Get configuration
    problem_size = "small"  # Can be "debug", "small", or "medium"
    config = get_training_config(problem_size)
    
    print(f"\nUsing {problem_size} problem configuration")
    print(f"Problem parameters: {config['selected_problem']}")
    
    # Generate problem
    generator = QuantumProblemGenerator(**config['selected_problem'], seed=42)
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    
    # Create environment
    env = DQCEnvironment(problem_data, {'max_steps': 200})
    
    # Create networks
    model_config = config['model']
    rgcn = RGCN(model_config)
    policy_net = PolicyNetwork(rgcn, model_config)
    value_net = ValueNetwork(rgcn, model_config)
    
    # Create PPO
    ppo = PPO(policy_net, value_net, config['ppo'])
    
    # Create improved trainer
    il_config = config['imitation_learning']
    trainer_config = {
        'bc_epochs': il_config['bc_epochs_per_update'],
        'bc_batch_size': il_config['bc_pretrain_batch_size'],
        'bc_lr': il_config['bc_pretrain_lr'],
        'expert_buffer_size': il_config['expert_buffer_size'],
        'expert_prob': il_config['initial_expert_prob'],
        'reward_config': il_config['reward_weights'],
        'enable_profiling': False
    }
    
    trainer = ImitationLearningTrainer(env, policy_net, value_net, ppo, trainer_config)
    
    # Training phases
    overall_start = time.time()
    
    # Phase 1: BC Pretraining
    bc_loss = phase1_bc_pretraining(trainer, config)
    
    # Phase 2: Mixed Training
    phase2_mixed_training(trainer, config)
    
    # Phase 3: Fine-tuning
    phase3_finetuning(trainer, config)
    
    # Save final model
    final_model_path = config['paths']['model_save']
    torch.save({
        'policy_state': policy_net.state_dict(),
        'value_state': value_net.state_dict(),
        'config': config,
        'final_stats': dict(trainer.stats)
    }, final_model_path)
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {overall_time:.2f} seconds")
    print(f"Model saved to {final_model_path}")
    
    # Print final statistics
    if trainer.stats['ppo_rewards']:
        rewards = list(trainer.stats['ppo_rewards'])
        print(f"Final average reward: {np.mean(rewards[-20:]):.2f}")
        print(f"Best reward: {max(rewards):.2f}")
    
    if trainer.stats['completion_rate']:
        completions = list(trainer.stats['completion_rate'])
        print(f"Final completion rate: {np.mean(completions[-20:]):.2%}")
        print(f"Best completion rate: {max(completions):.2%}")

if __name__ == "__main__":
    main()