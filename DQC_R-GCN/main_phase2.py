"""
Main training script with integrated monitoring and adaptive training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
import yaml
import pickle
from pathlib import Path

from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms.ppo import PPO
from training.imitation_learning import ImitationLearningTrainer
from utils.training_monitor import TrainingMonitor

def load_config(config_path="configs/training_config.yaml"):
    """Load and validate configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Save config backup
    backup_path = Path(config['paths']['config_backup'])
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, 'w') as f:
        yaml.dump(config, f)
    
    return config

def phase1_bc_pretraining_monitored(trainer, monitor, config):
    """Enhanced Phase 1 with monitoring and early stopping."""
    print("\n" + "="*60)
    print("PHASE 1: ENHANCED BC PRETRAINING")
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
    
    # BC pretraining with monitoring
    print(f"\nBC Pretraining for up to {il_config['bc_pretrain_epochs']} epochs...")
    
    best_loss = float('inf')
    no_improvement_count = 0
    patience = config['training']['phase1_early_stop_patience']
    min_expert_prob = config['training']['phase1_min_expert_prob']
    
    for epoch in range(il_config['bc_pretrain_epochs']):
        loss = trainer.behavioral_cloning_update()
        
        # Calculate expert action probabilities
        expert_probs = test_expert_probabilities(trainer, num_samples=10)
        
        # Update metrics
        metrics = {
            'bc_loss': loss,
            'expert_probability': np.mean(expert_probs) if expert_probs else 0
        }
        
        actions = monitor.update_metrics(metrics, phase="phase1_bc")
        
        # Track best loss
        if loss < best_loss:
            best_loss = loss
            no_improvement_count = 0
            
            # Save best BC model
            if actions and actions.get('save_best_model'):
                save_bc_model(trainer, config, f"best_bc_model_epoch{epoch}.pt")
        else:
            no_improvement_count += 1
        
        # Logging
        if epoch % 10 == 0:
            avg_expert_prob = np.mean(expert_probs) if expert_probs else 0
            print(f"  Epoch {epoch}: Loss={loss:.4f} (Best: {best_loss:.4f}), "
                  f"Expert Prob={avg_expert_prob:.3f}")
        
        # Early stopping
        if no_improvement_count >= patience:
            print(f"\n[EARLY STOP] No improvement for {patience} epochs")
            break
        
        # Check if minimum expert probability achieved
        if expert_probs and np.mean(expert_probs) >= min_expert_prob:
            print(f"\n[SUCCESS] Achieved target expert probability: {np.mean(expert_probs):.3f}")
            break
    
    print(f"\nPhase 1 complete. Final BC loss: {loss:.4f}")
    return best_loss

def phase2_mixed_training_adaptive(trainer, monitor, config):
    """Adaptive Phase 2 with performance monitoring."""
    print("\n" + "="*60)
    print("PHASE 2: ADAPTIVE MIXED TRAINING")
    print("="*60)
    
    il_config = config['imitation_learning']
    training_config = config['training']
    
    timesteps = training_config['phase2_timesteps']
    num_updates = timesteps // trainer.ppo.rollout_length
    
    print(f"\nTraining for {timesteps} timesteps ({num_updates} updates)")
    
    # Track performance for adaptive expert probability
    recent_performance = []
    expert_prob_adjustments = 0
    
    for update in range(num_updates):
        # Get scheduled expert probability
        current_steps = update * trainer.ppo.rollout_length
        scheduled_prob = monitor.get_expert_probability_schedule(current_steps)
        
        # Apply any adaptive adjustments
        if trainer.expert_prob < scheduled_prob:
            trainer.expert_prob = scheduled_prob
        
        # Collect rollouts
        trainer.collect_mixed_rollouts(trainer.ppo.rollout_length)
        
        # PPO update
        if len(trainer.ppo.rollout_buffer) > 0:
            ppo_stats = trainer.ppo.update()
        
        # Periodic BC update
        if update % il_config['bc_update_frequency'] == 0 and len(trainer.expert_buffer) > 0:
            # Check if BC loss is still high enough to warrant updates
            current_bc_loss = trainer.stats['bc_loss'][-1] if trainer.stats['bc_loss'] else float('inf')
            
            if current_bc_loss > il_config.get('min_bc_loss_threshold', 1.5):
                for _ in range(il_config['bc_epochs_per_update']):
                    bc_loss = trainer.behavioral_cloning_update()
        
        # Calculate metrics
        metrics = calculate_current_metrics(trainer)
        metrics['expert_probability'] = trainer.expert_prob
        
        # Monitor and get actions
        actions = monitor.update_metrics(metrics, phase="phase2")
        
        # Handle monitor actions
        if actions:
            if actions.get('save_best_model'):
                monitor.save_best_model(trainer.policy_net, trainer.value_net, dict(trainer.stats))
            
            if actions.get('increase_expert_prob'):
                # Adaptive increase in expert probability
                trainer.expert_prob = min(1.0, trainer.expert_prob + actions['increase_expert_prob'])
                expert_prob_adjustments += 1
                print(f"  [ADAPTIVE] Increased expert prob to {trainer.expert_prob:.2f}")
            
            if actions.get('early_stop'):
                print("  [EARLY STOP] Triggered by monitor")
                break
        
        # Logging
        if update % training_config['log_frequency'] == 0:
            log_training_progress(trainer, update, num_updates, metrics)
        
        # Save checkpoint
        if update % training_config['save_frequency'] == 0:
            save_checkpoint(trainer, config, f"phase2_checkpoint_{update}.pt")
        
        # Track recent performance
        recent_performance.append(metrics.get('completion_rate', 0))
        if len(recent_performance) > 20:
            recent_performance.pop(0)
        
        # Check if performance is consistently good
        if len(recent_performance) >= 10:
            avg_recent = np.mean(recent_performance[-10:])
            if avg_recent < training_config['phase2_performance_threshold']:
                if training_config.get('phase2_adaptive_expert', True):
                    # Increase expert probability if performance drops
                    trainer.expert_prob = min(1.0, trainer.expert_prob + 0.05)
                    print(f"  [ADAPTIVE] Performance below threshold, increased expert prob to {trainer.expert_prob:.2f}")
    
    print(f"\nPhase 2 complete. Expert prob adjustments: {expert_prob_adjustments}")

def phase3_finetuning_safeguarded(trainer, monitor, config):
    """Phase 3 with performance safeguards."""
    print("\n" + "="*60)
    print("PHASE 3: SAFEGUARDED FINE-TUNING")
    print("="*60)
    
    training_config = config['training']
    
    # Set minimum expert probability
    min_expert_prob = training_config.get('phase3_min_expert_prob', 0.1)
    trainer.expert_prob = max(min_expert_prob, training_config.get('phase3_expert_prob', 0.05))
    
    timesteps = training_config['phase3_timesteps']
    num_updates = timesteps // trainer.ppo.rollout_length
    
    print(f"\nFine-tuning for {timesteps} timesteps")
    print(f"Expert probability: {trainer.expert_prob:.2f} (min: {min_expert_prob:.2f})")
    
    # Track performance for safeguards
    baseline_performance = calculate_current_metrics(trainer).get('avg_reward', 0)
    performance_drops = 0
    
    for update in range(num_updates):
        # Collect rollouts
        trainer.collect_mixed_rollouts(trainer.ppo.rollout_length)
        
        # PPO update
        if len(trainer.ppo.rollout_buffer) > 0:
            ppo_stats = trainer.ppo.update()
        
        # Calculate metrics
        metrics = calculate_current_metrics(trainer)
        metrics['expert_probability'] = trainer.expert_prob
        
        # Monitor
        actions = monitor.update_metrics(metrics, phase="phase3")
        
        # Handle actions
        if actions:
            if actions.get('save_best_model'):
                monitor.save_best_model(trainer.policy_net, trainer.value_net, dict(trainer.stats))
            
            if actions.get('revert_checkpoint'):
                print("  [SAFEGUARD] Performance drop detected, considering revert")
                performance_drops += 1
                
                if performance_drops >= 3:
                    print("  [SAFEGUARD] Multiple drops detected, reverting to Phase 2 settings")
                    trainer.expert_prob = 0.3  # Increase expert guidance
                    break
        
        # Check performance safeguards
        if training_config.get('phase3_performance_safeguard', True):
            current_performance = metrics.get('avg_reward', 0)
            if current_performance < baseline_performance * 0.5:
                print(f"  [SAFEGUARD] Performance dropped significantly ({current_performance:.2f} vs {baseline_performance:.2f})")
                trainer.expert_prob = min(1.0, trainer.expert_prob * 2)  # Double expert probability
        
        # Logging
        if update % training_config['log_frequency'] == 0:
            log_training_progress(trainer, update, num_updates, metrics, phase="Phase 3")
    
    print(f"\nPhase 3 complete. Performance drops: {performance_drops}")

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
    
    return probs

def calculate_current_metrics(trainer):
    """Calculate current training metrics."""
    metrics = {}
    
    if trainer.stats['ppo_rewards']:
        rewards = list(trainer.stats['ppo_rewards'])[-10:]
        metrics['avg_reward'] = np.mean(rewards) if rewards else 0
    
    if trainer.stats['completion_rate']:
        completions = list(trainer.stats['completion_rate'])[-10:]
        metrics['completion_rate'] = np.mean(completions) if completions else 0
    
    if trainer.stats['expert_match_rate']:
        matches = list(trainer.stats['expert_match_rate'])[-10:]
        metrics['expert_match_rate'] = np.mean(matches) if matches else 0
    
    if trainer.stats['bc_loss']:
        bc_losses = list(trainer.stats['bc_loss'])[-5:]
        metrics['bc_loss'] = np.mean(bc_losses) if bc_losses else 0
    
    return metrics

def log_training_progress(trainer, update, total_updates, metrics, phase="Phase 2"):
    """Log training progress."""
    print(f"\n[{phase}] Update {update}/{total_updates} (Expert prob: {trainer.expert_prob:.2f})")
    
    for key, value in metrics.items():
        if key == 'completion_rate' or key == 'expert_match_rate':
            print(f"  {key}: {value:.2%}")
        elif key != 'expert_probability':
            print(f"  {key}: {value:.2f}")

def save_bc_model(trainer, config, filename):
    """Save BC pretrained model."""
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = checkpoint_dir / filename
    torch.save({
        'policy_state': trainer.policy_net.state_dict(),
        'value_state': trainer.value_net.state_dict()
    }, save_path)
    print(f"  BC model saved to {save_path}")

def save_checkpoint(trainer, config, filename):
    """Save training checkpoint."""
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = checkpoint_dir / filename
    torch.save({
        'policy_state': trainer.policy_net.state_dict(),
        'value_state': trainer.value_net.state_dict(),
        'trainer_stats': dict(trainer.stats)
    }, save_path)

def main():
    """Main training with monitoring and adaptive strategies."""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("OPTIMIZED TRAINING WITH MONITORING")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Select problem size
    problem_size = "small"  # Can be changed to "medium" or "debug"
    problem_config = config['problem'][problem_size]
    
    print(f"\nUsing {problem_size} problem configuration")
    
    # Generate problem
    generator = QuantumProblemGenerator(**problem_config, seed=42)
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
    
    # Create trainer
    il_config = config['imitation_learning']
    trainer_config = {
        'bc_epochs': il_config['bc_epochs_per_update'],
        'bc_batch_size': il_config['bc_pretrain_batch_size'],
        'bc_lr': il_config['bc_pretrain_lr'],
        'expert_buffer_size': il_config['expert_buffer_size'],
        'expert_prob': il_config['initial_expert_prob'],
        'reward_config': il_config['reward_weights'],
        'enable_profiling': config['training'].get('enable_profiling', False),
        'debug_mode': config['training'].get('verbose', False),
        'debug_frequency': 100
    }
    
    trainer = ImitationLearningTrainer(env, policy_net, value_net, ppo, trainer_config)
    
    # Create monitor
    monitor = TrainingMonitor(config)
    
    # Training phases
    overall_start = time.time()
    
    try:
        # Phase 1: Enhanced BC Pretraining
        bc_loss = phase1_bc_pretraining_monitored(trainer, monitor, config)
        
        # Phase 2: Adaptive Mixed Training
        phase2_mixed_training_adaptive(trainer, monitor, config)
        
        # Phase 3: Safeguarded Fine-tuning
        phase3_finetuning_safeguarded(trainer, monitor, config)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        final_model_path = config['paths']['model_save']
        torch.save({
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
            'config': config,
            'final_stats': dict(trainer.stats)
        }, final_model_path)
        
        # Print summary
        monitor.print_summary()
        
        # Clean up
        monitor.close()
    
    overall_time = time.time() - overall_start
    
    print(f"\nTotal training time: {overall_time:.2f} seconds")
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()