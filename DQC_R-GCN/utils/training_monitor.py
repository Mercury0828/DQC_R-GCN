"""
Training monitor with real-time visualization and model management.
"""

import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

class TrainingMonitor:
    """Comprehensive training monitor with visualization and model management."""
    
    def __init__(self, config: Dict):
        """Initialize training monitor."""
        self.config = config
        self.training_config = config.get('training', {})
        self.paths = config.get('paths', {})
        
        # Metrics tracking
        self.metrics_history = {
            'avg_reward': deque(maxlen=1000),
            'completion_rate': deque(maxlen=1000),
            'expert_match_rate': deque(maxlen=1000),
            'bc_loss': deque(maxlen=1000),
            'value_loss': deque(maxlen=1000),
            'policy_loss': deque(maxlen=1000),
            'expert_probability': deque(maxlen=1000),
            'episode_length': deque(maxlen=1000),
        }
        
        # Best model tracking
        self.best_metrics = {
            'avg_reward': -float('inf'),
            'completion_rate': 0.0,
            'combined_score': -float('inf')  # Weighted combination
        }
        self.best_model_update = 0
        
        # Early stopping
        self.early_stop_config = self.training_config.get('early_stop', {})
        self.patience_counter = 0
        self.last_best_metric = -float('inf')
        
        # Performance safeguards
        self.performance_threshold = self.training_config.get('phase2_performance_threshold', 0.8)
        self.adaptive_expert = self.training_config.get('phase2_adaptive_expert', True)
        
        # Visualization
        self.viz_config = self.training_config.get('visualization', {})
        self.plot_dir = Path(self.viz_config.get('plot_dir', 'plots/'))
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize plots
        if self.viz_config.get('enabled', False):
            self._init_plots()
        
        # Logging
        self.log_dir = Path(self.paths.get('log_dir', 'logs/'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Tensorboard
        if self.training_config.get('enable_tensorboard', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(self.paths.get('tensorboard_dir', 'tensorboard/'))
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(tb_dir)
            except ImportError:
                print("Warning: Tensorboard not available")
                self.tb_writer = None
        else:
            self.tb_writer = None
        
        # Update counter
        self.global_step = 0
        
    def _init_plots(self):
        """Initialize matplotlib plots."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Training Progress Monitor')
        
        # Configure subplots
        self.axes[0, 0].set_title('Rewards')
        self.axes[0, 0].set_xlabel('Updates')
        self.axes[0, 0].set_ylabel('Average Reward')
        
        self.axes[0, 1].set_title('Completion Rate')
        self.axes[0, 1].set_xlabel('Updates')
        self.axes[0, 1].set_ylabel('Rate (%)')
        
        self.axes[1, 0].set_title('Expert Matching')
        self.axes[1, 0].set_xlabel('Updates')
        self.axes[1, 0].set_ylabel('Match Rate (%)')
        
        self.axes[1, 1].set_title('BC Loss')
        self.axes[1, 1].set_xlabel('Updates')
        self.axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
    
    def update_metrics(self, metrics: Dict[str, float], phase: str = "training"):
        """Update metrics and check conditions."""
        self.global_step += 1
        
        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics_history and value is not None:
                self.metrics_history[key].append(value)
        
        # Log to tensorboard
        if self.tb_writer:
            for key, value in metrics.items():
                if value is not None:
                    self.tb_writer.add_scalar(f"{phase}/{key}", value, self.global_step)
        
        # Check for best model
        if self._is_best_model(metrics):
            self.best_model_update = self.global_step
            self._update_best_metrics(metrics)
            return {'save_best_model': True}
        
        # Check early stopping
        if self.early_stop_config.get('enabled', False):
            if self._check_early_stopping(metrics):
                return {'early_stop': True}
        
        # Check performance safeguards
        safeguard_actions = self._check_safeguards(metrics)
        if safeguard_actions:
            return safeguard_actions
        
        # Update visualization
        if self.viz_config.get('enabled', False):
            if self.global_step % self.viz_config.get('update_frequency', 10) == 0:
                self._update_plots()
        
        # Save metrics to file
        self._log_metrics(metrics, phase)
        
        return {}
    
    def _is_best_model(self, metrics: Dict) -> bool:
        """Check if current model is best so far."""
        metric_name = self.training_config.get('best_model', {}).get('metric', 'avg_reward')
        
        if metric_name == 'combined_score':
            # Weighted combination of metrics
            score = (metrics.get('avg_reward', 0) * 0.5 + 
                    metrics.get('completion_rate', 0) * 100 * 0.3 +
                    metrics.get('expert_match_rate', 0) * 100 * 0.2)
            return score > self.best_metrics['combined_score']
        else:
            current_value = metrics.get(metric_name, -float('inf'))
            return current_value > self.best_metrics.get(metric_name, -float('inf'))
    
    def _update_best_metrics(self, metrics: Dict):
        """Update best metrics."""
        for key in self.best_metrics:
            if key == 'combined_score':
                self.best_metrics[key] = (
                    metrics.get('avg_reward', 0) * 0.5 + 
                    metrics.get('completion_rate', 0) * 100 * 0.3 +
                    metrics.get('expert_match_rate', 0) * 100 * 0.2
                )
            else:
                self.best_metrics[key] = metrics.get(key, self.best_metrics[key])
    
    def _check_early_stopping(self, metrics: Dict) -> bool:
        """Check early stopping criteria."""
        metric_name = self.early_stop_config.get('metric', 'avg_reward')
        current_value = metrics.get(metric_name, 0)
        min_delta = self.early_stop_config.get('min_delta', 0.01)
        
        if current_value > self.last_best_metric + min_delta:
            self.last_best_metric = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        patience = self.early_stop_config.get('patience', 100)
        if self.patience_counter >= patience:
            print(f"\n[EARLY STOPPING] No improvement for {patience} updates")
            return True
        
        return False
    
    def _check_safeguards(self, metrics: Dict) -> Optional[Dict]:
        """Check performance safeguards and suggest actions."""
        actions = {}
        
        # Check completion rate
        completion_rate = metrics.get('completion_rate', 0)
        if completion_rate < self.performance_threshold:
            if self.adaptive_expert:
                # Suggest increasing expert probability
                actions['increase_expert_prob'] = 0.1  # Increase by 10%
                print(f"\n[SAFEGUARD] Completion rate {completion_rate:.2%} below threshold, increasing expert probability")
        
        # Check for performance collapse
        if len(self.metrics_history['avg_reward']) > 20:
            recent_rewards = list(self.metrics_history['avg_reward'])[-10:]
            older_rewards = list(self.metrics_history['avg_reward'])[-20:-10]
            
            if np.mean(recent_rewards) < np.mean(older_rewards) * 0.5:
                actions['revert_checkpoint'] = True
                print(f"\n[SAFEGUARD] Performance collapse detected, suggesting checkpoint revert")
        
        return actions if actions else None
    
    def _update_plots(self):
        """Update visualization plots."""
        if not self.viz_config.get('enabled', False):
            return
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot metrics - FIX: ensure consistent lengths
        
        # Rewards
        if self.metrics_history['avg_reward']:
            rewards = list(self.metrics_history['avg_reward'])
            steps = list(range(len(rewards)))
            self.axes[0, 0].plot(steps, rewards)
            self.axes[0, 0].set_title('Average Reward')
            self.axes[0, 0].set_xlabel('Updates')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].grid(True)
        
        # Completion rate
        if self.metrics_history['completion_rate']:
            completions = list(self.metrics_history['completion_rate'])
            completion_pct = [x * 100 for x in completions]
            steps = list(range(len(completion_pct)))  # FIX: generate steps for this data
            self.axes[0, 1].plot(steps, completion_pct)
            self.axes[0, 1].set_title('Completion Rate')
            self.axes[0, 1].set_xlabel('Updates')
            self.axes[0, 1].set_ylabel('Rate (%)')
            self.axes[0, 1].set_ylim([0, 105])
            self.axes[0, 1].grid(True)
        
        # Expert match rate
        if self.metrics_history['expert_match_rate']:
            matches = list(self.metrics_history['expert_match_rate'])
            match_pct = [x * 100 for x in matches]
            steps = list(range(len(match_pct)))  # FIX: generate steps for this data
            self.axes[1, 0].plot(steps, match_pct)
            self.axes[1, 0].set_title('Expert Match Rate')
            self.axes[1, 0].set_xlabel('Updates')
            self.axes[1, 0].set_ylabel('Match Rate (%)')
            self.axes[1, 0].set_ylim([0, 105])
            self.axes[1, 0].grid(True)
        
        # BC Loss
        if self.metrics_history['bc_loss']:
            bc_losses = list(self.metrics_history['bc_loss'])
            steps = list(range(len(bc_losses)))  # FIX: generate steps for this data
            self.axes[1, 1].plot(steps, bc_losses)
            self.axes[1, 1].set_title('BC Loss')
            self.axes[1, 1].set_xlabel('Updates')
            self.axes[1, 1].set_ylabel('Loss')
            self.axes[1, 1].set_yscale('log')
            self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if self.viz_config.get('save_plots', True):
            plot_path = self.plot_dir / f"training_progress_{self.global_step}.png"
            plt.savefig(plot_path, dpi=100)
    
    def _log_metrics(self, metrics: Dict, phase: str):
        """Log metrics to file."""
        log_entry = {
            'step': self.global_step,
            'phase': phase,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def save_best_model(self, policy_net, value_net, trainer_stats: Dict):
        """Save best model."""
        save_path = self.training_config.get('best_model', {}).get('save_path', 'best_model.pt')
        
        torch.save({
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
            'best_metrics': dict(self.best_metrics),
            'trainer_stats': trainer_stats,
            'global_step': self.global_step,
            'config': self.config
        }, save_path)
        
        print(f"[BEST MODEL] Saved at step {self.global_step} with metrics: {self.best_metrics}")
    
    def load_checkpoint(self, checkpoint_path: str, policy_net, value_net):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        policy_net.load_state_dict(checkpoint['policy_state'])
        value_net.load_state_dict(checkpoint['value_state'])
        
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        
        print(f"[CHECKPOINT] Loaded from {checkpoint_path}")
        return checkpoint.get('trainer_stats', {})
    
    def get_expert_probability_schedule(self, current_step: int) -> float:
        """Get expert probability based on schedule."""
        decay_config = self.config.get('imitation_learning', {}).get('expert_decay', {})
        
        if decay_config.get('type') == 'staged':
            stages = decay_config.get('stages', [])
            for stage in stages:
                if current_step <= stage['steps']:
                    return stage['prob']
            return stages[-1]['prob'] if stages else 0.2
        else:
            # Linear decay fallback
            initial = self.config.get('imitation_learning', {}).get('initial_expert_prob', 0.5)
            final = self.config.get('imitation_learning', {}).get('final_expert_prob', 0.1)
            total_steps = self.training_config.get('phase2_timesteps', 25000)
            
            if current_step >= total_steps:
                return final
            
            progress = current_step / total_steps
            return initial * (1 - progress) + final * progress
    
    def close(self):
        """Clean up resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.viz_config.get('enabled', False):
            plt.close(self.fig)
    
    def print_summary(self):
        """Print training summary."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if self.metrics_history['avg_reward']:
            recent_rewards = list(self.metrics_history['avg_reward'])[-20:]
            print(f"Recent average reward: {np.mean(recent_rewards):.2f}")
            print(f"Best average reward: {self.best_metrics['avg_reward']:.2f}")
        
        if self.metrics_history['completion_rate']:
            recent_completion = list(self.metrics_history['completion_rate'])[-20:]
            print(f"Recent completion rate: {np.mean(recent_completion):.2%}")
            print(f"Best completion rate: {self.best_metrics['completion_rate']:.2%}")
        
        print(f"Best model saved at update: {self.best_model_update}")
        print("="*60)