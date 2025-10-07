import torch
import json
from pathlib import Path
from typing import Dict

class CheckpointManager:
    """Manage model checkpoints and recovery."""
    
    def __init__(self, checkpoint_dir="checkpoints/"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
        
    def save_checkpoint(self, policy_net, value_net, metrics, step, phase):
        """Save a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{phase}_{step}.pt"
        
        checkpoint = {
            'step': step,
            'phase': phase,
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoints.append({
            'path': checkpoint_path,
            'step': step,
            'phase': phase,
            'reward': metrics.get('avg_reward', 0)
        })
        
        # 保存checkpoint索引
        with open(self.checkpoint_dir / "checkpoints.json", 'w') as f:
            json.dump(self.checkpoints, f, default=str)
        
        print(f"  [CHECKPOINT] Saved at step {step}, phase {phase}")
        
    def load_best_checkpoint(self, policy_net, value_net, phase=None):
        """Load the best checkpoint."""
        if not self.checkpoints:
            return False
            
        # 找最佳checkpoint
        if phase:
            candidates = [c for c in self.checkpoints if c['phase'] == phase]
        else:
            candidates = self.checkpoints
            
        if not candidates:
            return False
            
        best = max(candidates, key=lambda x: x['reward'])
        checkpoint = torch.load(best['path'])
        
        policy_net.load_state_dict(checkpoint['policy_state'])
        value_net.load_state_dict(checkpoint['value_state'])
        
        print(f"  [CHECKPOINT] Loaded best from {best['phase']} with reward {best['reward']:.2f}")
        return True