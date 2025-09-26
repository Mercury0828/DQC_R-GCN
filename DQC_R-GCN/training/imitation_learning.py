"""
Imitation learning trainer for UNIQ-RL - Fixed version.
Combines behavioral cloning with online RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

# Try to import wandb but don't fail if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import expert policy
from expert_policy.uniq_expert import UNIQExpertPolicy
from rewards.enhanced_rewards import EnhancedRewardFunction

# Try to import performance monitor
try:
    from utils.performance_monitor import perf_monitor, profile
    PERF_MONITOR_AVAILABLE = True
except ImportError:
    PERF_MONITOR_AVAILABLE = False
    # Create dummy decorators
    def profile(name=None):
        def decorator(func):
            return func
        return decorator
    
    # Create dummy perf_monitor
    class DummyPerfMonitor:
        def timer(self, name):
            from contextlib import nullcontext
            return nullcontext()
        def get_stats(self, name):
            return {}
        def print_summary(self, top_n=10):
            pass
        enabled = False
    
    perf_monitor = DummyPerfMonitor()

class OptimizedImitationLearningTrainer:
    """
    Optimized trainer with batched operations and performance monitoring.
    """
    
    def __init__(self, env, policy_net, value_net, ppo_algorithm, config: Dict):
        """Initialize optimized imitation learning trainer."""
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.ppo = ppo_algorithm
        self.config = config
        
        # Expert policy
        self.expert = UNIQExpertPolicy(env)
        
        # Enhanced reward function
        reward_config = config.get('reward_config', {})
        self.reward_fn = EnhancedRewardFunction(reward_config)
        
        # Imitation learning parameters
        self.bc_epochs = config.get('bc_epochs', 10)
        self.bc_batch_size = config.get('bc_batch_size', 64)
        self.expert_buffer_size = config.get('expert_buffer_size', 10000)
        self.expert_prob = config.get('expert_prob', 0.3)
        
        # Expert demonstration buffer
        self.expert_buffer = deque(maxlen=self.expert_buffer_size)
        
        # Behavioral cloning optimizer
        self.bc_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('bc_lr', 3e-4)
        )
        
        # Training statistics
        self.stats = {
            'bc_loss': deque(maxlen=100),
            'expert_match_rate': deque(maxlen=100),
            'ppo_rewards': deque(maxlen=100),
            'completion_rate': deque(maxlen=100)
        }
        
        # Performance monitoring
        self.enable_profiling = config.get('enable_profiling', False) and PERF_MONITOR_AVAILABLE
        if self.enable_profiling:
            perf_monitor.enabled = True
        
        # Initialize wandb if configured
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="uniq-rl",
                config=config,
                name=config.get('run_name', 'imitation_learning_optimized')
            )
    
    @profile("generate_expert_demonstrations")
    def generate_expert_demonstrations(self, num_episodes: int = 10):
        """Generate expert demonstrations with performance monitoring."""
        print(f"Generating {num_episodes} expert demonstrations...")
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            with perf_monitor.timer(f"expert_episode_{episode}"):
                trajectory = self._generate_single_expert_trajectory()
                
                # Preprocess and add to buffer
                with perf_monitor.timer("preprocess_trajectory"):
                    self._preprocess_and_store_trajectory(trajectory)
            
            # Calculate episode statistics
            total_reward = sum(t['reward'] for t in trajectory)
            completion = self.env._get_completion_rate()
            
            # Get timing info safely
            episode_time = time.time() - episode_start
            
            print(f"  Episode {episode + 1}: Length={len(trajectory)}, "
                  f"Reward={total_reward:.2f}, Completion={completion:.2%}, "
                  f"Time={episode_time:.2f}s")
        
        print(f"Total expert transitions: {len(self.expert_buffer)}")
    
    def _generate_single_expert_trajectory(self) -> List[Dict]:
        """Generate a single expert trajectory."""
        trajectory = []
        state = self.env.reset()
        max_steps = 1000
        
        for step in range(max_steps):
            state_repr = self.env.state.get_state_representation()
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                break
            
            with perf_monitor.timer("expert_action_selection"):
                expert_action = self.expert.get_expert_action(state_repr, valid_actions)
            
            next_state, reward, done, info = self.env.step(expert_action)
            
            trajectory.append({
                'state': state_repr,
                'action': expert_action,
                'reward': reward,
                'next_state': self.env.state.get_state_representation(),
                'done': done,
                'valid_actions': valid_actions
            })
            
            if done:
                break
            
            state = next_state
        
        return trajectory
    
    def _preprocess_and_store_trajectory(self, trajectory: List[Dict]):
        """Preprocess trajectory for efficient batch processing."""
        for transition in trajectory:
            # Pre-compute action encoding for faster BC
            action_encoding = self._encode_action_for_bc(
                transition['action'],
                transition['valid_actions']
            )
            
            if action_encoding is not None:
                preprocessed = {
                    'state': transition['state'],
                    'action': transition['action'],
                    'action_encoding': action_encoding,
                    'valid_actions': transition['valid_actions'],
                    'reward': transition['reward']
                }
                self.expert_buffer.append(preprocessed)
    
    def _encode_action_for_bc(self, action: Tuple, valid_actions: Dict) -> Optional[Dict]:
        """Pre-encode action for behavioral cloning."""
        # Create a mapping of all valid actions
        all_actions = []
        for q, u in valid_actions.get('map', []):
            all_actions.append(('map', (q, u)))
        for g, t in valid_actions.get('schedule', []):
            all_actions.append(('schedule', (g, t)))
        
        # Find index of expert action
        for idx, (act_type, act_params) in enumerate(all_actions):
            if act_type == action[0] and act_params == action[1]:
                return {
                    'index': idx,
                    'total_actions': len(all_actions),
                    'all_actions': all_actions
                }
        
        return None
    
    @profile("behavioral_cloning_batch_update")
    def behavioral_cloning_update(self):
        """Optimized batch behavioral cloning update."""
        if len(self.expert_buffer) < self.bc_batch_size:
            return 0.0
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(self.bc_epochs):
            # Sample and batch process
            with perf_monitor.timer("bc_batch_sampling"):
                batch = self._sample_bc_batch(self.bc_batch_size)
            
            if batch is None:
                continue
            
            with perf_monitor.timer("bc_forward_backward"):
                # Batch forward pass
                loss = self._compute_bc_loss_batch(batch)
                
                if loss is not None:
                    # Backward pass
                    self.bc_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                    self.bc_optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            self.stats['bc_loss'].append(avg_loss)
            return avg_loss
        
        return 0.0
    
    def _sample_bc_batch(self, batch_size: int) -> Optional[List[Dict]]:
        """Sample a batch for BC training."""
        if len(self.expert_buffer) < batch_size:
            return None
        
        # Random sampling
        indices = np.random.choice(len(self.expert_buffer), batch_size, replace=False)
        batch = [self.expert_buffer[i] for i in indices]
        
        # Filter valid transitions
        valid_batch = []
        for item in batch:
            if item['action_encoding'] is not None:
                valid_batch.append(item)
        
        return valid_batch if valid_batch else None
    
    def _compute_bc_loss_batch(self, batch: List[Dict]) -> Optional[torch.Tensor]:
        """Compute BC loss for a batch of transitions."""
        losses = []
        
        # Group by similar action spaces for more efficient processing
        grouped = self._group_by_action_space(batch)
        
        for group in grouped:
            if not group:
                continue
            
            # Process group together
            group_losses = self._process_bc_group(group)
            losses.extend(group_losses)
        
        if losses:
            return torch.mean(torch.stack(losses))
        return None
    
    def _group_by_action_space(self, batch: List[Dict]) -> List[List[Dict]]:
        """Group transitions by similar action space size for batch processing."""
        groups = {}
        for item in batch:
            size = item['action_encoding']['total_actions']
            if size not in groups:
                groups[size] = []
            groups[size].append(item)
        return list(groups.values())
    
    def _process_bc_group(self, group: List[Dict]) -> List[torch.Tensor]:
        """Process a group of similar transitions."""
        losses = []
        
        for item in group:
            state = item['state']
            valid_actions = item['valid_actions']
            target_idx = item['action_encoding']['index']
            
            # Get policy output
            with perf_monitor.timer("bc_policy_forward"):
                action_probs, _, action_info = self.policy_net.forward(state, valid_actions)
            
            if len(action_info) > target_idx:
                # Compute cross-entropy loss
                if isinstance(action_probs, torch.Tensor):
                    if action_probs.dim() == 0:
                        action_probs = action_probs.unsqueeze(0)
                    
                    # Create target distribution
                    target = torch.zeros_like(action_probs)
                    target[target_idx] = 1.0
                    
                    # Cross-entropy loss
                    loss = -torch.sum(target * torch.log(action_probs + 1e-8))
                    losses.append(loss)
        
        return losses
    
    @profile("collect_mixed_rollouts")
    def collect_mixed_rollouts(self, n_steps: int):
        """Optimized rollout collection with mixed policy."""
        self.ppo.rollout_buffer.reset()
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        expert_actions_used = 0
        
        for step in range(n_steps):
            with perf_monitor.timer("rollout_step"):
                state_repr = self.env.state.get_state_representation()
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions['map'] and not valid_actions['schedule']:
                    break
                
                # Decide policy
                use_expert = np.random.random() < self.expert_prob
                
                if use_expert:
                    with perf_monitor.timer("expert_action_in_rollout"):
                        action = self.expert.get_expert_action(state_repr, valid_actions)
                        expert_actions_used += 1
                    
                    with torch.no_grad():
                        _, log_prob = self.policy_net.get_action(state_repr, valid_actions)
                        value = self.value_net(state_repr)
                else:
                    with torch.no_grad():
                        with perf_monitor.timer("policy_action_in_rollout"):
                            action, log_prob = self.policy_net.get_action(
                                state_repr, valid_actions, deterministic=False
                            )
                        value = self.value_net(state_repr)
                
                # Execute action
                next_state, base_reward, done, info = self.env.step(action)
                
                # Calculate enhanced reward
                expert_action = self.expert.get_expert_action(state_repr, valid_actions)
                reward = self.reward_fn.calculate_reward(
                    self.env, action, info, expert_action
                )
                
                # Store transition
                value_scalar = value.item() if isinstance(value, torch.Tensor) else float(value)
                log_prob_scalar = log_prob.item() if isinstance(log_prob, torch.Tensor) else float(log_prob)
                
                self.ppo.rollout_buffer.add(
                    state=state_repr,
                    action=action,
                    reward=reward,
                    value=value_scalar,
                    log_prob=log_prob_scalar,
                    done=done,
                    valid_actions=valid_actions
                )
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    self.stats['ppo_rewards'].append(episode_reward)
                    self.stats['completion_rate'].append(self.env._get_completion_rate())
                    
                    state = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    expert_actions_used = 0
                else:
                    state = next_state
        
        if episode_length > 0:
            self.stats['expert_match_rate'].append(expert_actions_used / episode_length)
    
    def train(self, total_timesteps: int):
        """Main training loop with performance monitoring."""
        print("="*60)
        print("OPTIMIZED UNIQ-RL TRAINING WITH IMITATION LEARNING")
        print("="*60)
        
        overall_start = time.time()
        
        # Phase 1: Generate expert demonstrations
        print("\nPhase 1: Generating expert demonstrations...")
        phase1_start = time.time()
        with perf_monitor.timer("phase1_expert_demos"):
            self.generate_expert_demonstrations(num_episodes=20)
        print(f"Phase 1 completed in {time.time() - phase1_start:.2f}s")
        
        # Phase 2: Behavioral cloning pretraining
        print("\nPhase 2: Behavioral cloning pretraining...")
        phase2_start = time.time()
        with perf_monitor.timer("phase2_bc_pretraining"):
            for epoch in range(20):
                bc_loss = self.behavioral_cloning_update()
                if epoch % 5 == 0:
                    print(f"  BC Epoch {epoch}: Loss={bc_loss:.4f}")
        print(f"Phase 2 completed in {time.time() - phase2_start:.2f}s")
        
        # Phase 3: Mixed training with PPO
        print("\nPhase 3: Mixed training with PPO...")
        phase3_start = time.time()
        with perf_monitor.timer("phase3_mixed_training"):
            num_updates = total_timesteps // self.ppo.rollout_length
            initial_expert_prob = self.expert_prob
            
            for update in range(num_updates):
                # Decay expert probability
                progress = update / num_updates
                self.expert_prob = initial_expert_prob * (1 - progress * 0.8)
                
                # Collect rollouts
                with perf_monitor.timer(f"rollout_collection_{update}"):
                    self.collect_mixed_rollouts(self.ppo.rollout_length)
                
                # PPO update
                if len(self.ppo.rollout_buffer) > 0:
                    with perf_monitor.timer(f"ppo_update_{update}"):
                        ppo_stats = self.ppo.update()
                
                # Periodic BC
                if update % 10 == 0 and len(self.expert_buffer) > 0:
                    with perf_monitor.timer(f"bc_update_{update}"):
                        bc_loss = self.behavioral_cloning_update()
                
                # Logging
                if update % 10 == 0:
                    self._log_training_stats(update, num_updates)
        
        print(f"Phase 3 completed in {time.time() - phase3_start:.2f}s")
        
        overall_time = time.time() - overall_start
        print(f"\nTraining completed in {overall_time:.2f} seconds!")
        
        # Print performance summary
        if self.enable_profiling:
            perf_monitor.print_summary()
        
        self._print_final_stats()
    
    def _log_training_stats(self, update: int, total_updates: int):
        """Log training statistics with timing info."""
        print(f"\nUpdate {update}/{total_updates} (Expert prob: {self.expert_prob:.2f})")
        
        # Log performance stats if available
        if self.enable_profiling:
            rollout_stats = perf_monitor.get_stats(f"rollout_collection_{update}")
            if 'last' in rollout_stats:
                print(f"  Rollout time: {rollout_stats['last']:.2f}s")
        
        # Log training metrics
        if self.stats['ppo_rewards']:
            rewards_list = list(self.stats['ppo_rewards'])
            avg_reward = np.mean(rewards_list[-10:]) if len(rewards_list) > 0 else 0
            print(f"  Avg Reward: {avg_reward:.2f}")
        
        if self.stats['completion_rate']:
            completion_list = list(self.stats['completion_rate'])
            avg_completion = np.mean(completion_list[-10:]) if len(completion_list) > 0 else 0
            print(f"  Avg Completion: {avg_completion:.2%}")
        
        if self.stats['expert_match_rate']:
            match_list = list(self.stats['expert_match_rate'])
            avg_match = np.mean(match_list[-10:]) if len(match_list) > 0 else 0
            print(f"  Expert Match Rate: {avg_match:.2%}")
        
        if self.stats['bc_loss']:
            bc_loss_list = list(self.stats['bc_loss'])
            if bc_loss_list:
                avg_bc_loss = np.mean(bc_loss_list[-10:])
                print(f"  BC Loss: {avg_bc_loss:.4f}")
        
        # Log to wandb if available
        if self.use_wandb:
            log_dict = {
                'update': update,
                'expert_prob': self.expert_prob
            }
            if self.stats['ppo_rewards']:
                log_dict['reward'] = avg_reward
            if self.stats['completion_rate']:
                log_dict['completion_rate'] = avg_completion
            if self.stats['expert_match_rate']:
                log_dict['expert_match_rate'] = avg_match
            wandb.log(log_dict)
    
    def _print_final_stats(self):
        """Print final training statistics."""
        print("\n" + "="*60)
        print("FINAL TRAINING STATISTICS")
        print("="*60)
        
        if self.stats['ppo_rewards']:
            rewards = list(self.stats['ppo_rewards'])
            print(f"Final Avg Reward: {np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards):.2f}")
            print(f"Best Reward: {max(rewards):.2f}")
        
        if self.stats['completion_rate']:
            completions = list(self.stats['completion_rate'])
            print(f"Final Avg Completion: {np.mean(completions[-20:]) if len(completions) >= 20 else np.mean(completions):.2%}")
            print(f"Best Completion: {max(completions):.2%}")
        
        print("="*60)