"""
Imitation learning trainer with detailed debugging for action matching.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque, Counter
import time
import json

# Import expert policy with debug
from expert_policy.uniq_expert import UNIQExpertPolicy
from rewards.enhanced_rewards import EnhancedRewardFunction

# Try to import performance monitor
try:
    from utils.performance_monitor import perf_monitor, profile
    PERF_MONITOR_AVAILABLE = True
except ImportError:
    PERF_MONITOR_AVAILABLE = False
    def profile(name=None):
        def decorator(func):
            return func
        return decorator
    
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

class ImitationLearningTrainer:
    """
    Trainer with extensive debugging for action matching issues.
    """
    
    def __init__(self, env, policy_net, value_net, ppo_algorithm, config: Dict):
        """Initialize with debug capabilities."""
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.ppo = ppo_algorithm
        self.config = config
        
        # Enable debug mode
        self.debug_mode = config.get('debug_mode', True)
        self.debug_frequency = config.get('debug_frequency', 100)  # Log every N steps
        
        # Expert policy with debug
        self.expert = UNIQExpertPolicy(env, debug=False)  # Set to True for very detailed logs
        
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
            lr=float(config.get('bc_lr', 3e-4))
        )
        
        # Training statistics
        self.stats = {
            'bc_loss': deque(maxlen=100),
            'expert_match_rate': deque(maxlen=100),
            'ppo_rewards': deque(maxlen=100),
            'completion_rate': deque(maxlen=100)
        }
        
        # Debug statistics
        self.debug_stats = {
            'action_matches': [],
            'action_mismatches': [],
            'expert_actions': Counter(),
            'policy_actions': Counter(),
            'bc_match_details': [],
            'valid_action_sizes': []
        }
        
        # Performance monitoring
        self.enable_profiling = config.get('enable_profiling', False) and PERF_MONITOR_AVAILABLE
        if self.enable_profiling:
            perf_monitor.enabled = True
    
    def generate_expert_demonstrations(self, num_episodes: int = 10):
        """Generate expert demonstrations with detailed logging."""
        print(f"Generating {num_episodes} expert demonstrations...")
        
        for episode in range(num_episodes):
            episode_start = time.time()
            trajectory = self._generate_single_expert_trajectory_debug(episode)
            
            # Preprocess and add to buffer
            self._preprocess_and_store_trajectory(trajectory)
            
            # Calculate episode statistics
            total_reward = sum(t['reward'] for t in trajectory)
            completion = self.env._get_completion_rate()
            episode_time = time.time() - episode_start
            
            print(f"  Episode {episode + 1}: Length={len(trajectory)}, "
                  f"Reward={total_reward:.2f}, Completion={completion:.2%}, "
                  f"Time={episode_time:.2f}s")
            
            # Debug: Show action distribution
            if self.debug_mode and episode == 0:
                self._log_trajectory_debug(trajectory)
        
        print(f"Total expert transitions: {len(self.expert_buffer)}")
        
        # Debug: Show expert action distribution
        if self.debug_mode:
            self._log_expert_action_distribution()
    
    def _generate_single_expert_trajectory_debug(self, episode_num: int) -> List[Dict]:
        """Generate trajectory with debug info."""
        trajectory = []
        state = self.env.reset()
        max_steps = 1000
        
        for step in range(max_steps):
            state_repr = self.env.state.get_state_representation()
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                break
            
            # Get expert action with debug
            expert_action = self.expert.get_expert_action(state_repr, valid_actions)
            
            # Debug: Log first few actions in detail
            if self.debug_mode and episode_num == 0 and step < 5:
                # print(f"\n[DEBUG Episode {episode_num} Step {step}]")
                # print(f"  Expert action: {expert_action}")
                # print(f"  Valid actions: map={len(valid_actions['map'])}, "
                #       f"schedule={len(valid_actions['schedule'])}")
                
                # Get policy action for comparison
                with torch.no_grad():
                    policy_action, _ = self.policy_net.get_action(state_repr, valid_actions)
                    print(f"  Policy would choose: {policy_action}")
                    
                    # Check if they match
                    match = (expert_action[0] == policy_action[0] and 
                            expert_action[1] == policy_action[1])
                    print(f"  Actions match: {match}")
            
            # Record action in stats
            self.debug_stats['expert_actions'][str(expert_action)] += 1
            
            next_state, reward, done, info = self.env.step(expert_action)
            
            trajectory.append({
                'state': state_repr,
                'action': expert_action,
                'reward': reward,
                'next_state': self.env.state.get_state_representation(),
                'done': done,
                'valid_actions': valid_actions,
                'step': step  # Add step for debugging
            })
            
            if done:
                break
            
            state = next_state
        
        return trajectory
    
    def _log_trajectory_debug(self, trajectory: List[Dict]):
        """Log detailed trajectory information."""
        print("\n[DEBUG] Trajectory Analysis:")
        
        # Action type distribution
        map_actions = sum(1 for t in trajectory if t['action'][0] == 'map')
        schedule_actions = sum(1 for t in trajectory if t['action'][0] == 'schedule')
        
        print(f"  Total steps: {len(trajectory)}")
        print(f"  Map actions: {map_actions} ({map_actions/len(trajectory)*100:.1f}%)")
        print(f"  Schedule actions: {schedule_actions} ({schedule_actions/len(trajectory)*100:.1f}%)")
        
        # First few actions
        print("\n  First 5 actions:")
        for i, t in enumerate(trajectory[:5]):
            print(f"    Step {i}: {t['action']}")
    
    def _log_expert_action_distribution(self):
        """Log expert action distribution."""
        print("\n[DEBUG] Expert Action Distribution:")
        
        # Sort by frequency
        sorted_actions = sorted(self.debug_stats['expert_actions'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        # Show top actions
        print("  Top 10 expert actions:")
        for action_str, count in sorted_actions[:10]:
            print(f"    {action_str}: {count} times")
    
    def behavioral_cloning_update(self):
        """BC update with detailed debugging."""
        if len(self.expert_buffer) < self.bc_batch_size:
            return 0.0
        
        total_loss = 0
        num_batches = 0
        match_count = 0
        total_count = 0
        
        for epoch in range(self.bc_epochs):
            # Sample batch
            batch = self._sample_bc_batch(self.bc_batch_size)
            
            if batch is None:
                continue
            
            # Process batch with debugging
            loss, matches, total = self._compute_bc_loss_batch_debug(batch)
            
            if loss is not None:
                # Backward pass
                self.bc_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.bc_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                match_count += matches
                total_count += total
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            self.stats['bc_loss'].append(avg_loss)
            
            # Log matching rate
            if total_count > 0:
                match_rate = match_count / total_count
                print(f"[DEBUG BC] Match rate in batch: {match_rate:.2%} ({match_count}/{total_count})")
            
            return avg_loss
        
        return 0.0
    
    def _compute_bc_loss_batch_debug(self, batch: List[Dict]) -> Tuple[Optional[torch.Tensor], int, int]:
        """Compute BC loss with detailed debugging."""
        losses = []
        match_count = 0
        total_count = 0
        
        for item in batch[:5] if self.debug_mode else batch:  # Debug first 5 items
            state = item['state']
            valid_actions = item['valid_actions']
            expert_action = item['action']
            
            # Get policy output
            action_probs, _, action_info = self.policy_net.forward(state, valid_actions)
            
            # Debug: Check if expert action is in action_info
            expert_found = False
            expert_idx = None
            
            for idx, (act_type, act_params) in enumerate(action_info):
                if act_type == expert_action[0] and act_params == expert_action[1]:
                    expert_found = True
                    expert_idx = idx
                    break
            
            # if self.debug_mode and total_count < 5:  # Log first few
            #     print(f"\n[DEBUG BC Item {total_count}]")
            #     print(f"  Expert action: {expert_action}")
            #     print(f"  Action info length: {len(action_info)}")
            #     print(f"  Expert found: {expert_found}")
            #     if expert_found:
            #         print(f"  Expert index: {expert_idx}")
            #         if isinstance(action_probs, torch.Tensor) and action_probs.numel() > expert_idx:
            #             print(f"  Expert probability: {action_probs[expert_idx].item():.4f}")
            #     else:
            #         print(f"  First 3 actions in info: {action_info[:3]}")
            
            if expert_found:
                match_count += 1
                
                # Compute cross-entropy loss
                if isinstance(action_probs, torch.Tensor):
                    if action_probs.dim() == 0:
                        action_probs = action_probs.unsqueeze(0)
                    
                    # Create target distribution
                    target = torch.zeros_like(action_probs)
                    target[expert_idx] = 1.0
                    
                    # Cross-entropy loss
                    loss = -torch.sum(target * torch.log(action_probs + 1e-8))
                    losses.append(loss)
            
            total_count += 1
        
        # Store debug stats
        self.debug_stats['bc_match_details'].append({
            'match_count': match_count,
            'total_count': total_count,
            'match_rate': match_count / total_count if total_count > 0 else 0
        })
        
        if losses:
            return torch.mean(torch.stack(losses)), match_count, total_count
        return None, match_count, total_count
    
    def collect_mixed_rollouts(self, n_steps: int):
        """Collect rollouts with detailed action matching debug."""
        self.ppo.rollout_buffer.reset()
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        expert_actions_used = 0
        action_matches = 0
        
        for step in range(n_steps):
            state_repr = self.env.state.get_state_representation()
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                break
            
            # Record valid action sizes
            self.debug_stats['valid_action_sizes'].append({
                'map': len(valid_actions['map']),
                'schedule': len(valid_actions['schedule'])
            })
            
            # Get both expert and policy actions for comparison
            expert_action = self.expert.get_expert_action(state_repr, valid_actions)
            
            # Decide which to use
            use_expert = np.random.random() < self.expert_prob
            
            if use_expert:
                action = expert_action
                expert_actions_used += 1
                
                with torch.no_grad():
                    _, log_prob = self.policy_net.get_action(state_repr, valid_actions)
                    value = self.value_net(state_repr)
            else:
                with torch.no_grad():
                    action, log_prob = self.policy_net.get_action(
                        state_repr, valid_actions, deterministic=False
                    )
                    value = self.value_net(state_repr)
                
                # Check if policy matches expert
                if action[0] == expert_action[0] and action[1] == expert_action[1]:
                    action_matches += 1
            
            # Debug logging
            # if self.debug_mode and step % self.debug_frequency == 0:
            #     print(f"\n[DEBUG Rollout Step {step}]")
            #     print(f"  Expert action: {expert_action}")
            #     print(f"  Policy action: {action if not use_expert else 'N/A (using expert)'}")
            #     print(f"  Action match: {action[0] == expert_action[0] and action[1] == expert_action[1]}")
            
            # Execute action
            next_state, base_reward, done, info = self.env.step(action)
            
            # Calculate enhanced reward
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
                action_matches = 0
            else:
                state = next_state
        
        if episode_length > 0:
            match_rate = action_matches / episode_length if not use_expert else expert_actions_used / episode_length
            self.stats['expert_match_rate'].append(match_rate)
    
    def train(self, total_timesteps: int):
        """Training with comprehensive debugging."""
        print("="*60)
        print("DEBUG IMITATION LEARNING TRAINING")
        print("="*60)
        
        overall_start = time.time()
        
        # Phase 1: Generate expert demonstrations
        print("\nPhase 1: Generating expert demonstrations...")
        self.generate_expert_demonstrations(num_episodes=5 if self.debug_mode else 20)
        
        # Phase 2: Behavioral cloning pretraining
        print("\nPhase 2: Behavioral cloning pretraining...")
        for epoch in range(20):
            bc_loss = self.behavioral_cloning_update()
            if epoch % 5 == 0:
                print(f"  BC Epoch {epoch}: Loss={bc_loss:.4f}")
        
        # Print BC matching statistics
        if self.debug_mode and self.debug_stats['bc_match_details']:
            avg_match_rate = np.mean([d['match_rate'] for d in self.debug_stats['bc_match_details']])
            print(f"\n[DEBUG] Average BC match rate: {avg_match_rate:.2%}")
        
        # Phase 3: Mixed training with PPO
        print("\nPhase 3: Mixed training with PPO...")
        num_updates = total_timesteps // self.ppo.rollout_length
        initial_expert_prob = self.expert_prob
        
        for update in range(min(5, num_updates) if self.debug_mode else num_updates):
            # Decay expert probability
            progress = update / num_updates
            self.expert_prob = initial_expert_prob * (1 - progress * 0.8)
            
            # Collect rollouts
            self.collect_mixed_rollouts(self.ppo.rollout_length)
            
            # PPO update
            if len(self.ppo.rollout_buffer) > 0:
                ppo_stats = self.ppo.update()
            
            # Periodic BC
            if update % 10 == 0 and len(self.expert_buffer) > 0:
                bc_loss = self.behavioral_cloning_update()
            
            # Logging
            if update % 10 == 0:
                self._log_training_stats(update, num_updates)
        
        overall_time = time.time() - overall_start
        print(f"\nTraining completed in {overall_time:.2f} seconds!")
        
        # Print final debug statistics
        if self.debug_mode:
            self._print_debug_summary()
        
        self._print_final_stats()
    
    def _print_debug_summary(self):
        """Print comprehensive debug summary."""
        print("\n" + "="*60)
        print("DEBUG SUMMARY")
        print("="*60)
        
        # Valid action sizes
        if self.debug_stats['valid_action_sizes']:
            avg_map = np.mean([s['map'] for s in self.debug_stats['valid_action_sizes']])
            avg_schedule = np.mean([s['schedule'] for s in self.debug_stats['valid_action_sizes']])
            print(f"\nAverage valid actions:")
            print(f"  Map: {avg_map:.1f}")
            print(f"  Schedule: {avg_schedule:.1f}")
        
        # BC matching details
        if self.debug_stats['bc_match_details']:
            match_rates = [d['match_rate'] for d in self.debug_stats['bc_match_details']]
            print(f"\nBC Matching:")
            print(f"  Mean match rate: {np.mean(match_rates):.2%}")
            print(f"  Std match rate: {np.std(match_rates):.2%}")
            print(f"  Min match rate: {np.min(match_rates):.2%}")
            print(f"  Max match rate: {np.max(match_rates):.2%}")
    
    def _log_training_stats(self, update: int, total_updates: int):
        """Enhanced logging with debug info."""
        print(f"\nUpdate {update}/{total_updates} (Expert prob: {self.expert_prob:.2f})")
        
        # Standard metrics
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
    
    def _sample_bc_batch(self, batch_size: int) -> Optional[List[Dict]]:
        """Sample a batch for BC training."""
        if len(self.expert_buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.expert_buffer), batch_size, replace=False)
        return [self.expert_buffer[i] for i in indices]
    
    def _preprocess_and_store_trajectory(self, trajectory: List[Dict]):
        """Store trajectory in buffer."""
        for transition in trajectory:
            self.expert_buffer.append(transition)