"""
Imitation learning trainer for UNIQ-RL.
Combines behavioral cloning with online RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import wandb

from expert_policy.uniq_expert import UNIQExpertPolicy
from rewards.enhanced_rewards import EnhancedRewardFunction

class ImitationLearningTrainer:
    """
    Trainer that combines imitation learning with PPO.
    """
    
    def __init__(self, env, policy_net, value_net, ppo_algorithm, config: Dict):
        """
        Initialize imitation learning trainer.
        
        Args:
            env: DQC environment
            policy_net: Policy network
            value_net: Value network
            ppo_algorithm: PPO instance
            config: Training configuration
        """
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
        self.expert_prob = config.get('expert_prob', 0.3)  # Probability of using expert action
        
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
        
        # Initialize wandb if configured
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="uniq-rl",
                config=config,
                name=config.get('run_name', 'imitation_learning')
            )
    
    def generate_expert_demonstrations(self, num_episodes: int = 10):
        """
        Generate expert demonstrations for behavioral cloning.
        
        Args:
            num_episodes: Number of expert episodes to generate
        """
        print(f"Generating {num_episodes} expert demonstrations...")
        
        for episode in range(num_episodes):
            trajectory = self.expert.generate_expert_trajectory(self.env)
            
            # Add to expert buffer
            for transition in trajectory:
                self.expert_buffer.append(transition)
            
            # Calculate episode statistics
            total_reward = sum(t['reward'] for t in trajectory)
            completion = self.env._get_completion_rate()
            
            print(f"  Episode {episode + 1}: Length={len(trajectory)}, "
                  f"Reward={total_reward:.2f}, Completion={completion:.2%}")
        
        print(f"Total expert transitions: {len(self.expert_buffer)}")
    
    def behavioral_cloning_update(self):
        """
        Perform behavioral cloning update on expert demonstrations.
        """
        if len(self.expert_buffer) < self.bc_batch_size:
            return
        
        total_loss = 0
        num_updates = 0
        
        for epoch in range(self.bc_epochs):
            # Sample batch from expert buffer
            batch_indices = np.random.choice(
                len(self.expert_buffer),
                size=min(self.bc_batch_size, len(self.expert_buffer)),
                replace=False
            )
            
            batch_loss = 0
            
            for idx in batch_indices:
                transition = self.expert_buffer[idx]
                state = transition['state']
                expert_action = transition['action']
                valid_actions = transition['valid_actions']
                
                # Skip if no valid actions
                if not valid_actions['map'] and not valid_actions['schedule']:
                    continue
                
                # Get policy output
                action_probs, _, action_info = self.policy_net.forward(state, valid_actions)
                
                # Find expert action in action list
                expert_idx = None
                for i, (act_type, act_params) in enumerate(action_info):
                    if act_type == expert_action[0] and act_params == expert_action[1]:
                        expert_idx = i
                        break
                
                if expert_idx is not None:
                    # Calculate cross-entropy loss
                    if isinstance(action_probs, torch.Tensor):
                        if action_probs.dim() == 0:
                            action_probs = action_probs.unsqueeze(0)
                        
                        # Create target distribution (one-hot for expert action)
                        target = torch.zeros_like(action_probs)
                        target[expert_idx] = 1.0
                        
                        # Cross-entropy loss
                        loss = -torch.sum(target * torch.log(action_probs + 1e-8))
                        
                        # Backward pass
                        self.bc_optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                        self.bc_optimizer.step()
                        
                        batch_loss += loss.item()
                        num_updates += 1
            
            if num_updates > 0:
                total_loss += batch_loss / num_updates
        
        if num_updates > 0:
            avg_loss = total_loss / self.bc_epochs
            self.stats['bc_loss'].append(avg_loss)
            return avg_loss
        
        return 0.0
    
    def collect_mixed_rollouts(self, n_steps: int):
        """
        Collect rollouts with mixture of learned and expert policy.
        
        Args:
            n_steps: Number of steps to collect
        """
        self.ppo.rollout_buffer.reset()
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        expert_actions_used = 0
        
        for step in range(n_steps):
            state_repr = self.env.state.get_state_representation()
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                break
            
            # Decide whether to use expert or learned policy
            use_expert = np.random.random() < self.expert_prob
            
            if use_expert:
                # Get expert action
                action = self.expert.get_expert_action(state_repr, valid_actions)
                expert_actions_used += 1
                
                # Calculate what the policy would have done for logging
                with torch.no_grad():
                    _, log_prob = self.policy_net.get_action(state_repr, valid_actions)
                    value = self.value_net(state_repr)
            else:
                # Get action from learned policy
                with torch.no_grad():
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
            
            # Convert tensors to scalars
            value_scalar = value.item() if isinstance(value, torch.Tensor) else float(value)
            log_prob_scalar = log_prob.item() if isinstance(log_prob, torch.Tensor) else float(log_prob)
            
            # Store transition - use the correct method signature
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
        
        # Track expert usage rate
        if episode_length > 0:
            self.stats['expert_match_rate'].append(expert_actions_used / episode_length)
    
    def train(self, total_timesteps: int):
        """
        Main training loop combining imitation learning and RL.
        
        Args:
            total_timesteps: Total training timesteps
        """
        print("="*60)
        print("UNIQ-RL TRAINING WITH IMITATION LEARNING")
        print("="*60)
        
        # Phase 1: Generate expert demonstrations
        print("\nPhase 1: Generating expert demonstrations...")
        self.generate_expert_demonstrations(num_episodes=20)
        
        # Phase 2: Behavioral cloning pretraining
        print("\nPhase 2: Behavioral cloning pretraining...")
        for epoch in range(20):
            bc_loss = self.behavioral_cloning_update()
            if epoch % 5 == 0:
                print(f"  BC Epoch {epoch}: Loss={bc_loss:.4f}")
        
        # Phase 3: Mixed training with PPO
        print("\nPhase 3: Mixed training with PPO...")
        
        num_updates = total_timesteps // self.ppo.rollout_length
        
        # Gradually decrease expert probability
        initial_expert_prob = self.expert_prob
        
        for update in range(num_updates):
            # Decay expert probability
            progress = update / num_updates
            self.expert_prob = initial_expert_prob * (1 - progress * 0.8)  # Decay to 20% of initial
            
            # Collect mixed rollouts
            self.collect_mixed_rollouts(self.ppo.rollout_length)
            
            # PPO update
            if len(self.ppo.rollout_buffer) > 0:
                ppo_stats = self.ppo.update()
            
            # Periodic behavioral cloning
            if update % 10 == 0 and len(self.expert_buffer) > 0:
                bc_loss = self.behavioral_cloning_update()
            
            # Logging
            if update % 10 == 0:
                self._log_training_stats(update, num_updates)
        
        print("\nTraining completed!")
        self._print_final_stats()
    
    def _log_training_stats(self, update: int, total_updates: int):
        """Log training statistics."""
        print(f"\nUpdate {update}/{total_updates} (Expert prob: {self.expert_prob:.2f})")
        
        if self.stats['ppo_rewards']:
            # Convert deque to list for slicing
            rewards_list = list(self.stats['ppo_rewards'])
            avg_reward = np.mean(rewards_list[-10:]) if len(rewards_list) > 0 else 0
            print(f"  Avg Reward: {avg_reward:.2f}")
        else:
            avg_reward = 0
        
        if self.stats['completion_rate']:
            # Convert deque to list for slicing
            completion_list = list(self.stats['completion_rate'])
            avg_completion = np.mean(completion_list[-10:]) if len(completion_list) > 0 else 0
            print(f"  Avg Completion: {avg_completion:.2%}")
        else:
            avg_completion = 0
        
        if self.stats['expert_match_rate']:
            # Convert deque to list for slicing
            match_list = list(self.stats['expert_match_rate'])
            avg_match = np.mean(match_list[-10:]) if len(match_list) > 0 else 0
            print(f"  Expert Match Rate: {avg_match:.2%}")
        else:
            avg_match = 0
        
        if self.stats['bc_loss']:
            # Convert deque to list for slicing
            bc_loss_list = list(self.stats['bc_loss'])
            avg_bc_loss = np.mean(bc_loss_list[-10:]) if len(bc_loss_list) > 0 else 0
            print(f"  BC Loss: {avg_bc_loss:.4f}")
        else:
            avg_bc_loss = 0
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'reward': avg_reward,
                'completion_rate': avg_completion,
                'expert_match_rate': avg_match,
                'bc_loss': avg_bc_loss,
                'expert_prob': self.expert_prob,
                'update': update
            })
    
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