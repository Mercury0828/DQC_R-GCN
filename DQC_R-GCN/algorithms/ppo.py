import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.valid_actions = []
    
    def add(self, state=None, action=None, reward=None, value=None, 
            log_prob=None, done=None, valid_actions=None, **kwargs):
        """Add a transition to the buffer."""
        if state is not None:
            self.states.append(state)
        if action is not None:
            self.actions.append(action)
        if reward is not None:
            self.rewards.append(float(reward))
        if value is not None:
            self.values.append(float(value))
        if log_prob is not None:
            self.log_probs.append(float(log_prob))
        if done is not None:
            self.dones.append(bool(done))
        if valid_actions is not None:
            self.valid_actions.append(valid_actions)
    
    def get(self):
        """Get all data from buffer."""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones,
            'valid_actions': self.valid_actions
        }
    
    def __len__(self):
        return len(self.rewards)


class PPO:
    """Stable PPO implementation with optional KL regularization."""
    
    def __init__(self, policy_net, value_net, config):
        """Initialize PPO algorithm."""
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        
        # Hyperparameters - ensure correct types
        self.lr = float(config.get('learning_rate', 1e-4))
        self.gamma = float(config.get('gamma', 0.99))
        self.gae_lambda = float(config.get('gae_lambda', 0.95))
        self.clip_epsilon = float(config.get('clip_epsilon', 0.2))
        self.value_coef = float(config.get('value_coef', 0.5))
        self.entropy_coef = float(config.get('entropy_coef', 0.01))
        self.max_grad_norm = float(config.get('max_grad_norm', 0.5))
        
        # KL regularization
        self.use_kl_reg = config.get('use_kl_regularization', False)
        self.kl_coef = float(config.get('kl_coef', 0.01))
        
        self.rollout_length = int(config.get('rollout_length', 256))
        self.minibatch_size = int(config.get('minibatch_size', 64))
        self.num_epochs = int(config.get('num_epochs', 10))
        
        # Optimizers
        all_params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.lr)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()
        
        # Training statistics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'total_reward': deque(maxlen=100),
            'episode_length': deque(maxlen=100),
            'kl_div': deque(maxlen=100)
        }
    
    def collect_rollouts(self, env, n_steps: int):
        """Collect rollouts from environment."""
        self.rollout_buffer.reset()
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(n_steps):
            state_repr = env.state.get_state_representation()
            valid_actions = env.get_valid_actions()
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                print(f"Warning: No valid actions at step {step}")
                break
            
            with torch.no_grad():
                action, log_prob = self.policy_net.get_action(
                    state_repr, valid_actions, deterministic=False
                )
                value = self.value_net(state_repr)
                
                value_scalar = value.item() if isinstance(value, torch.Tensor) else float(value)
                log_prob_scalar = log_prob.item() if isinstance(log_prob, torch.Tensor) else float(log_prob)
            
            next_state, reward, done, info = env.step(action)
            
            self.rollout_buffer.add(
                state=state_repr,
                action=action,
                reward=float(reward),
                value=value_scalar,
                log_prob=log_prob_scalar,
                done=bool(done),
                valid_actions=valid_actions
            )
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                self.training_stats['total_reward'].append(episode_reward)
                self.training_stats['episode_length'].append(episode_length)
                
                state = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state
        
        return True
    
    def compute_returns_and_advantages(self):
        """Compute returns and advantages using GAE."""
        data = self.rollout_buffer.get()
        
        if not data['rewards']:
            return torch.tensor([]), torch.tensor([])
        
        rewards = torch.tensor(data['rewards'], dtype=torch.float32)
        values = torch.tensor(data['values'], dtype=torch.float32)
        dones = torch.tensor(data['dones'], dtype=torch.float32)
        
        returns = []
        advantages = []
        
        last_value = 0
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            
            returns.insert(0, advantage + values[t])
            advantages.insert(0, advantage)
            
            last_advantage = advantage
        
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self):
        """Update policy and value networks with safe KL regularization."""
        data = self.rollout_buffer.get()
        
        if len(data['rewards']) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'kl_div': 0}
        
        returns, advantages = self.compute_returns_and_advantages()
        
        if len(returns) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'kl_div': 0}
        
        old_log_probs = torch.tensor(data['log_probs'], dtype=torch.float32)
        
        all_policy_losses = []
        all_value_losses = []
        all_kl_divs = []
        
        for i in range(min(len(data['states']), len(returns))):
            state = data['states'][i]
            action = data['actions'][i]
            old_log_prob = old_log_probs[i]
            return_ = returns[i]
            advantage = advantages[i]
            valid_actions = data['valid_actions'][i]
            
            if not valid_actions['map'] and not valid_actions['schedule']:
                continue
            
            # Get current policy output
            action_probs, _, action_info = self.policy_net.forward(state, valid_actions)
            
            # Compute KL divergence if enabled (safe version)
            kl_div = torch.tensor(0.0)
            if self.use_kl_reg and isinstance(action_probs, torch.Tensor):
                # Store old distribution for KL
                with torch.no_grad():
                    old_probs = action_probs.clone().detach()
                
                # Compute KL only if dimensions match
                if old_probs.shape == action_probs.shape:
                    kl_div = torch.sum(
                        old_probs * (torch.log(old_probs + 1e-10) - torch.log(action_probs + 1e-10))
                    )
                    all_kl_divs.append(kl_div.item())
            
            # Find log prob of taken action
            log_prob = None
            for idx, (act_type, act_params) in enumerate(action_info):
                if act_type == action[0] and act_params == action[1]:
                    if isinstance(action_probs, torch.Tensor):
                        if action_probs.dim() == 0:
                            log_prob = torch.log(action_probs + 1e-10)
                        else:
                            if idx < len(action_probs):
                                log_prob = torch.log(action_probs[idx] + 1e-10)
                    break
            
            if log_prob is not None:
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2)
                
                # Add KL penalty if enabled
                if self.use_kl_reg:
                    policy_loss = policy_loss + self.kl_coef * kl_div
                
                all_policy_losses.append(policy_loss.item())
            
            # Value loss
            value_pred = self.value_net(state)
            if isinstance(value_pred, torch.Tensor):
                value_loss = F.mse_loss(value_pred.squeeze(), return_)
                all_value_losses.append(value_loss.item())
        
        # Update statistics
        results = {}
        
        if all_policy_losses:
            results['policy_loss'] = np.mean(all_policy_losses)
            self.training_stats['policy_loss'].append(results['policy_loss'])
        else:
            results['policy_loss'] = 0
            
        if all_value_losses:
            results['value_loss'] = np.mean(all_value_losses)
            self.training_stats['value_loss'].append(results['value_loss'])
        else:
            results['value_loss'] = 0
        
        if all_kl_divs:
            results['kl_div'] = np.mean(all_kl_divs)
            self.training_stats['kl_div'].append(results['kl_div'])
        else:
            results['kl_div'] = 0
        
        return results