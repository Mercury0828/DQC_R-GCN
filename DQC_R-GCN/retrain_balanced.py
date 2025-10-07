import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from collections import deque

from data.quantum_problem_generator import QuantumProblemGenerator
from environment import DQCEnvironment
from models import RGCN, PolicyNetwork, ValueNetwork
from algorithms.ppo import PPO
from expert_policy.uniq_expert import UNIQExpertPolicy

class FixedBalancedTrainer:
    """修复的平衡训练器"""
    
    def __init__(self, env, policy_net, value_net, ppo, expert):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.ppo = ppo
        self.expert = expert
        
        # 动作统计
        self.action_stats = deque(maxlen=100)
        self.recent_map_ratio = 0.5
        
    def smart_action_selection(self, state_repr, valid_actions):
        """智能动作选择，平衡map和schedule"""
        
        # 获取专家建议
        expert_action = self.expert.get_expert_action(state_repr, valid_actions)
        
        # 获取模型动作
        with torch.no_grad():
            model_action, log_prob = self.policy_net.get_action(
                state_repr, valid_actions, deterministic=False
            )
        
        # 如果专家和模型都选择有效动作
        if expert_action[0] != 'noop' and model_action[0] != 'noop':
            # 计算最近的map比例
            if len(self.action_stats) > 10:
                recent_maps = sum(1 for a in list(self.action_stats)[-20:] if a == 'map')
                self.recent_map_ratio = recent_maps / min(20, len(self.action_stats))
            
            # 如果map过多(>70%)，优先选择schedule
            if self.recent_map_ratio > 0.7:
                if expert_action[0] == 'schedule':
                    return expert_action, log_prob, True
                elif model_action[0] == 'schedule':
                    return model_action, log_prob, False
            
            # 如果schedule过多(<30% map)，优先选择map
            elif self.recent_map_ratio < 0.3:
                if expert_action[0] == 'map':
                    return expert_action, log_prob, True
                elif model_action[0] == 'map':
                    return model_action, log_prob, False
            
            # 否则，70%概率用模型，30%用专家
            if np.random.random() < 0.7:
                return model_action, log_prob, False
            else:
                return expert_action, log_prob, True
        
        # 如果只有一个有效，选择有效的
        if expert_action[0] != 'noop':
            return expert_action, log_prob, True
        elif model_action[0] != 'noop':
            return model_action, log_prob, False
        else:
            return ('noop', ()), log_prob, False
    
    def train_step(self, num_episodes=10):
        """训练一步"""
        
        total_rewards = []
        completions = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_actions = []
            
            for step in range(200):
                state_repr = self.env.state.get_state_representation()
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions['map'] and not valid_actions['schedule']:
                    break
                
                # 智能选择动作
                action, log_prob, is_expert = self.smart_action_selection(
                    state_repr, valid_actions
                )
                
                if action[0] == 'noop':
                    break
                
                # 记录动作类型
                episode_actions.append(action[0])
                self.action_stats.append(action[0])
                
                # 获取价值估计
                with torch.no_grad():
                    value = self.value_net(state_repr)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 修改奖励
                # 1. 基础奖励
                modified_reward = reward
                
                # 2. 动作平衡奖励
                if len(episode_actions) > 5:
                    recent_5 = episode_actions[-5:]
                    map_count = sum(1 for a in recent_5 if a == 'map')
                    if map_count == 5:  # 连续5个map
                        modified_reward -= 2.0
                    elif map_count == 0:  # 连续5个schedule
                        modified_reward -= 1.0
                
                # 3. 完成度奖励
                if done:
                    completion = self.env._get_completion_rate()
                    if completion == 1.0:
                        modified_reward += 50
                    else:
                        modified_reward += completion * 20
                
                # 存储经验
                self.ppo.rollout_buffer.add(
                    state=state_repr,
                    action=action,
                    reward=modified_reward,
                    value=value.item() if isinstance(value, torch.Tensor) else value,
                    log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                    done=done,
                    valid_actions=valid_actions
                )
                
                episode_reward += reward  # 使用原始奖励统计
                
                if done:
                    break
                
                state = next_state
            
            total_rewards.append(episode_reward)
            completions.append(self.env._get_completion_rate())
            
            # 计算动作分布
            if episode_actions:
                map_ratio = sum(1 for a in episode_actions if a == 'map') / len(episode_actions)
                if episode % 3 == 0:
                    print(f"  Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Completion={self.env._get_completion_rate():.2%}, "
                          f"Map ratio={map_ratio:.2%}")
        
        # PPO更新
        if len(self.ppo.rollout_buffer) > 0:
            update_stats = self.ppo.update()
        
        return np.mean(total_rewards), np.mean(completions)

def main():
    """主训练函数"""
    
    # 加载配置
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建环境
    problem_config = config['problem']['small']
    generator = QuantumProblemGenerator(**problem_config, seed=42)
    generator.generate_all_parameters()
    problem_data = generator.get_model_data()
    env = DQCEnvironment(problem_data, {'max_steps': 200})
    
    # 创建模型
    model_config = config['model']
    rgcn = RGCN(model_config)
    policy_net = PolicyNetwork(rgcn, model_config)
    value_net = ValueNetwork(rgcn, model_config)
    
    # 初始化模型
    env.reset()
    state_repr = env.state.get_state_representation()
    _ = policy_net.rgcn(state_repr)
    
    # 加载已有模型（如果存在）
    if Path('best_model.pt').exists():
        print("Loading existing model...")
        checkpoint = torch.load('best_model.pt', map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy_state'])
        value_net.load_state_dict(checkpoint['value_state'])
        print("Model loaded successfully")
    
    # 创建优化器
    ppo_config = config['ppo']
    ppo_config['learning_rate'] = 5e-6  # 很小的学习率
    ppo = PPO(policy_net, value_net, ppo_config)
    
    # 创建专家和训练器
    expert = UNIQExpertPolicy(env, debug=False)
    trainer = FixedBalancedTrainer(env, policy_net, value_net, ppo, expert)
    
    print("\nStarting balanced retraining...")
    print("="*60)
    
    best_reward = -float('inf')
    best_completion = 0
    
    for iteration in range(50):
        print(f"\nIteration {iteration+1}/50")
        
        # 训练
        avg_reward, avg_completion = trainer.train_step(num_episodes=10)
        
        print(f"Average: Reward={avg_reward:.2f}, Completion={avg_completion:.2%}")
        
        # 保存最佳模型
        if avg_reward > best_reward or avg_completion > best_completion:
            if avg_completion >= best_completion:  # 优先考虑完成率
                best_reward = avg_reward
                best_completion = avg_completion
                
                torch.save({
                    'policy_state': policy_net.state_dict(),
                    'value_state': value_net.state_dict(),
                    'iteration': iteration,
                    'avg_reward': avg_reward,
                    'avg_completion': avg_completion
                }, 'balanced_model.pt')
                
                print(f"  ✓ Saved new best model!")
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Best: Reward={best_reward:.2f}, Completion={best_completion:.2%}")

if __name__ == "__main__":
    main()