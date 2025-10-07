# training/fast_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List
import time
from tqdm import tqdm  # 添加进度条

class FastPlacementTrainer:
    """优化的快速训练器"""
    
    def __init__(self, env, policy_net, value_net, ppo_algorithm, config: Dict):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.ppo = ppo_algorithm
        self.config = config
        
        # 专家策略
        from expert_policy.uniq_expert import UNIQExpertPolicy
        self.expert = UNIQExpertPolicy(env, debug=False)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=1e-3
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
    def fast_placement_training(self, num_epochs=30):
        """快速placement训练"""
        print("\n" + "="*60)
        print("FAST PLACEMENT TRAINING")
        print("="*60)
        
        # 1. 快速收集专家数据（一次性）
        print("\nCollecting expert demonstrations...")
        expert_data = self._collect_expert_data_fast(num_episodes=20)
        print(f"Collected {len(expert_data)} examples")
        
        # 2. 快速训练
        print("\nTraining placement network...")
        
        # 转换为tensor以加速
        self._prepare_training_data(expert_data)
        
        # 训练循环（使用进度条）
        with tqdm(total=num_epochs, desc="Training") as pbar:
            for epoch in range(num_epochs):
                loss = self._train_epoch_fast(expert_data)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # 偶尔测试
                if epoch % 10 == 0 and epoch > 0:
                    quality = self._quick_test()
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'quality': f'{quality:.2f}'})
        
        print("\nPlacement training completed!")
    
    def _collect_expert_data_fast(self, num_episodes=20):
        """快速收集专家数据"""
        data = []
        
        for ep in range(num_episodes):
            self.env.reset()
            
            # 只收集前12个mapping动作
            for step in range(12):
                state_repr = self.env.state.get_state_representation()
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions['map']:
                    break
                
                # 专家动作
                expert_action = self.expert.get_expert_action(state_repr, valid_actions)
                
                if expert_action[0] == 'map':
                    # 简化存储
                    data.append({
                        'gate_features': state_repr['gate_features'],
                        'qubit_features': state_repr['qubit_features'],
                        'qpu_features': state_repr['qpu_features'],
                        'action': expert_action[1],  # (qubit, qpu)
                        'valid_map': valid_actions['map']
                    })
                    
                    self.env.step(expert_action)
        
        return data
    
    def _prepare_training_data(self, expert_data):
        """预处理数据以加速训练"""
        # 预计算所有特征的tensor版本
        for item in expert_data:
            item['gate_tensor'] = torch.FloatTensor(item['gate_features'])
            item['qubit_tensor'] = torch.FloatTensor(item['qubit_features'])
            item['qpu_tensor'] = torch.FloatTensor(item['qpu_features'])
    
    def _train_epoch_fast(self, data):
        """快速训练一个epoch"""
        total_loss = 0
        batch_size = 32
        
        # 随机打乱
        indices = np.random.permutation(len(data))
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            loss = 0
            
            self.optimizer.zero_grad()
            
            for idx in batch_idx:
                item = data[idx]
                
                # 构建state_repr
                state_repr = {
                    'gate_features': item['gate_features'],
                    'qubit_features': item['qubit_features'],
                    'qpu_features': item['qpu_features'],
                    'edges': {'depends_on': [], 'acts_on': [], 'assigned_to': [], 'communicates_with': []}
                }
                
                # Forward pass
                with torch.set_grad_enabled(True):
                    action_probs, _, action_info = self.policy_net.forward(
                        state_repr, 
                        {'map': item['valid_map'], 'schedule': []}
                    )
                    
                    # 找到专家动作
                    target_action = item['action']
                    target_idx = None
                    
                    for j, (act_type, act_params) in enumerate(action_info):
                        if act_type == 'map' and act_params == target_action:
                            target_idx = j
                            break
                    
                    if target_idx is not None and len(action_info) > 0:
                        # 简化的损失计算
                        if action_probs.dim() == 0:
                            action_probs = action_probs.unsqueeze(0)
                        
                        log_probs = torch.log(action_probs + 1e-8)
                        loss -= log_probs[target_idx]
            
            if loss > 0:
                loss = loss / len(batch_idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
        
        return total_loss / max(1, len(indices) // batch_size)
    
    def _quick_test(self):
        """快速测试placement质量"""
        self.env.reset()
        comm_cost = 0
        
        # 快速完成mapping
        with torch.no_grad():
            for _ in range(12):
                state_repr = self.env.state.get_state_representation()
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions['map']:
                    break
                
                action, _ = self.policy_net.get_action(state_repr, valid_actions, deterministic=True)
                if action[0] == 'map':
                    self.env.step(action)
        
        # 计算通信成本
        for g_id, gate in self.env.problem_data['gates'].items():
            control = gate['control']
            target = gate['target']
            
            if control in self.env.state.qubit_mapping and target in self.env.state.qubit_mapping:
                c_qpu = self.env.state.qubit_mapping[control]
                t_qpu = self.env.state.qubit_mapping[target]
                
                if c_qpu != t_qpu:
                    comm_cost += self.env.problem_data['C'].get((c_qpu, t_qpu), 0)
        
        return comm_cost
    
    def integrated_training(self, total_timesteps=10000):
        """集成训练：placement + scheduling"""
        print("\n" + "="*60)
        print("INTEGRATED FAST TRAINING")
        print("="*60)
        
        num_updates = total_timesteps // self.ppo.rollout_length
        expert_prob = 0.8
        
        with tqdm(total=num_updates, desc="PPO Training") as pbar:
            for update in range(num_updates):
                # 收集rollouts
                self.ppo.rollout_buffer.reset()
                episode_rewards = []
                
                for _ in range(self.ppo.rollout_length):
                    state_repr = self.env.state.get_state_representation()
                    valid_actions = self.env.get_valid_actions()
                    
                    if not valid_actions['map'] and not valid_actions['schedule']:
                        # Episode结束
                        episode_rewards.append(sum(self.ppo.rollout_buffer.rewards) if self.ppo.rollout_buffer.rewards else 0)
                        self.env.reset()
                        continue
                    
                    # 混合策略
                    if np.random.random() < expert_prob:
                        action = self.expert.get_expert_action(state_repr, valid_actions)
                    else:
                        with torch.no_grad():
                            action, _ = self.policy_net.get_action(state_repr, valid_actions)
                    
                    # 执行
                    _, reward, done, _ = self.env.step(action)
                    
                    # 简化的buffer添加
                    with torch.no_grad():
                        value = self.value_net(state_repr)
                        _, log_prob = self.policy_net.get_action(state_repr, valid_actions)
                    
                    self.ppo.rollout_buffer.add(
                        state=state_repr,
                        action=action,
                        reward=reward,
                        value=value.item() if isinstance(value, torch.Tensor) else value,
                        log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                        done=done,
                        valid_actions=valid_actions
                    )
                    
                    if done:
                        self.env.reset()
                
                # PPO更新
                if len(self.ppo.rollout_buffer) > 0:
                    self.ppo.update()
                
                # 衰减专家概率
                expert_prob = max(0.3, expert_prob * 0.99)
                
                # 更新进度条
                pbar.update(1)
                if episode_rewards:
                    pbar.set_postfix({
                        'reward': f'{np.mean(episode_rewards):.2f}',
                        'expert_p': f'{expert_prob:.2f}'
                    })