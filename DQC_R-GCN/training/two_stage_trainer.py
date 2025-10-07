# training/two_stage_trainer.py
import torch
import numpy as np
from collections import deque
from typing import Dict, List
import time

class TwoStageTrainer:
    """两阶段训练器：先学placement，再学完整策略"""
    
    def __init__(self, env, policy_net, value_net, ppo_algorithm, config: Dict):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.ppo = ppo_algorithm
        self.config = config
        
        # 使用增强的奖励函数
        from rewards.enhanced_rewards_v2 import EnhancedRewardFunctionV2
        reward_config = config.get('reward_config', {})
        self.reward_fn = EnhancedRewardFunctionV2(reward_config)
        
        # 专家策略
        from expert_policy.uniq_expert import UNIQExpertPolicy
        self.expert = UNIQExpertPolicy(env, debug=False)
        
        # 训练统计
        self.stats = {
            'placement_quality': deque(maxlen=100),
            'communication_cost': deque(maxlen=100),
            'bc_loss': deque(maxlen=100),
            'completion_rate': deque(maxlen=100),
        }
        
        # BC optimizer
        self.bc_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=float(config.get('bc_lr', 5e-4))
        )
    
    def stage1_placement_training(self, num_episodes: int = 200):
        """阶段1：只训练placement（mapping）"""
        print("\n" + "="*60)
        print("STAGE 1: PLACEMENT TRAINING")
        print("="*60)
        
        # 收集专家的placement演示
        print("\nCollecting expert placement demonstrations...")
        placement_buffer = []
        
        for episode in range(50):
            self.env.reset()
            episode_mappings = []
            
            # 只收集mapping动作
            for step in range(self.env.state.num_qubits):
                state_repr = self.env.state.get_state_representation()
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions['map']:
                    break
                
                # 获取专家动作
                expert_action = self.expert.get_expert_action(state_repr, valid_actions)
                
                if expert_action[0] == 'map':
                    episode_mappings.append({
                        'state': state_repr,
                        'action': expert_action,
                        'valid_actions': valid_actions
                    })
                    
                    # 执行动作
                    self.env.step(expert_action)
            
            placement_buffer.extend(episode_mappings)
            
            if episode % 10 == 0:
                print(f"  Collected {len(placement_buffer)} mapping examples")
        
        print(f"\nTotal placement examples: {len(placement_buffer)}")
        
        # 训练placement策略
        print("\nTraining placement policy...")
        
        for epoch in range(num_episodes):
            # 批量训练
            batch_size = 32
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # 随机采样
            indices = np.random.permutation(len(placement_buffer))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_loss = 0
                
                for idx in batch_indices:
                    sample = placement_buffer[idx]
                    state = sample['state']
                    expert_action = sample['action']
                    valid_actions = sample['valid_actions']
                    
                    # Forward pass
                    action_probs, _, action_info = self.policy_net.forward(state, valid_actions)
                    
                    # 找到专家动作的索引
                    expert_idx = None
                    for j, (act_type, act_params) in enumerate(action_info):
                        if act_type == expert_action[0] and act_params == expert_action[1]:
                            expert_idx = j
                            break
                    
                    if expert_idx is not None:
                        # 计算交叉熵损失
                        if isinstance(action_probs, torch.Tensor):
                            if action_probs.dim() == 0:
                                action_probs = action_probs.unsqueeze(0)
                            
                            # 创建目标分布
                            target = torch.zeros_like(action_probs)
                            target[expert_idx] = 1.0
                            
                            # 损失
                            loss = -torch.sum(target * torch.log(action_probs + 1e-8))
                            batch_loss += loss
                            
                            # 统计准确率
                            predicted_idx = torch.argmax(action_probs).item()
                            if predicted_idx == expert_idx:
                                correct_predictions += 1
                            total_predictions += 1
                
                # 更新
                if batch_loss > 0:
                    self.bc_optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                    self.bc_optimizer.step()
                    
                    total_loss += batch_loss.item()
            
            # 打印进度
            if epoch % 20 == 0:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"  Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={accuracy:.2%}")
                
                # 测试placement质量
                if epoch % 40 == 0:
                    self._test_placement_quality()
    
    def stage2_full_training(self, timesteps: int = 50000):
        """阶段2：训练完整策略"""
        print("\n" + "="*60)
        print("STAGE 2: FULL POLICY TRAINING")
        print("="*60)
        
        num_updates = timesteps // self.ppo.rollout_length
        
        # 使用较高的初始专家概率
        expert_prob = 0.7
        expert_decay = 0.95  # 缓慢衰减
        
        for update in range(num_updates):
            # 收集rollouts（混合专家和学习策略）
            self.ppo.rollout_buffer.reset()
            
            for _ in range(self.ppo.rollout_length):
                state_repr = self.env.state.get_state_representation()
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions['map'] and not valid_actions['schedule']:
                    self.env.reset()
                    continue
                
                # 决定使用专家还是策略
                use_expert = np.random.random() < expert_prob
                
                if use_expert:
                    action = self.expert.get_expert_action(state_repr, valid_actions)
                    
                    # 计算策略的log_prob
                    with torch.no_grad():
                        _, log_prob = self.policy_net.get_action(state_repr, valid_actions)
                        value = self.value_net(state_repr)
                else:
                    with torch.no_grad():
                        action, log_prob = self.policy_net.get_action(
                            state_repr, valid_actions, deterministic=False
                        )
                        value = self.value_net(state_repr)
                
                # 执行动作
                next_state, base_reward, done, info = self.env.step(action)
                
                # 使用增强奖励
                expert_action = self.expert.get_expert_action(state_repr, valid_actions)
                reward = self.reward_fn.calculate_reward(
                    self.env, action, info, expert_action
                )
                
                # 存储transition
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
                    # 记录统计
                    self.stats['completion_rate'].append(self.env._get_completion_rate())
                    comm_cost = self._calculate_communication_cost()
                    self.stats['communication_cost'].append(comm_cost)
                    
                    self.env.reset()
            
            # PPO更新
            if len(self.ppo.rollout_buffer) > 0:
                ppo_stats = self.ppo.update()
            
            # 衰减专家概率
            expert_prob *= expert_decay
            expert_prob = max(0.3, expert_prob)  # 最低保持30%
            
            # 打印进度
            if update % 10 == 0:
                self._print_training_stats(update, num_updates, expert_prob)
    
    def _test_placement_quality(self):
        """测试当前placement策略的质量"""
        self.env.reset()
        
        # 执行完整的mapping
        for _ in range(self.env.state.num_qubits):
            state_repr = self.env.state.get_state_representation()
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions['map']:
                break
            
            with torch.no_grad():
                action, _ = self.policy_net.get_action(
                    state_repr, valid_actions, deterministic=True
                )
            
            if action[0] == 'map':
                self.env.step(action)
        
        # 计算通信成本
        comm_cost = self._calculate_communication_cost()
        quality = 1.0 - min(1.0, comm_cost / 50.0)  # 归一化
        
        self.stats['placement_quality'].append(quality)
        print(f"    Placement test: Communication cost={comm_cost:.2f}, Quality={quality:.2%}")
    
    def _calculate_communication_cost(self):
        """计算当前mapping的通信成本"""
        total_cost = 0
        
        for g_id, gate in self.env.problem_data['gates'].items():
            control = gate['control']
            target = gate['target']
            
            if control in self.env.state.qubit_mapping and target in self.env.state.qubit_mapping:
                control_qpu = self.env.state.qubit_mapping[control]
                target_qpu = self.env.state.qubit_mapping[target]
                
                if control_qpu != target_qpu:
                    comm_cost = self.env.problem_data['C'].get((control_qpu, target_qpu), 0)
                    total_cost += comm_cost
        
        return total_cost
    
    def _print_training_stats(self, update, total_updates, expert_prob):
        """打印训练统计"""
        print(f"\nUpdate {update}/{total_updates} (Expert prob: {expert_prob:.2f})")
        
        if self.stats['completion_rate']:
            avg_completion = np.mean(list(self.stats['completion_rate'])[-10:])
            print(f"  Completion rate: {avg_completion:.2%}")
        
        if self.stats['communication_cost']:
            avg_comm = np.mean(list(self.stats['communication_cost'])[-10:])
            print(f"  Avg communication cost: {avg_comm:.2f}")
            
            # 与Expert基准比较（约29.59）
            expert_baseline = 30.0
            ratio = avg_comm / expert_baseline
            if ratio < 1.2:
                print(f"  ✓ Good placement quality (within 20% of expert)")
            elif ratio < 1.5:
                print(f"  ⚠ Moderate placement quality ({ratio:.1f}x expert cost)")
            else:
                print(f"  ❌ Poor placement quality ({ratio:.1f}x expert cost)")