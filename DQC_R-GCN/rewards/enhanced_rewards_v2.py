# rewards/enhanced_rewards_v2.py
import torch
import numpy as np
from typing import Dict, Tuple, Optional

class EnhancedRewardFunctionV2:
    """增强奖励函数：重点优化placement质量"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Placement相关权重（大幅增强）
        self.r_colocate = config.get('r_colocate', 5.0)
        self.r_separate = config.get('r_separate', 3.0)
        self.r_communication = config.get('r_communication', 2.0)
        self.r_placement_quality = config.get('r_placement_quality', 10.0)
        
        # 其他权重
        self.w_alloc = config.get('w_alloc', 2.0)
        self.w_schedule = config.get('w_schedule', 1.0)
        self.w_progress = config.get('w_progress', 0.001)
        self.w_expert = config.get('w_expert', 3.0)
        
        self.r_time = config.get('r_time', 2.0)
        self.r_parallel_epr = config.get('r_parallel_epr', 1.0)
        self.r_step = config.get('r_step', 0.0001)
        self.r_expert_match = config.get('r_expert_match', 10.0)
        self.r_completion = config.get('r_completion', 30.0)
        
        # 预计算qubit交互强度
        self.interaction_strengths = None
        
    def precompute_interactions(self, env):
        """预计算qubit之间的交互强度"""
        self.interaction_strengths = {}
        gates = env.problem_data['gates']
        
        # 计算每对qubit的交互次数
        from collections import defaultdict
        interactions = defaultdict(int)
        
        for gate in gates.values():
            key = tuple(sorted([gate['control'], gate['target']]))
            interactions[key] += 1
        
        self.interaction_strengths = interactions
        
    def calculate_reward(self, env, action: Tuple, info: Dict, 
                        expert_action: Optional[Tuple] = None) -> float:
        """计算奖励，重点关注placement质量"""
        
        # 预计算交互（如果还没有）
        if self.interaction_strengths is None:
            self.precompute_interactions(env)
        
        reward = 0.0
        
        if action[0] == 'map':
            reward += self._calculate_enhanced_allocation_reward(env, action[1])
        elif action[0] == 'schedule':
            reward += self._calculate_scheduling_reward(env, action[1])
        
        # 步数惩罚（很小）
        reward -= self.w_progress * self.r_step
        
        # 专家匹配奖励（增强）
        if expert_action is not None:
            if action[0] == expert_action[0] and action[1] == expert_action[1]:
                reward += self.w_expert * self.r_expert_match
            elif action[0] == expert_action[0]:
                # 即使参数不同，动作类型匹配也有小奖励
                reward += self.w_expert * self.r_expert_match * 0.2
        
        # 完成奖励
        if info.get('done', False):
            completion_rate = env._get_completion_rate()
            reward += self.r_completion * completion_rate
            
            # 额外的placement质量奖励
            if completion_rate >= 1.0:
                placement_quality = self._evaluate_placement_quality(env)
                reward += self.r_placement_quality * placement_quality
        
        return reward
    
    def _calculate_enhanced_allocation_reward(self, env, params: Tuple) -> float:
        """增强的allocation奖励计算"""
        qubit, qpu = params
        reward = 0.0
        
        # 获取所有相关gates
        gates = env.problem_data['gates']
        
        # 计算与已映射qubits的交互
        for g_id, gate in gates.items():
            other_qubit = None
            interaction_weight = 1.0
            
            # 获取交互强度
            if gate['control'] == qubit and gate['target'] in env.state.mapped_qubits:
                other_qubit = gate['target']
                key = tuple(sorted([qubit, other_qubit]))
                interaction_weight = self.interaction_strengths.get(key, 1)
                
            elif gate['target'] == qubit and gate['control'] in env.state.mapped_qubits:
                other_qubit = gate['control']
                key = tuple(sorted([qubit, other_qubit]))
                interaction_weight = self.interaction_strengths.get(key, 1)
            
            if other_qubit is not None:
                other_qpu = env.state.qubit_mapping[other_qubit]
                
                if qpu == other_qpu:
                    # Co-location奖励（根据交互强度加权）
                    reward += self.w_alloc * self.r_colocate * interaction_weight
                else:
                    # 分离惩罚（根据通信成本和交互强度）
                    comm_cost = env.problem_data['C'].get((qpu, other_qpu), 0)
                    penalty = self.w_alloc * self.r_separate * comm_cost * interaction_weight
                    reward -= penalty
                    
                    # 额外的通信惩罚
                    reward -= self.r_communication * comm_cost * 0.5
        
        # 负载均衡考虑
        qpu_load = sum(1 for q, u in env.state.qubit_mapping.items() if u == qpu)
        capacity = env.problem_data['Cap'][qpu]
        load_ratio = qpu_load / capacity if capacity > 0 else 0
        
        # 适度的负载均衡奖励
        if load_ratio < 0.8:
            reward += 0.5 * (1 - load_ratio)
        elif load_ratio > 0.9:
            reward -= 1.0  # 过载惩罚
        
        return reward
    
    def _calculate_scheduling_reward(self, env, params: Tuple) -> float:
        """调度奖励（保持原有逻辑）"""
        gate, time_slot = params
        reward = 0.0
        
        # 时间效率
        if time_slot > 0:
            reward += self.w_schedule * self.r_time / time_slot
        else:
            reward += self.w_schedule * self.r_time
        
        # 并行EPR生成奖励
        if env.state.is_remote_gate(gate):
            parallel_count = sum(
                1 for other_gate in env.state.scheduled_gates
                if other_gate != gate 
                and env.state.gate_schedule.get(other_gate) == time_slot
                and env.state.is_remote_gate(other_gate)
            )
            
            if parallel_count > 0:
                reward += self.w_schedule * self.r_parallel_epr * np.sqrt(parallel_count)
        
        return reward
    
    def _evaluate_placement_quality(self, env) -> float:
        """评估placement质量（0到1之间）"""
        # 计算总通信成本
        total_comm_cost = 0
        local_gates = 0
        remote_gates = 0
        
        for g_id, gate in env.problem_data['gates'].items():
            control = gate['control']
            target = gate['target']
            
            if control in env.state.qubit_mapping and target in env.state.qubit_mapping:
                control_qpu = env.state.qubit_mapping[control]
                target_qpu = env.state.qubit_mapping[target]
                
                if control_qpu == target_qpu:
                    local_gates += 1
                else:
                    remote_gates += 1
                    comm_cost = env.problem_data['C'].get((control_qpu, target_qpu), 0)
                    total_comm_cost += comm_cost
        
        # 计算质量分数（越低的通信成本越好）
        # 使用Expert的通信成本作为基准（约29.59）
        expert_baseline = 30.0
        
        if total_comm_cost < expert_baseline:
            quality = 1.0  # 比expert更好
        elif total_comm_cost < expert_baseline * 1.5:
            quality = 1.0 - (total_comm_cost - expert_baseline) / expert_baseline
        else:
            quality = 0.3  # 差太多了
        
        return max(0.0, min(1.0, quality))