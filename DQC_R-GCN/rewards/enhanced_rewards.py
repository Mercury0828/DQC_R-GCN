"""
Enhanced reward functions for UNIQ-RL training with scheduling incentives.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

class EnhancedRewardFunction:
    """
    Enhanced reward function incorporating UNIQ insights and scheduling incentives.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize reward function.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        
        # Reward weights
        self.w_alloc = config.get('w_alloc', 1.0)
        self.w_schedule = config.get('w_schedule', 1.0)
        self.w_progress = config.get('w_progress', 0.01)
        self.w_expert = config.get('w_expert', 0.5)
        
        # Reward components
        self.r_colocate = config.get('r_colocate', 0.5)
        self.r_separate = config.get('r_separate', 1.0)
        self.r_time = config.get('r_time', 1.0)
        self.r_parallel_epr = config.get('r_parallel_epr', 0.5)
        self.r_step = config.get('r_step', 0.01)
        self.r_expert_match = config.get('r_expert_match', 1.0)
        self.r_completion = config.get('r_completion', 10.0)
        self.r_intermediate_completion = config.get('r_intermediate_completion', 5.0)
        
        # 新增：调度激励参数
        self.r_progress_bonus = config.get('r_progress_bonus', 2.0)
        self.r_schedule_delay_penalty = config.get('r_schedule_delay_penalty', 0.5)
        self.r_balanced_progress = config.get('r_balanced_progress', 1.0)
        
    def calculate_reward(self, env, action: Tuple, info: Dict, 
                        expert_action: Optional[Tuple] = None) -> float:
        """
        Calculate comprehensive reward for an action with scheduling incentives.
        
        Args:
            env: Environment instance after action
            action: Executed action
            info: Action execution info
            expert_action: Expert's recommended action (for imitation bonus)
            
        Returns:
            Total reward
        """
        reward = 0.0
        
        # 获取当前状态统计
        num_mapped = len(env.state.mapped_qubits)
        num_scheduled = len(env.state.scheduled_gates)
        num_qubits = env.state.num_qubits
        num_gates = env.state.num_gates
        
        # 计算进度
        map_progress = num_mapped / num_qubits if num_qubits > 0 else 0
        schedule_progress = num_scheduled / num_gates if num_gates > 0 else 0
        
        # Base reward components
        if action[0] == 'map':
            reward += self._calculate_allocation_reward(env, action[1])
            
            # 如果mapping进度超前于scheduling太多，给予惩罚
            progress_diff = map_progress - schedule_progress
            if progress_diff > 0.3:  # mapping领先太多
                penalty = self.r_schedule_delay_penalty * progress_diff
                reward -= penalty
                
            # 如果所有qubits都映射了但gates还没调度完，严重惩罚
            if map_progress >= 1.0 and schedule_progress < 0.5:
                reward -= 2.0
                
        elif action[0] == 'schedule':
            reward += self._calculate_scheduling_reward(env, action[1])
            
            # 调度进度奖励
            reward += self.r_progress_bonus * (1.0 / num_gates) if num_gates > 0 else 0
            
            # 如果是关键路径上的门，额外奖励
            gate = action[1][0]
            successor_count = self._count_gate_successors(env, gate)
            if successor_count > 2:  # 有多个后继的关键门
                reward += 0.5
            
            # 平衡进度奖励：当scheduling追赶mapping时给予奖励
            if schedule_progress < map_progress:
                catch_up_bonus = self.r_balanced_progress * (map_progress - schedule_progress)
                reward += catch_up_bonus
        
        # Progress penalty (大幅减少对scheduling的惩罚)
        if action[0] == 'schedule':
            reward -= self.w_progress * self.r_step * 0.01  # 调度动作几乎无惩罚
        else:
            reward -= self.w_progress * self.r_step
        
        # Expert imitation bonus
        if expert_action is not None:
            reward += self._calculate_expert_bonus(action, expert_action)
        
        # 中间完成奖励
        if schedule_progress > 0.25 and schedule_progress % 0.25 < 0.05:  # 每完成25%
            reward += self.r_intermediate_completion * schedule_progress
        
        # Completion bonus
        if info.get('done', False):
            completion_rate = env._get_completion_rate()
            reward += self.r_completion * completion_rate
            
            # 完全完成的额外奖励
            if completion_rate >= 1.0:
                reward += 10.0
            elif completion_rate >= 0.8:
                reward += 5.0
        
        # Invalid action penalty
        if not info.get('success', True):
            reward -= 0.5
        
        return reward
    
    def _calculate_allocation_reward(self, env, params: Tuple) -> float:
        """Calculate reward for qubit allocation."""
        qubit, qpu = params
        reward = 0.0
        
        # Check interactions with mapped qubits
        for g_id, gate in env.problem_data['gates'].items():
            other_qubit = None
            
            if gate['control'] == qubit and gate['target'] in env.state.mapped_qubits:
                other_qubit = gate['target']
            elif gate['target'] == qubit and gate['control'] in env.state.mapped_qubits:
                other_qubit = gate['control']
            
            if other_qubit is not None:
                other_qpu = env.state.qubit_mapping[other_qubit]
                
                if qpu == other_qpu:
                    # Reward for co-location
                    reward += self.w_alloc * self.r_colocate
                else:
                    # Penalty for separation (weighted by communication cost)
                    comm_cost = env.problem_data['C'].get((qpu, other_qpu), 0)
                    reward -= self.w_alloc * self.r_separate * comm_cost
        
        # Load balancing bonus
        qpu_load = sum(1 for q, u in env.state.qubit_mapping.items() if u == qpu)
        capacity = env.problem_data['Cap'][qpu]
        load_ratio = qpu_load / capacity if capacity > 0 else 0
        
        # Reward for balanced loading
        if load_ratio < 0.8:  # Not overloaded
            reward += 0.1 * (1 - load_ratio)
        
        return reward
    
    def _calculate_scheduling_reward(self, env, params: Tuple) -> float:
        """Calculate reward for gate scheduling."""
        gate, time_slot = params
        reward = 0.0
        
        # Time efficiency reward (增加权重)
        if time_slot > 0:
            reward += self.w_schedule * self.r_time * 2.0 / time_slot  # 加倍时间效率奖励
        else:
            reward += self.w_schedule * self.r_time * 2.0
        
        # Check for parallel EPR generation opportunity
        if env.state.is_remote_gate(gate):
            # Count other remote gates scheduled at same time
            parallel_count = 0
            for other_gate in env.state.scheduled_gates:
                if other_gate != gate:
                    other_time = env.state.gate_schedule.get(other_gate)
                    if other_time == time_slot and env.state.is_remote_gate(other_gate):
                        parallel_count += 1
            
            if parallel_count > 0:
                reward += self.w_schedule * self.r_parallel_epr * np.sqrt(parallel_count)
        
        # Critical path bonus
        successor_count = self._count_gate_successors(env, gate)
        if successor_count > 0:
            reward += 0.3 * successor_count / (time_slot + 1)
        
        return reward
    
    def _calculate_expert_bonus(self, action: Tuple, expert_action: Tuple) -> float:
        """Calculate bonus for matching expert action."""
        if action[0] == expert_action[0] and action[1] == expert_action[1]:
            return self.w_expert * self.r_expert_match
        elif action[0] == expert_action[0]:  # Partial match (same type)
            return self.w_expert * self.r_expert_match * 0.3
        return 0.0
    
    def _count_gate_successors(self, env, gate: int) -> int:
        """Count gates that depend on this gate."""
        count = 0
        for pred, succ in env.problem_data['P']:
            if pred == gate:
                count += 1
        return count