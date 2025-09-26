"""
Fixed UNIQ expert policy for imitation learning.
Uses the GreedyJIT algorithm from UNIQ paper as expert demonstrations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class UNIQExpertPolicy:
    """
    Expert policy based on UNIQ's Greedy-JIT algorithm.
    Provides expert demonstrations for imitation learning.
    """
    
    def __init__(self, env):
        """
        Initialize UNIQ expert policy.
        
        Args:
            env: DQC environment instance
        """
        self.env = env
        self.problem_data = env.problem_data
        
        # Cache for interaction graph
        self.interaction_graph = self._build_interaction_graph()
        
    def _build_interaction_graph(self):
        """Build qubit interaction graph from gates."""
        interactions = defaultdict(int)
        
        for g_id, gate in self.problem_data['gates'].items():
            con, tar = gate['control'], gate['target']
            key = (min(con, tar), max(con, tar))
            interactions[key] += 1
            
        return interactions
    
    def get_expert_action(self, state_repr: Dict, valid_actions: Dict) -> Tuple:
        """
        Get expert action based on UNIQ algorithm.
        
        Args:
            state_repr: Current state representation
            valid_actions: Dictionary of valid actions
            
        Returns:
            Expert action (type, params)
        """
        # Determine action type based on UNIQ strategy
        unmapped_qubits = self.env.state.get_unmapped_qubits()
        ready_gates = self.env.state.get_ready_gates()
        
        # Priority 1: Map qubits if unmapped exist
        if unmapped_qubits and valid_actions['map']:
            return self._get_expert_mapping_action(unmapped_qubits, valid_actions['map'])
        
        # Priority 2: Schedule ready gates
        if ready_gates and valid_actions['schedule']:
            return self._get_expert_scheduling_action(ready_gates, valid_actions['schedule'])
        
        # Default: take any valid action
        if valid_actions['map']:
            return ('map', valid_actions['map'][0])
        elif valid_actions['schedule']:
            return ('schedule', valid_actions['schedule'][0])
        else:
            return ('noop', ())
    
    def _get_expert_mapping_action(self, unmapped_qubits: List[int], 
                                  valid_map_actions: List[Tuple]) -> Tuple:
        """
        Get expert mapping action based on UNIQ's greedy strategy.
        
        The strategy:
        1. Calculate interaction strength for each unmapped qubit
        2. Map qubit with highest interaction first
        3. Choose QPU that minimizes communication cost
        """
        # Calculate interaction strength for unmapped qubits
        qubit_scores = {}
        for q in unmapped_qubits:
            score = 0
            for (q1, q2), weight in self.interaction_graph.items():
                if q == q1 or q == q2:
                    score += weight
            qubit_scores[q] = score
        
        # Sort by interaction strength (highest first)
        sorted_qubits = sorted(qubit_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find best action for highest priority qubit
        for qubit, _ in sorted_qubits:
            # Find valid actions for this qubit
            qubit_actions = [(q, u) for q, u in valid_map_actions if q == qubit]
            
            if qubit_actions:
                # Choose QPU that minimizes communication cost
                best_action = self._choose_best_qpu(qubit, qubit_actions)
                return ('map', best_action)
        
        # Default: take first valid action
        return ('map', valid_map_actions[0])
    
    def _choose_best_qpu(self, qubit: int, actions: List[Tuple]) -> Tuple:
        """
        Choose best QPU for qubit based on minimizing communication cost.
        """
        best_score = float('inf')
        best_action = actions[0]
        
        for q, qpu in actions:
            score = 0
            
            # Calculate communication cost with already mapped qubits
            for (q1, q2), weight in self.interaction_graph.items():
                other_qubit = None
                if q1 == qubit and q2 in self.env.state.mapped_qubits:
                    other_qubit = q2
                elif q2 == qubit and q1 in self.env.state.mapped_qubits:
                    other_qubit = q1
                
                if other_qubit is not None:
                    other_qpu = self.env.state.qubit_mapping[other_qubit]
                    comm_cost = self.problem_data['C'].get((qpu, other_qpu), 0)
                    score += weight * comm_cost
            
            # Also consider QPU load balancing
            current_load = sum(1 for mapped_q, mapped_qpu in self.env.state.qubit_mapping.items() 
                             if mapped_qpu == qpu)
            capacity = self.problem_data['Cap'][qpu]
            load_penalty = (current_load / capacity) * 10  # Penalty for overloading
            score += load_penalty
            
            if score < best_score:
                best_score = score
                best_action = (q, qpu)
        
        return best_action
    
    def _get_expert_scheduling_action(self, ready_gates: List[int], 
                                     valid_schedule_actions: List[Tuple]) -> Tuple:
        """
        Get expert scheduling action based on UNIQ's JIT strategy.
        
        The strategy:
        1. Prioritize gates on critical path
        2. Schedule as early as possible
        3. Enable parallel EPR generation when possible
        """
        # Calculate gate priorities
        gate_priorities = self._calculate_gate_priorities(ready_gates)
        
        # Sort gates by priority
        sorted_gates = sorted(gate_priorities.items(), key=lambda x: x[1], reverse=True)
        
        # Find best scheduling action
        for gate, _ in sorted_gates:
            # Find valid actions for this gate
            gate_actions = [(g, t) for g, t in valid_schedule_actions if g == gate]
            
            if gate_actions:
                # Choose earliest time slot (JIT principle)
                earliest_action = min(gate_actions, key=lambda x: x[1])
                return ('schedule', earliest_action)
        
        # Default: take earliest available action
        if valid_schedule_actions:
            return ('schedule', min(valid_schedule_actions, key=lambda x: x[1]))
        
        return ('noop', ())
    
    def _calculate_gate_priorities(self, gates: List[int]) -> Dict[int, float]:
        """
        Calculate priority scores for gates.
        Higher score = higher priority.
        """
        priorities = {}
        
        for g in gates:
            score = 0
            
            # Factor 1: Number of successor gates (critical path) - simplified
            successors = self._count_successors_simple(g)
            score += successors * 10
            
            # Factor 2: Is it a cross-QPU gate? (schedule early to manage EPR)
            if self.env.state.is_remote_gate(g):
                score += 5
            
            # Factor 3: Gate depth in dependency graph - simplified
            depth = self._get_gate_depth_simple(g)
            score += (100 - depth)  # Earlier gates get higher priority
            
            priorities[g] = score
        
        return priorities
    
    def _count_successors_simple(self, gate: int) -> int:
        """Count number of gates that depend on this gate (simplified)."""
        count = 0
        for pred, succ in self.problem_data['P']:
            if pred == gate:
                count += 1
        return count
    
    def _get_gate_depth_simple(self, gate: int) -> int:
        """Get depth of gate in dependency graph (simplified)."""
        depth = 0
        for pred, succ in self.problem_data['P']:
            if succ == gate:
                depth = max(depth, 1)  # Simplified - just check if it has predecessors
        return depth
    
    def generate_expert_trajectory(self, env, max_steps: int = 1000) -> List[Dict]:
        """
        Generate a complete expert trajectory.
        
        Args:
            env: Environment instance (will be reset)
            max_steps: Maximum steps in trajectory
            
        Returns:
            List of transitions
        """
        trajectory = []
        state = env.reset()
        
        for step in range(max_steps):
            state_repr = env.state.get_state_representation()
            valid_actions = env.get_valid_actions()
            
            # Get expert action
            if not valid_actions['map'] and not valid_actions['schedule']:
                break
            
            expert_action = self.get_expert_action(state_repr, valid_actions)
            
            # Execute action
            next_state, reward, done, info = env.step(expert_action)
            
            # Store transition
            trajectory.append({
                'state': state_repr,
                'action': expert_action,
                'reward': reward,
                'next_state': env.state.get_state_representation(),
                'done': done,
                'valid_actions': valid_actions
            })
            
            if done:
                break
            
            state = next_state
        
        return trajectory