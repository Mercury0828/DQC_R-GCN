"""
UNIQ expert policy with debugging capabilities - Fixed version.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx

class UNIQExpertPolicy:
    """
    Expert policy based on UNIQ's Greedy-JIT algorithm with debug logging.
    """
    
    def __init__(self, env, debug=False):
        """
        Initialize UNIQ expert policy.
        
        Args:
            env: DQC environment instance
            debug: Enable debug logging
        """
        self.env = env
        self.problem_data = env.problem_data
        self.debug = debug
        
        # Build and cache data structures
        self.interaction_graph = self._build_interaction_graph()
        self.successor_cache = self._precompute_successors()
        self.depth_cache = self._precompute_depths()
        self.interaction_strengths = self._precompute_interaction_strengths()
        
        # Debug statistics
        self.action_history = []
    
    def _build_interaction_graph(self):
        """Build qubit interaction graph from gates."""
        interactions = defaultdict(int)
        
        for g_id, gate in self.problem_data['gates'].items():
            con, tar = gate['control'], gate['target']
            key = (min(con, tar), max(con, tar))
            interactions[key] += 1
        
        return interactions
    
    def _precompute_successors(self) -> Dict[int, Set[int]]:
        """Precompute all successors for each gate."""
        # Build dependency graph
        dep_graph = nx.DiGraph()
        
        # Add all gate nodes first
        num_gates = len(self.problem_data['gates'])
        dep_graph.add_nodes_from(range(num_gates))
        
        # Then add edges from precedence relations
        if self.problem_data['P']:
            dep_graph.add_edges_from(self.problem_data['P'])
        
        # Compute transitive closure for all successors
        successors = {}
        for gate in range(num_gates):
            reachable = set()
            
            if gate in dep_graph:
                try:
                    immediate_successors = list(dep_graph.successors(gate))
                    queue = immediate_successors.copy()
                    reachable.update(immediate_successors)
                    
                    while queue:
                        node = queue.pop(0)
                        for succ in dep_graph.successors(node):
                            if succ not in reachable:
                                reachable.add(succ)
                                queue.append(succ)
                except:
                    pass
            
            successors[gate] = reachable
        
        return successors
    
    def _precompute_depths(self) -> Dict[int, int]:
        """Precompute gate depths in dependency graph."""
        dep_graph = nx.DiGraph()
        
        # Add all gate nodes first
        num_gates = len(self.problem_data['gates'])
        dep_graph.add_nodes_from(range(num_gates))
        
        # Then add edges
        if self.problem_data['P']:
            dep_graph.add_edges_from(self.problem_data['P'])
        
        depths = {}
        
        # Topological sort to process in order
        try:
            topo_order = list(nx.topological_sort(dep_graph))
        except:
            topo_order = range(num_gates)
        
        for gate in topo_order:
            if gate in dep_graph:
                try:
                    pred_depths = [depths.get(pred, -1) for pred in dep_graph.predecessors(gate)]
                    depths[gate] = max(pred_depths) + 1 if pred_depths else 0
                except:
                    depths[gate] = 0
            else:
                depths[gate] = 0
        
        return depths
    
    def _precompute_interaction_strengths(self) -> Dict[int, float]:
        """Precompute interaction strengths for all qubits."""
        strengths = {}
        num_gates = len(self.problem_data['gates'])
        
        # Get number of qubits from problem data
        num_qubits = len(self.problem_data['Q'])
        
        for qubit in range(num_qubits):
            score = 0
            for (q1, q2), weight in self.interaction_graph.items():
                if q1 == qubit or q2 == qubit:
                    score += weight
            strengths[qubit] = score / num_gates if num_gates > 0 else 0
        
        return strengths
    
    def get_expert_action(self, state_repr: Dict, valid_actions: Dict) -> Tuple:
        """
        Get expert action with detailed debugging.
        """
        # Get current state info from environment, not state_repr
        unmapped_qubits = self.env.state.get_unmapped_qubits() if self.env.state else []
        ready_gates = self.env.state.get_ready_gates() if self.env.state else []
        
        # Debug: Log current state
        if self.debug:
            print(f"\n[EXPERT DEBUG] State analysis:")
            print(f"  Unmapped qubits: {unmapped_qubits}")
            print(f"  Ready gates: {ready_gates}")
            print(f"  Valid map actions: {len(valid_actions.get('map', []))}")
            print(f"  Valid schedule actions: {len(valid_actions.get('schedule', []))}")
        
        # Priority 1: Map qubits if unmapped exist
        if unmapped_qubits and valid_actions.get('map'):
            action = self._get_expert_mapping_action(unmapped_qubits, valid_actions['map'])
            if self.debug:
                print(f"[EXPERT DECISION] Mapping action chosen: {action}")
            self.action_history.append(('map', action))
            return action
        
        # Priority 2: Schedule ready gates
        if ready_gates and valid_actions.get('schedule'):
            action = self._get_expert_scheduling_action(ready_gates, valid_actions['schedule'])
            if self.debug:
                print(f"[EXPERT DECISION] Scheduling action chosen: {action}")
            self.action_history.append(('schedule', action))
            return action
        
        # Default: take any valid action
        if valid_actions.get('map'):
            action = ('map', valid_actions['map'][0])
        elif valid_actions.get('schedule'):
            action = ('schedule', valid_actions['schedule'][0])
        else:
            action = ('noop', ())
        
        if self.debug:
            print(f"[EXPERT DECISION] Default action: {action}")
        
        self.action_history.append(action)
        return action
    
    def _get_expert_mapping_action(self, unmapped_qubits: List[int], 
                                  valid_map_actions: List[Tuple]) -> Tuple:
        """Get expert mapping action with debugging."""
        # Use precomputed interaction strengths
        qubit_scores = [(q, self.interaction_strengths.get(q, 0)) 
                        for q in unmapped_qubits]
        
        # Sort by interaction strength
        qubit_scores.sort(key=lambda x: x[1], reverse=True)
        
        if self.debug:
            print(f"[EXPERT MAP] Qubit priorities:")
            for q, score in qubit_scores[:5]:  # Show top 5
                print(f"    Qubit {q}: strength={score:.3f}")
        
        # Find best action for highest priority qubit
        for qubit, score in qubit_scores:
            qubit_actions = [(q, u) for q, u in valid_map_actions if q == qubit]
            
            if qubit_actions:
                best_action = self._choose_best_qpu(qubit, qubit_actions)
                if self.debug:
                    print(f"[EXPERT MAP] Selected qubit {qubit} -> QPU {best_action[1]}")
                return ('map', best_action)
        
        # Fallback: return first valid action
        if valid_map_actions:
            return ('map', valid_map_actions[0])
        return ('noop', ())
    
    def _choose_best_qpu(self, qubit: int, actions: List[Tuple]) -> Tuple:
        """Choose best QPU with debugging."""
        best_score = float('inf')
        best_action = actions[0]
        
        # Check if state exists before accessing
        if not self.env.state:
            return best_action
        
        # Precompute mapped qubit locations
        mapped_locations = {}
        if hasattr(self.env.state, 'qubit_mapping'):
            for q, qpu in self.env.state.qubit_mapping.items():
                mapped_locations[q] = qpu
        
        scores = []
        for q, qpu in actions:
            score = 0
            
            # Communication cost calculation
            for (q1, q2), weight in self.interaction_graph.items():
                if q1 == qubit and q2 in mapped_locations:
                    other_qpu = mapped_locations[q2]
                    comm_cost = self.problem_data['C'].get((qpu, other_qpu), 0)
                    score += weight * comm_cost
                elif q2 == qubit and q1 in mapped_locations:
                    other_qpu = mapped_locations[q1]
                    comm_cost = self.problem_data['C'].get((qpu, other_qpu), 0)
                    score += weight * comm_cost
            
            # Load balancing
            current_load = sum(1 for mapped_q, mapped_qpu in mapped_locations.items() 
                             if mapped_qpu == qpu)
            capacity = self.problem_data['Cap'][qpu]
            load_penalty = (current_load / capacity) * 10 if capacity > 0 else 0
            score += load_penalty
            
            scores.append((qpu, score))
            
            if score < best_score:
                best_score = score
                best_action = (q, qpu)
        
        if self.debug and scores:
            print(f"[EXPERT QPU] QPU scores for qubit {qubit}:")
            for qpu, score in scores:
                print(f"    QPU {qpu}: score={score:.2f}")
        
        return best_action
    
    def _get_expert_scheduling_action(self, ready_gates: List[int], 
                                     valid_schedule_actions: List[Tuple]) -> Tuple:
        """Get expert scheduling action with debugging."""
        # Calculate priorities using precomputed data
        gate_priorities = []
        
        for g in ready_gates:
            successor_count = len(self.successor_cache.get(g, set()))
            
            # Check if gate is remote
            is_remote = False
            if self.env.state and hasattr(self.env.state, 'is_remote_gate'):
                is_remote = self.env.state.is_remote_gate(g)
            
            depth = self.depth_cache.get(g, 0)
            
            priority = successor_count * 10
            if is_remote:
                priority += 5
            priority += (100 - depth)
            
            gate_priorities.append((g, priority))
        
        # Sort by priority
        gate_priorities.sort(key=lambda x: x[1], reverse=True)
        
        if self.debug:
            print(f"[EXPERT SCHEDULE] Gate priorities:")
            for g, priority in gate_priorities[:5]:  # Show top 5
                print(f"    Gate {g}: priority={priority}")
        
        # Find best scheduling action
        for gate, priority in gate_priorities:
            gate_actions = [(g, t) for g, t in valid_schedule_actions if g == gate]
            
            if gate_actions:
                # Choose earliest time slot (JIT principle)
                earliest_action = min(gate_actions, key=lambda x: x[1])
                if self.debug:
                    print(f"[EXPERT SCHEDULE] Selected gate {gate} at time {earliest_action[1]}")
                return ('schedule', earliest_action)
        
        # Fallback
        if valid_schedule_actions:
            return ('schedule', min(valid_schedule_actions, key=lambda x: x[1]))
        return ('noop', ())
    
    def generate_expert_trajectory(self, env, max_steps: int = 1000) -> List[Dict]:
        """Generate a complete expert trajectory."""
        trajectory = []
        state = env.reset()
        
        for step in range(max_steps):
            state_repr = env.state.get_state_representation()
            valid_actions = env.get_valid_actions()
            
            if not valid_actions.get('map') and not valid_actions.get('schedule'):
                break
            
            expert_action = self.get_expert_action(state_repr, valid_actions)
            next_state, reward, done, info = env.step(expert_action)
            
            trajectory.append({
                'state': state_repr,
                'action': expert_action,
                'reward': reward,
                'next_state': env.state.get_state_representation() if env.state else {},
                'done': done,
                'valid_actions': valid_actions
            })
            
            if done:
                break
            
            state = next_state
        
        return trajectory
    
    def get_debug_stats(self) -> Dict:
        """Get debug statistics."""
        if not self.action_history:
            return {}
        
        map_actions = sum(1 for a in self.action_history if a[0] == 'map')
        schedule_actions = sum(1 for a in self.action_history if a[0] == 'schedule')
        
        return {
            'total_actions': len(self.action_history),
            'map_actions': map_actions,
            'schedule_actions': schedule_actions,
            'map_ratio': map_actions / len(self.action_history) if self.action_history else 0,
            'schedule_ratio': schedule_actions / len(self.action_history) if self.action_history else 0
        }