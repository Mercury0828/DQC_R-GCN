import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Union
from .state import DQCState
from .constraints import ConstraintChecker

class DQCEnvironment(gym.Env):
    """OpenAI Gym environment for DQC optimization."""
    
    def __init__(self, problem_data: Dict, config: Dict = None):
        """
        Initialize DQC environment.
        
        Args:
            problem_data: Problem instance from QuantumProblemGenerator
            config: Environment configuration
        """
        super().__init__()
        
        self.problem_data = problem_data
        self.config = config or {}
        
        # Initialize state
        self.state = None
        self.constraint_checker = None
        
        # Define action and observation spaces (simplified for now)
        # Will be refined when implementing the full network
        self.action_space = spaces.Discrete(1000)  # Placeholder
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1000,), dtype=np.float32
        )
        
        # Reward configuration
        self.reward_config = {
            'w_alloc': self.config.get('w_alloc', 1.0),
            'w_schedule': self.config.get('w_schedule', 1.0),
            'w_progress': self.config.get('w_progress', 0.01),
            'r_colocate': self.config.get('r_colocate', 0.5),
            'r_separate': self.config.get('r_separate', 1.0),
            'r_time': self.config.get('r_time', 1.0),
            'r_parallel_epr': self.config.get('r_parallel_epr', 0.5),
            'r_step': self.config.get('r_step', 0.01)
        }
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = self.config.get('max_steps', 1000)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = DQCState(self.problem_data)
        self.constraint_checker = ConstraintChecker(self.state)
        self.step_count = 0
        
        # Return observation
        try:
            obs = self._get_observation()
        except Exception as e:
            # If there's an error, return a default observation
            print(f"Warning: Error getting observation: {e}")
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        return obs
    
    def step(self, action: Union[Tuple, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Either a tuple (action_type, params) or an encoded action
        
        Returns:
            observation: Next state observation
            reward: Immediate reward
            done: Whether episode is finished
            info: Additional information
        """
        self.step_count += 1
        
        # Decode action if necessary
        if isinstance(action, int):
            action = self._decode_action(action)
        
        action_type, params = action
        
        # Execute action
        success = False
        if action_type == 'map':
            success = self._execute_mapping(params)
        elif action_type == 'schedule':
            success = self._execute_scheduling(params)
        
        # Calculate reward
        reward = self._calculate_reward(action_type, params, success)
        
        # Check if done
        done = self._is_done()
        
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'action_type': action_type,
            'action_params': params,
            'success': success,
            'step_count': self.step_count,
            'completion_rate': self._get_completion_rate()
        }
        
        return observation, reward, done, info
    
    def _execute_mapping(self, params: Tuple[int, int]) -> bool:
        """Execute qubit mapping action."""
        qubit, qpu = params
        
        if not self.constraint_checker.can_map_qubit(qubit, qpu):
            return False
        
        self.state.qubit_mapping[qubit] = qpu
        self.state.mapped_qubits.add(qubit)
        return True
    
    def _execute_scheduling(self, params: Tuple[int, int]) -> bool:
        """Execute gate scheduling action."""
        gate, time_slot = params
        
        if not self.constraint_checker.can_schedule_gate(gate, time_slot):
            return False
        
        self.state.gate_schedule[gate] = time_slot
        self.state.scheduled_gates.add(gate)
        
        # Update EPR inventory if remote gate
        if self.state.is_remote_gate(gate):
            control, target = self.state.get_gate_qubits(gate)
            u1 = self.state.qubit_mapping[control]
            u2 = self.state.qubit_mapping[target]
            
            # Generate EPR pair (simplified - in-slot generation)
            self.state.epr_inventory[time_slot][u1][u2] = \
                self.state.epr_inventory[time_slot][u1].get(u2, 0) + 1
            self.state.epr_inventory[time_slot][u2][u1] = \
                self.state.epr_inventory[time_slot][u2].get(u1, 0) + 1
            
            # Store EPR generation info
            self.state.epr_schedule[(gate, time_slot)] = {
                'type': 'in_slot',
                'qpus': (u1, u2)
            }
        
        # Update current time
        self.state.current_time = max(self.state.current_time, time_slot)
        
        return True
    
    def _calculate_reward(self, action_type: str, params: Tuple, success: bool) -> float:
        """Calculate immediate reward for an action."""
        if not success:
            return -0.1  # Small penalty for invalid actions
        
        reward = 0.0
        cfg = self.reward_config
        
        if action_type == 'map':
            qubit, qpu = params
            reward += self._calculate_allocation_reward(qubit, qpu)
        elif action_type == 'schedule':
            gate, time_slot = params
            reward += self._calculate_scheduling_reward(gate, time_slot)
        
        # Progress penalty
        reward -= cfg['w_progress'] * cfg['r_step']
        
        return reward
    
    def _calculate_allocation_reward(self, qubit: int, qpu: int) -> float:
        """Calculate reward for qubit allocation."""
        cfg = self.reward_config
        reward = 0.0
        
        # Check interaction with already mapped qubits
        for g_id, gate in self.state.gates.items():
            other_qubit = None
            if gate['control'] == qubit and gate['target'] in self.state.mapped_qubits:
                other_qubit = gate['target']
            elif gate['target'] == qubit and gate['control'] in self.state.mapped_qubits:
                other_qubit = gate['control']
            
            if other_qubit is not None:
                other_qpu = self.state.qubit_mapping[other_qubit]
                if qpu == other_qpu:
                    # Co-located
                    reward += cfg['w_alloc'] * cfg['r_colocate']
                else:
                    # Separated - penalize by communication cost
                    comm_cost = self.state.comm_costs.get((qpu, other_qpu), 0)
                    reward -= cfg['w_alloc'] * cfg['r_separate'] * comm_cost
        
        return reward
    
    def _calculate_scheduling_reward(self, gate: int, time_slot: int) -> float:
        """Calculate reward for gate scheduling."""
        cfg = self.reward_config
        reward = 0.0
        
        # Time efficiency reward
        if time_slot > 0:
            reward += cfg['w_schedule'] * cfg['r_time'] / time_slot
        
        # Check for parallel EPR generation
        if self.state.is_remote_gate(gate):
            # Check if other EPR pairs are generated in same slot
            parallel_count = 0
            for other_gate in self.state.scheduled_gates:
                if other_gate != gate and self.state.gate_schedule.get(other_gate) == time_slot:
                    if self.state.is_remote_gate(other_gate):
                        parallel_count += 1
            
            if parallel_count > 0:
                reward += cfg['w_schedule'] * cfg['r_parallel_epr']
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is finished."""
        # Episode ends when all gates are scheduled or max steps reached
        all_scheduled = len(self.state.scheduled_gates) == self.state.num_gates
        max_steps_reached = self.step_count >= self.max_steps
        return all_scheduled or max_steps_reached
    
    def _get_completion_rate(self) -> float:
        """Get task completion rate."""
        return len(self.state.scheduled_gates) / self.state.num_gates
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Simplified observation for now
        # Will be replaced with graph representation for R-GCN
        state_repr = self.state.get_state_representation()
        
        # Flatten features for now
        obs = []
        
        # Gate features
        if state_repr['gate_features'].size > 0:
            obs.extend(state_repr['gate_features'].flatten())
        
        # Qubit features  
        obs.extend(state_repr['qubit_features'].flatten())
        
        # QPU features
        obs.extend(state_repr['qpu_features'].flatten())
        
        # Pad to fixed size
        obs = np.array(obs, dtype=np.float32)
        if len(obs) < self.observation_space.shape[0]:
            obs = np.pad(obs, (0, self.observation_space.shape[0] - len(obs)))
        else:
            obs = obs[:self.observation_space.shape[0]]
        
        return obs
    
    def _decode_action(self, action_id: int) -> Tuple[str, Tuple]:
        """Decode integer action to (type, params)."""
        # This is a placeholder - will be replaced with proper action encoding
        valid_actions = self.constraint_checker.get_valid_actions()
        
        all_actions = []
        for qubit, qpu in valid_actions['map']:
            all_actions.append(('map', (qubit, qpu)))
        for gate, time in valid_actions['schedule']:
            all_actions.append(('schedule', (gate, time)))
        
        if not all_actions:
            return ('noop', ())
        
        action_idx = action_id % len(all_actions)
        return all_actions[action_idx]
    
    def get_valid_actions(self) -> Dict[str, list]:
        """Get current valid actions."""
        return self.constraint_checker.get_valid_actions()
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"\n=== DQC Environment State ===")
            print(f"Step: {self.step_count}/{self.max_steps}")
            print(f"Mapped qubits: {len(self.state.mapped_qubits)}/{self.state.num_qubits}")
            print(f"Scheduled gates: {len(self.state.scheduled_gates)}/{self.state.num_gates}")
            print(f"Current time: {self.state.current_time}")
            print(f"Completion rate: {self._get_completion_rate():.2%}")