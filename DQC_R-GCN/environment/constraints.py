import numpy as np
from typing import Dict, Tuple, Optional

class ConstraintChecker:
    """Check constraints for DQC problem."""
    
    def __init__(self, state):
        self.state = state
    
    def can_map_qubit(self, qubit: int, qpu: int) -> bool:
        """Check if a qubit can be mapped to a QPU."""
        # Check if qubit is already mapped
        if qubit in self.state.mapped_qubits:
            return False
        
        # Check QPU capacity
        current_load = sum(
            1 for q, assigned_qpu in self.state.qubit_mapping.items() 
            if assigned_qpu == qpu
        )
        if current_load >= self.state.qpu_capacities[qpu]:
            return False
        
        return True
    
    def can_schedule_gate(self, gate: int, time_slot: int) -> bool:
        """Check if a gate can be scheduled at a given time slot."""
        # Check if gate is already scheduled
        if gate in self.state.scheduled_gates:
            return False
        
        # Check time slot validity
        if time_slot < 0 or time_slot >= self.state.time_horizon:
            return False
        
        # Check precedence constraints
        predecessors = list(self.state.dependency_graph.predecessors(gate))
        for pred in predecessors:
            if pred not in self.state.scheduled_gates:
                return False
            if self.state.gate_schedule[pred] >= time_slot:
                return False
        
        # Check if gate qubits are mapped
        control, target = self.state.get_gate_qubits(gate)
        if control not in self.state.mapped_qubits or target not in self.state.mapped_qubits:
            return False
        
        # If remote gate, check EPR capacity
        if self.state.is_remote_gate(gate):
            u1 = self.state.qubit_mapping[control]
            u2 = self.state.qubit_mapping[target]
            
            # Check EPR capacity at both QPUs
            epr_count_u1 = sum(
                self.state.epr_inventory[time_slot][u1].get(v, 0) 
                for v in range(self.state.num_qpus) if v != u1
            )
            epr_count_u2 = sum(
                self.state.epr_inventory[time_slot][u2].get(v, 0) 
                for v in range(self.state.num_qpus) if v != u2
            )
            
            if epr_count_u1 >= self.state.epr_capacities[u1]:
                return False
            if epr_count_u2 >= self.state.epr_capacities[u2]:
                return False
        
        return True
    
    def get_valid_actions(self) -> Dict[str, list]:
        """Get all valid actions in the current state."""
        valid_actions = {
            'map': [],      # (qubit, qpu) pairs
            'schedule': []  # (gate, time_slot) pairs
        }
        
        # Valid mapping actions
        for qubit in self.state.get_unmapped_qubits():
            for qpu in range(self.state.num_qpus):
                if self.can_map_qubit(qubit, qpu):
                    valid_actions['map'].append((qubit, qpu))
        
        # Valid scheduling actions
        ready_gates = self.state.get_ready_gates()
        for gate in ready_gates:
            # Only check gates whose qubits are mapped
            control, target = self.state.get_gate_qubits(gate)
            if control in self.state.mapped_qubits and target in self.state.mapped_qubits:
                # Try scheduling in next few time slots
                for t in range(self.state.current_time, 
                             min(self.state.current_time + 10, self.state.time_horizon)):
                    if self.can_schedule_gate(gate, t):
                        valid_actions['schedule'].append((gate, t))
        
        return valid_actions