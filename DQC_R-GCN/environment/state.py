import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

@dataclass
class DQCState:
    """Represents the current state of the DQC optimization problem."""
    
    def __init__(self, problem_data: Dict):
        """
        Initialize DQC state from problem data.
        
        Args:
            problem_data: Dictionary containing problem parameters from QuantumProblemGenerator
        """
        # Problem parameters
        self.num_qubits = len(problem_data['Q'])
        self.num_gates = len(problem_data['G'])
        self.num_qpus = len(problem_data['U'])
        self.time_horizon = problem_data['H']
        
        # Static problem data
        self.gates = problem_data['gates']
        self.precedence = problem_data['P']
        self.qpu_capacities = problem_data['Cap']
        self.epr_capacities = problem_data['E']
        self.comm_costs = problem_data['C']
        
        # Dynamic state variables
        self.qubit_mapping = {}  # q -> u mapping (partial)
        self.gate_schedule = {}  # g -> t mapping (partial)
        self.epr_schedule = {}   # (g, t) -> EPR generation info
        
        # Track which gates/qubits have been processed
        self.mapped_qubits = set()
        self.scheduled_gates = set()
        self.current_time = 0
        
        # EPR inventory tracking: epr_inventory[t][u][v] = count
        self.epr_inventory = self._initialize_epr_inventory()
        
        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()
        
    def _initialize_epr_inventory(self):
        """Initialize EPR pair inventory for all time slots."""
        inventory = {}
        for t in range(self.time_horizon):
            inventory[t] = {}
            for u in range(self.num_qpus):
                inventory[t][u] = {}
                for v in range(self.num_qpus):
                    if u != v:
                        inventory[t][u][v] = 0
        return inventory
    
    def _build_dependency_graph(self):
        """Build dependency graph for gates."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_gates))
        G.add_edges_from(self.precedence)
        return G
    
    def get_ready_gates(self) -> List[int]:
        """Get gates that are ready to be scheduled (all predecessors scheduled)."""
        ready_gates = []
        for g in range(self.num_gates):
            if g not in self.scheduled_gates:
                # Check if all predecessors are scheduled
                predecessors = list(self.dependency_graph.predecessors(g))
                if all(pred in self.scheduled_gates for pred in predecessors):
                    ready_gates.append(g)
        return ready_gates
    
    def get_unmapped_qubits(self) -> List[int]:
        """Get qubits that haven't been mapped to QPUs yet."""
        all_qubits = set(range(self.num_qubits))
        return list(all_qubits - self.mapped_qubits)
    
    def get_available_qpus_for_qubit(self, qubit: int) -> List[int]:
        """Get QPUs with available capacity for a qubit."""
        available = []
        for u in range(self.num_qpus):
            current_load = sum(1 for q, qpu in self.qubit_mapping.items() if qpu == u)
            if current_load < self.qpu_capacities[u]:
                available.append(u)
        return available
    
    def get_gate_qubits(self, gate_id: int) -> Tuple[int, int]:
        """Get control and target qubits for a gate."""
        gate = self.gates[gate_id]
        return gate['control'], gate['target']
    
    def is_remote_gate(self, gate_id: int) -> bool:
        """Check if a gate is remote (crosses QPUs)."""
        control, target = self.get_gate_qubits(gate_id)
        if control not in self.mapped_qubits or target not in self.mapped_qubits:
            return False
        return self.qubit_mapping[control] != self.qubit_mapping[target]
    
    def get_state_representation(self) -> Dict:
        """
        Get comprehensive state representation for neural network.
        Returns node features and adjacency information.
        """
        # This will be expanded when implementing R-GCN
        return {
            'gate_features': self._get_gate_features(),
            'qubit_features': self._get_qubit_features(),
            'qpu_features': self._get_qpu_features(),
            'edges': self._get_graph_edges()
        }
    
    def _get_gate_features(self) -> np.ndarray:
        """Extract features for gate nodes."""
        features = []
        ready_gates = set(self.get_ready_gates())
        
        for g in range(self.num_gates):
            if g in self.scheduled_gates:
                continue
                
            # Readiness status
            is_ready = 1.0 if g in ready_gates else 0.0
            
            # Normalized criticality (depth in DAG)
            try:
                # Get all shortest path lengths from this node
                lengths = nx.single_target_shortest_path_length(
                    self.dependency_graph.reverse(), g
                )
                depth = max(lengths.values()) if lengths else 0
                
                # Get maximum depth in the entire graph
                all_depths = []
                for node in range(self.num_gates):
                    try:
                        node_lengths = nx.single_target_shortest_path_length(
                            self.dependency_graph.reverse(), node
                        )
                        if node_lengths:
                            all_depths.append(max(node_lengths.values()))
                        else:
                            all_depths.append(0)
                    except:
                        all_depths.append(0)
                
                max_depth = max(all_depths) if all_depths else 1
                normalized_depth = depth / max_depth if max_depth > 0 else 0
            except:
                # If there's any issue with depth calculation, use default
                normalized_depth = 0.5
            
            features.append([is_ready, normalized_depth])
            
        return np.array(features) if features else np.empty((0, 2))
    
    def _get_qubit_features(self) -> np.ndarray:
        """Extract features for qubit nodes."""
        features = []
        
        for q in range(self.num_qubits):
            # Mapping status
            is_mapped = 1.0 if q in self.mapped_qubits else 0.0
            
            # QPU location (one-hot, simplified for now)
            qpu_location = np.zeros(self.num_qpus)
            if q in self.mapped_qubits:
                qpu_location[self.qubit_mapping[q]] = 1.0
            
            # Normalized interaction strength
            interaction_strength = 0
            for g_id, gate in self.gates.items():
                if gate['control'] == q or gate['target'] == q:
                    interaction_strength += 1
            norm_interaction = interaction_strength / self.num_gates
            
            feature = [is_mapped] + qpu_location.tolist() + [norm_interaction]
            features.append(feature)
            
        return np.array(features)
    
    def _get_qpu_features(self) -> np.ndarray:
        """Extract features for QPU nodes."""
        features = []
        
        for u in range(self.num_qpus):
            # Qubit load
            current_load = sum(1 for q, qpu in self.qubit_mapping.items() if qpu == u)
            qubit_load = current_load / self.qpu_capacities[u]
            
            # Communication load (simplified for current time)
            comm_load = 0
            if self.current_time < self.time_horizon:
                for v in range(self.num_qpus):
                    if u != v and self.current_time in self.epr_inventory:
                        comm_load += self.epr_inventory[self.current_time][u].get(v, 0)
                comm_load = comm_load / self.epr_capacities[u] if self.epr_capacities[u] > 0 else 0
            
            features.append([qubit_load, comm_load])
            
        return np.array(features)
    
    def _get_graph_edges(self) -> Dict[str, List[Tuple[int, int]]]:
        """Get edges for the state graph."""
        edges = {
            'depends_on': [],
            'acts_on': [],
            'assigned_to': [],
            'communicates_with': []
        }
        
        # Gate dependencies
        for src, dst in self.precedence:
            if src not in self.scheduled_gates and dst not in self.scheduled_gates:
                edges['depends_on'].append((src, dst))
        
        # Gate-qubit relationships
        for g_id, gate in self.gates.items():
            if g_id not in self.scheduled_gates:
                edges['acts_on'].append((g_id, gate['control']))
                edges['acts_on'].append((g_id, gate['target']))
        
        # Qubit-QPU assignments
        for qubit, qpu in self.qubit_mapping.items():
            edges['assigned_to'].append((qubit, qpu))
        
        # QPU communication (based on remote gates)
        for g_id in self.scheduled_gates:
            if self.is_remote_gate(g_id):
                control, target = self.get_gate_qubits(g_id)
                u1, u2 = self.qubit_mapping[control], self.qubit_mapping[target]
                if u1 != u2:
                    edges['communicates_with'].append((u1, u2))
        
        return edges
    
    def copy(self):
        """Create a deep copy of the state."""
        import copy
        return copy.deepcopy(self)