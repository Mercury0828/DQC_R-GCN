import numpy as np
import random
import networkx as nx
from typing import Dict, List, Tuple, Set
import json

class QuantumProblemGenerator:
    def __init__(self, num_gates=200, num_qpus=5, num_qubits=100, qpu_capacity=20, time_horizon=200, seed=42):
        """
        Initialize the quantum problem generator with updated parameters.
        
        Parameters:
        - num_gates (g): number of CNOT gates (default: 200)
        - num_qpus (u): number of QPU nodes (default: 5)
        - num_qubits: total number of logical qubits (default: 100)
        - qpu_capacity: capacity of each QPU in qubits (default: 20)
        - time_horizon (H): time slots upper bound (default: 200)
        - seed: random seed for reproducibility
        """
        self.g = num_gates  # |G|
        self.u = num_qpus   # |U|
        self.num_qubits = num_qubits  # Total logical qubits
        self.qpu_capacity = qpu_capacity  # Qubits per QPU
        self.H = time_horizon
        self.v = self.u * self.H  # total candidate positions
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize data structures
        self.gates = []           # G: list of gates
        self.qubits = set()       # Q: set of logical qubits
        self.qpus = list(range(self.u))  # U: QPU indices
        self.precedence = []      # P: precedence relations
        self.comm_costs = {}      # C_uv: communication costs
        self.qpu_capacities = {}  # Cap_u: QPU capacities
        self.epr_capacities = {}  # E_u: EPR storage capacities
        self.alpha = 1.0          # objective weight
        
    def generate_quantum_circuit(self):
        """Generate a random quantum circuit with CNOT gates."""
        print("Generating quantum circuit...")
        
        # Use the specified number of qubits
        self.qubits = set(range(self.num_qubits))
        print(f"Generated {self.num_qubits} logical qubits")
        
        # Generate CNOT gates
        for i in range(self.g):
            # Randomly select control and target qubits
            control, target = random.sample(list(self.qubits), 2)
            gate = {
                'id': i,
                'control': control,
                'target': target,
                'type': 'CNOT'
            }
            self.gates.append(gate)
        
        print(f"Generated {len(self.gates)} CNOT gates")
        
    def generate_precedence_relations(self):
        """Generate precedence relations between gates."""
        print("Generating precedence relations...")
        
        # Create a DAG for gate dependencies
        # Probability that a gate depends on another gate
        dependency_prob = 0.15
        
        for i in range(len(self.gates)):
            for j in range(i + 1, len(self.gates)):
                gate_i = self.gates[i]
                gate_j = self.gates[j]
                
                # Check if gates share qubits (natural dependency)
                shared_qubits = {gate_i['control'], gate_i['target']} & {gate_j['control'], gate_j['target']}
                
                if shared_qubits and random.random() < dependency_prob:
                    # gate_i must execute before gate_j
                    self.precedence.append((i, j))
                elif random.random() < dependency_prob * 0.1:  # Random dependencies
                    self.precedence.append((i, j))
        
        # Ensure no cycles using topological sort
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.gates)))
        G.add_edges_from(self.precedence)
        
        if not nx.is_directed_acyclic_graph(G):
            # Remove edges to make it acyclic
            self.precedence = list(nx.edge_dfs(G, orientation='original'))
            
        print(f"Generated {len(self.precedence)} precedence relations")
        
    def generate_qpu_parameters(self):
        """Generate QPU-related parameters with better EPR capacity - FIXED VERSION."""
        print("Generating QPU parameters...")
        
        # Set all QPUs to have the specified capacity
        for u in self.qpus:
            self.qpu_capacities[u] = self.qpu_capacity
        
        # Verify total capacity is sufficient
        total_capacity = sum(self.qpu_capacities.values())
        print(f"Total QPU capacity: {total_capacity} (for {self.num_qubits} qubits)")
        
        if total_capacity < self.num_qubits:
            raise ValueError(f"Total QPU capacity ({total_capacity}) is less than number of qubits ({self.num_qubits})")
        
        # EPR pair storage capacities - more realistic values
        # Estimate based on potential cross-QPU communication
        # Worst case: each QPU might need to communicate with all others
        max_concurrent_comm = min(self.qpu_capacity, (self.u - 1) * 2)  # Factor of 2 for safety
        
        for u in self.qpus:
            # Set EPR capacity based on QPU's qubit capacity and communication needs
            # This ensures we have enough capacity for reasonable cross-QPU operations
            self.epr_capacities[u] = max(10, min(30, max_concurrent_comm))
        
        print(f"QPU capacities: {self.qpu_capacities}")
        print(f"EPR capacities: {self.epr_capacities}")
        
    def generate_communication_costs(self):
        """Generate communication cost matrix between QPUs."""
        print("Generating communication costs...")
        
        # Initialize communication costs
        for u in self.qpus:
            for v in self.qpus:
                if u == v:
                    self.comm_costs[(u, v)] = 0  # No cost for same QPU
                else:
                    # Generate realistic communication costs
                    # Costs are higher for distant QPUs
                    base_cost = random.uniform(1.0, 5.0)
                    distance_factor = 1 + 0.1 * abs(u - v)  # Simple distance model
                    self.comm_costs[(u, v)] = base_cost * distance_factor
        
        print("Communication cost matrix generated")
        
    def generate_all_parameters(self):
        """Generate all problem parameters."""
        print("="*50)
        print("QUANTUM PROBLEM GENERATOR")
        print("="*50)
        print(f"Problem scale:")
        print(f"  - Gates: {self.g}")
        print(f"  - QPUs: {self.u}")
        print(f"  - Qubits: {self.num_qubits}")
        print(f"  - QPU capacity: {self.qpu_capacity} qubits/QPU")
        print(f"  - Time horizon: {self.H}")
        print("="*50)
        
        self.generate_quantum_circuit()
        self.generate_precedence_relations()
        self.generate_qpu_parameters()
        self.generate_communication_costs()
        
        print("="*50)
        print("Parameter generation completed!")
        print("="*50)
        
    def export_parameters(self, filename="quantum_problem_instance.json"):
        """Export all parameters to a JSON file."""
        data = {
            'problem_scale': {
                'num_gates': self.g,
                'num_qpus': self.u,
                'num_qubits': self.num_qubits,
                'qpu_capacity': self.qpu_capacity,
                'time_horizon': self.H,
                'total_positions': self.v
            },
            'qubits': list(self.qubits),
            'gates': self.gates,
            'precedence_relations': self.precedence,
            'qpu_capacities': self.qpu_capacities,
            'epr_capacities': self.epr_capacities,
            'communication_costs': {f"{k[0]},{k[1]}": v for k, v in self.comm_costs.items()},
            'alpha': self.alpha
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Parameters exported to {filename}")
        
    def print_summary(self):
        """Print a summary of generated parameters."""
        print("\nPROBLEM SUMMARY:")
        print(f"- Number of logical qubits (|Q|): {len(self.qubits)}")
        print(f"- Number of CNOT gates (|G|): {len(self.gates)}")
        print(f"- Number of QPUs (|U|): {len(self.qpus)}")
        print(f"- QPU capacity (uniform): {self.qpu_capacity} qubits/QPU")
        print(f"- Time horizon (H): {self.H}")
        print(f"- Precedence relations (|P|): {len(self.precedence)}")
        print(f"- Total QPU capacity: {sum(self.qpu_capacities.values())}")
        print(f"- Average EPR capacity: {np.mean(list(self.epr_capacities.values())):.2f}")
        print(f"- Communication cost range: [{min(v for k, v in self.comm_costs.items() if k[0] != k[1]):.2f}, {max(self.comm_costs.values()):.2f}]")
        
    def get_model_data(self):
        """Return all data in a format suitable for optimization models."""
        return {
            # Sets
            'Q': list(self.qubits),
            'G': list(range(len(self.gates))),
            'U': self.qpus,
            'P': self.precedence,
            'H': self.H,
            
            # Gate information
            'gates': {i: {'control': gate['control'], 'target': gate['target']} 
                     for i, gate in enumerate(self.gates)},
            
            # Parameters
            'Cap': self.qpu_capacities,
            'E': self.epr_capacities,
            'C': self.comm_costs,
            'alpha': self.alpha
        }

# Example usage
if __name__ == "__main__":
    # Generate problem instance with specified scale
    generator = QuantumProblemGenerator(
        num_gates=200,      # 200 CNOT gates
        num_qpus=5,         # 5 QPUs
        num_qubits=100,     # 100 logical qubits
        qpu_capacity=20,    # 20 qubits per QPU
        time_horizon=200,   # Time slots
        seed=42
    )
    
    # Generate all parameters
    generator.generate_all_parameters()
    
    # Print summary
    generator.print_summary()
    
    # Export to file
    generator.export_parameters()
    
    # Get data for optimization model
    model_data = generator.get_model_data()
    
    print(f"\nModel data structure ready for optimization solver!")
    print(f"Available keys: {list(model_data.keys())}")
    
    # Example: Access specific data
    print(f"\nExample data access:")
    print(f"First 5 gates: {[(i, model_data['gates'][i]) for i in range(min(5, len(model_data['gates'])))]}")
    print(f"First 5 precedence relations: {model_data['P'][:5]}")
    print(f"QPU capacities: {model_data['Cap']}")