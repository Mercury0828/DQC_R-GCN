import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

class GraphUtils:
    """Utility functions for graph operations."""
    
    @staticmethod
    def build_interaction_graph(gates: Dict) -> nx.Graph:
        """
        Build qubit interaction graph from gates.
        
        Args:
            gates: Dictionary of gate information
        
        Returns:
            Interaction graph
        """
        G = nx.Graph()
        
        # Add edges for each gate
        for gate_id, gate in gates.items():
            control = gate['control']
            target = gate['target']
            
            if G.has_edge(control, target):
                G[control][target]['weight'] += 1
            else:
                G.add_edge(control, target, weight=1)
        
        return G
    
    @staticmethod
    def compute_interaction_strength(qubit: int, gates: Dict) -> float:
        """
        Compute interaction strength for a qubit.
        
        Args:
            qubit: Qubit index
            gates: Gate dictionary
        
        Returns:
            Normalized interaction strength
        """
        strength = 0
        for gate in gates.values():
            if gate['control'] == qubit or gate['target'] == qubit:
                strength += 1
        
        return strength / len(gates) if gates else 0
    
    @staticmethod
    def find_critical_path(dependency_graph: nx.DiGraph) -> List[int]:
        """
        Find critical path in dependency graph.
        
        Args:
            dependency_graph: Gate dependency DAG
        
        Returns:
            List of gate indices on critical path
        """
        if not dependency_graph.nodes():
            return []
        
        # Find longest path (critical path)
        try:
            path = nx.dag_longest_path(dependency_graph)
            return list(path)
        except:
            return []