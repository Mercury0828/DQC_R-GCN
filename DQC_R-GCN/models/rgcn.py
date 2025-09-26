import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import numpy as np
from typing import Dict, List, Tuple

# 如果没有torch_geometric，使用简化版本
try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax
    USE_TORCH_GEOMETRIC = True
except ImportError:
    USE_TORCH_GEOMETRIC = False
    
    class MessagePassing(nn.Module):
        """Simple MessagePassing base class without torch_geometric."""
        def __init__(self, aggr='add'):
            super().__init__()
            self.aggr = aggr
        
        def propagate(self, edge_index, x, size=None):
            """Simplified propagation."""
            if edge_index.shape[1] == 0:
                return torch.zeros_like(x)
            
            # Simple aggregation
            src, dst = edge_index
            out = torch.zeros_like(x)
            
            for s, d in zip(src, dst):
                if self.aggr == 'add':
                    out[d] += x[s]
                elif self.aggr == 'mean':
                    out[d] += x[s]
            
            return out

class RGCNLayer(MessagePassing):
    """Relational Graph Convolutional Network Layer."""
    
    def __init__(self, in_channels, out_channels, num_relations, num_bases=None, 
                 activation=True, dropout=0.0):
        super(RGCNLayer, self).__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.activation = activation
        self.dropout = dropout
        
        # Use basis decomposition to reduce parameters
        if num_bases is not None:
            self.weight_bases = nn.Parameter(
                torch.Tensor(num_bases, in_channels, out_channels)
            )
            self.weight_combs = nn.Parameter(
                torch.Tensor(num_relations, num_bases)
            )
        else:
            self.weights = nn.Parameter(
                torch.Tensor(num_relations, in_channels, out_channels)
            )
        
        # Self-loop weight
        self.weight_self = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.num_bases is not None:
            nn.init.xavier_uniform_(self.weight_bases)
            nn.init.xavier_uniform_(self.weight_combs)
        else:
            nn.init.xavier_uniform_(self.weights)
        
        nn.init.xavier_uniform_(self.weight_self)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index_dict, size=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index_dict: Dictionary of edge indices by relation type
            size: Total number of nodes (for validation)
        """
        # Self-loop transformation
        out = torch.matmul(x, self.weight_self)
        
        # Get the actual number of nodes
        num_nodes = x.shape[0]
        
        # Message passing for each relation
        for rel_idx, (rel_name, edge_index) in enumerate(edge_index_dict.items()):
            if edge_index.size(1) == 0:
                continue
                
            # Validate edge indices
            if edge_index.numel() > 0:
                max_idx = edge_index.max().item()
                if max_idx >= num_nodes:
                    print(f"Warning: Edge index {max_idx} exceeds number of nodes {num_nodes} for relation {rel_name}")
                    # Filter out invalid edges
                    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                    edge_index = edge_index[:, mask]
                    if edge_index.size(1) == 0:
                        continue
            
            # Get relation-specific weight
            if self.num_bases is not None:
                # Basis decomposition
                weight = torch.sum(
                    self.weight_combs[rel_idx].unsqueeze(-1).unsqueeze(-1) * 
                    self.weight_bases, dim=0
                )
            else:
                weight = self.weights[rel_idx]
            
            # Transform and propagate
            x_transformed = torch.matmul(x, weight)
            
            # Use the correct size for propagation
            out += self.propagate(edge_index, x=x_transformed, size=(num_nodes, num_nodes))
        
        out += self.bias
        
        if self.activation:
            out = F.relu(out)
        
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out
    
    def message(self, x_j):
        """Message function."""
        return x_j
    
    def update(self, aggr_out):
        """Update function."""
        return aggr_out


class RGCN(nn.Module):
    """R-GCN encoder for DQC state representation."""
    
    def __init__(self, config):
        """
        Initialize R-GCN encoder.
        
        Args:
            config: Dictionary with model configuration
        """
        super(RGCN, self).__init__()
        
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.num_relations = 4  # depends_on, acts_on, assigned_to, communicates_with
        self.num_bases = config.get('num_bases', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # Node type dimensions (will be set dynamically)
        self.gate_dim = 2  # readiness, criticality
        self.qubit_dim = None  # 1 + num_qpus + 1
        self.qpu_dim = 2  # qubit_load, comm_load
        
        # Feature projection layers for different node types
        self.gate_proj = None
        self.qubit_proj = None
        self.qpu_proj = None
        
        # R-GCN layers
        self.rgcn_layers = nn.ModuleList()
        
        # Will be initialized when we know the input dimensions
        self._layers_initialized = False
    
    def _initialize_layers(self, num_qpus):
        """Initialize layers based on problem dimensions."""
        if self._layers_initialized:
            return
        
        # Set qubit dimension
        self.qubit_dim = 2 + num_qpus  # is_mapped + one_hot_qpu + interaction_strength
        
        # Feature projection layers
        self.gate_proj = nn.Linear(self.gate_dim, self.hidden_dim)
        self.qubit_proj = nn.Linear(self.qubit_dim, self.hidden_dim)
        self.qpu_proj = nn.Linear(self.qpu_dim, self.hidden_dim)
        
        # R-GCN layers
        for i in range(self.num_layers):
            in_dim = self.hidden_dim
            out_dim = self.hidden_dim
            
            layer = RGCNLayer(
                in_dim, out_dim,
                num_relations=self.num_relations,
                num_bases=self.num_bases,
                activation=(i < self.num_layers - 1),  # No activation on last layer
                dropout=self.dropout if i < self.num_layers - 1 else 0.0
            )
            self.rgcn_layers.append(layer)
        
        self._layers_initialized = True
    
    def _prepare_edges(self, edges_dict: Dict, num_gates: int, num_qubits: int, num_qpus: int) -> Dict:
        """
        Prepare edge indices for R-GCN.
        Need to adjust indices based on node type offsets.
        """
        edge_index_dict = {}
        
        # Calculate total number of nodes
        total_nodes = num_gates + num_qubits + num_qpus
        
        # Offset for each node type in concatenated embedding
        gate_offset = 0
        qubit_offset = num_gates
        qpu_offset = num_gates + num_qubits
        
        # Process each edge type
        for rel_name, edge_list in edges_dict.items():
            if not edge_list:
                edge_index_dict[rel_name] = torch.empty(2, 0, dtype=torch.long)
                continue
            
            edges = []
            for src, dst in edge_list:
                # Adjust indices based on node types
                if rel_name == 'depends_on':
                    # gate -> gate
                    src_idx = src + gate_offset
                    dst_idx = dst + gate_offset
                    # Validate indices
                    if src_idx >= num_gates or dst_idx >= num_gates:
                        continue
                elif rel_name == 'acts_on':
                    # gate -> qubit
                    src_idx = src + gate_offset
                    dst_idx = dst + qubit_offset
                    # Validate indices
                    if src >= num_gates or dst >= num_qubits:
                        continue
                elif rel_name == 'assigned_to':
                    # qubit -> qpu
                    src_idx = src + qubit_offset
                    dst_idx = dst + qpu_offset
                    # Validate indices
                    if src >= num_qubits or dst >= num_qpus:
                        continue
                elif rel_name == 'communicates_with':
                    # qpu -> qpu
                    src_idx = src + qpu_offset
                    dst_idx = dst + qpu_offset
                    # Validate indices
                    if src >= num_qpus or dst >= num_qpus:
                        continue
                else:
                    continue
                
                # Final validation - ensure indices are within total node range
                if src_idx >= total_nodes or dst_idx >= total_nodes:
                    print(f"Warning: Invalid edge {rel_name}: ({src_idx}, {dst_idx}) exceeds total nodes {total_nodes}")
                    continue
                    
                edges.append([src_idx, dst_idx])
            
            if edges:
                edge_index_dict[rel_name] = torch.tensor(edges, dtype=torch.long).t()
            else:
                edge_index_dict[rel_name] = torch.empty(2, 0, dtype=torch.long)
        
        return edge_index_dict

    def forward(self, state_repr: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass through R-GCN.
        
        Args:
            state_repr: Dictionary with node features and edges
        
        Returns:
            Dictionary with embeddings for each node type
        """
        # Extract features
        gate_features = torch.FloatTensor(state_repr['gate_features'])
        qubit_features = torch.FloatTensor(state_repr['qubit_features'])
        qpu_features = torch.FloatTensor(state_repr['qpu_features'])
        
        # Initialize layers if needed
        num_qpus = qpu_features.shape[0]
        self._initialize_layers(num_qpus)
        
        # Get actual counts
        num_gates = gate_features.shape[0]
        num_qubits = qubit_features.shape[0]
        num_qpus = qpu_features.shape[0]
        
        # Handle empty gate features
        if num_gates > 0:
            gate_embeds = self.gate_proj(gate_features)
        else:
            gate_embeds = torch.empty(0, self.hidden_dim)
        
        qubit_embeds = self.qubit_proj(qubit_features)
        qpu_embeds = self.qpu_proj(qpu_features)
        
        # Concatenate all node embeddings
        node_embeds = torch.cat([gate_embeds, qubit_embeds, qpu_embeds], dim=0)
        
        # Convert edges to tensor format with validation
        edge_index_dict = self._prepare_edges(state_repr['edges'], num_gates, num_qubits, num_qpus)
        
        # Pass through R-GCN layers with proper size
        total_nodes = node_embeds.shape[0]
        
        for layer in self.rgcn_layers:
            # Pass the correct size to avoid index errors
            node_embeds = layer(node_embeds, edge_index_dict, size=total_nodes)
            
        # Split embeddings back by node type
        gate_end = num_gates
        qubit_end = gate_end + num_qubits
        
        result = {
            'gate_embeddings': node_embeds[:gate_end] if num_gates > 0 else torch.empty(0, self.hidden_dim),
            'qubit_embeddings': node_embeds[gate_end:qubit_end],
            'qpu_embeddings': node_embeds[qubit_end:],
            'all_embeddings': node_embeds
        }
        
        return result