import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ValueNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, rgcn_encoder, config):
        """
        Initialize value network.
        
        Args:
            rgcn_encoder: R-GCN encoder (shared with policy)
            config: Network configuration
        """
        super(ValueNetwork, self).__init__()
        
        self.rgcn = rgcn_encoder
        self.hidden_dim = config.get('hidden_dim', 128)
        self.dropout = config.get('dropout', 0.1)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, state_repr: Dict) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            state_repr: State representation from environment
        
        Returns:
            State value estimate
        """
        # Get embeddings from R-GCN
        embeddings = self.rgcn(state_repr)
        
        # Global pooling - mean over all node embeddings
        global_embed = embeddings['all_embeddings'].mean(dim=0)
        
        # Pass through value head
        value = self.value_head(global_embed)
        
        return value.squeeze()