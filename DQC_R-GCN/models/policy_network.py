import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

class PolicyNetwork(nn.Module):
    """Policy network with unified attention mechanism for mixed action space."""
    
    def __init__(self, rgcn_encoder, config):
        """
        Initialize policy network.
        
        Args:
            rgcn_encoder: R-GCN encoder for state representation
            config: Network configuration
        """
        super(PolicyNetwork, self).__init__()
        
        self.rgcn = rgcn_encoder
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # MLPs for different action types
        self.map_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.schedule_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Query generation from global state
        self.query_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Time encoding for scheduling actions
        self.time_encoder = nn.Embedding(1000, self.hidden_dim)  # Max time slots
    
    def forward(self, state_repr: Dict, valid_actions: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            state_repr: State representation from environment
            valid_actions: Dictionary of valid actions
        
        Returns:
            action_probs: Probability distribution over actions
            action_logits: Raw logits for actions
        """
        # Get embeddings from R-GCN
        embeddings = self.rgcn(state_repr)
        
        # Generate keys for all valid actions
        action_keys = []
        action_masks = []
        action_info = []  # Store action type and parameters
        
        # Process mapping actions
        for qubit, qpu in valid_actions.get('map', []):
            if qubit < embeddings['qubit_embeddings'].shape[0]:
                qubit_embed = embeddings['qubit_embeddings'][qubit]
                qpu_embed = embeddings['qpu_embeddings'][qpu]
                
                # Concatenate embeddings and pass through MLP
                combined = torch.cat([qubit_embed, qpu_embed])
                key = self.map_mlp(combined)
                
                action_keys.append(key)
                action_masks.append(1.0)
                action_info.append(('map', (qubit, qpu)))
        
        # Process scheduling actions
        for gate, time_slot in valid_actions.get('schedule', []):
            if gate < embeddings['gate_embeddings'].shape[0]:
                gate_embed = embeddings['gate_embeddings'][gate]
                time_embed = self.time_encoder(torch.tensor([time_slot]))
                
                # Concatenate embeddings and pass through MLP
                combined = torch.cat([gate_embed, time_embed.squeeze(0)])
                key = self.schedule_mlp(combined)
                
                action_keys.append(key)
                action_masks.append(1.0)
                action_info.append(('schedule', (gate, time_slot)))
        
        if not action_keys:
            # No valid actions - return uniform distribution
            return torch.tensor([1.0]), torch.tensor([0.0]), []
        
        # Stack keys
        action_keys = torch.stack(action_keys).unsqueeze(0)  # [1, num_actions, hidden_dim]
        action_masks = torch.tensor(action_masks).unsqueeze(0)  # [1, num_actions]
        
        # Generate query from global state
        global_embed = embeddings['all_embeddings'].mean(dim=0)  # Simple pooling
        query = self.query_generator(global_embed).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            query, action_keys, action_keys,
            key_padding_mask=(1 - action_masks).bool()
        )
        
        # Get action logits
        action_logits = torch.matmul(attn_output.squeeze(0), action_keys.squeeze(0).T)
        action_logits = action_logits.squeeze()
        
        # Apply masking and softmax
        masked_logits = action_logits.masked_fill(action_masks.squeeze() == 0, float('-inf'))
        action_probs = F.softmax(masked_logits, dim=-1)
        
        return action_probs, action_logits, action_info
    
    def get_action(self, state_repr: Dict, valid_actions: Dict, deterministic: bool = False):
        """
        Get action from policy.
        
        Args:
            state_repr: State representation
            valid_actions: Valid actions
            deterministic: Whether to use deterministic policy
        
        Returns:
            Selected action and its log probability
        """
        action_probs, action_logits, action_info = self.forward(state_repr, valid_actions)
        
        # Handle edge cases
        if not action_info:
            # No valid actions available
            return ('noop', ()), torch.tensor(0.0)
        
        # Ensure action_probs is a proper tensor
        if not isinstance(action_probs, torch.Tensor):
            action_probs = torch.tensor([action_probs])
        
        # Flatten if needed
        if action_probs.dim() == 0:
            action_probs = action_probs.unsqueeze(0)
        
        if deterministic:
            action_idx = torch.argmax(action_probs).item()
        else:
            # Add small epsilon to avoid log(0)
            action_probs = action_probs + 1e-10
            # Normalize to ensure it's a valid probability distribution
            action_probs = action_probs / action_probs.sum()
            
            try:
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().item()
            except:
                # If distribution fails, just take the best action
                action_idx = torch.argmax(action_probs).item()
        
        # Ensure action_idx is within bounds
        action_idx = min(action_idx, len(action_info) - 1)
        
        selected_action = action_info[action_idx]
        log_prob = torch.log(action_probs[action_idx])
        
        return selected_action, log_prob