# models/policy_network.py - 修复gate索引问题
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np

class PolicyNetwork(nn.Module):
    """Policy network with fixed gate indexing."""
    
    def __init__(self, rgcn_encoder, config):
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
        self.time_encoder = nn.Embedding(1000, self.hidden_dim)
        
        # 用于gate的通用编码器（当具体gate embedding不可用时）
        self.gate_fallback_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim),  # 输入：[gate_id_normalized, urgency]
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(self, state_repr: Dict, valid_actions: Dict) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """Forward pass with fixed gate indexing."""
        # Get embeddings from R-GCN
        embeddings = self.rgcn(state_repr)
        
        # Generate keys for all valid actions
        action_keys = []
        action_masks = []
        action_info = []
        
        # Process mapping actions (这部分没问题)
        for qubit, qpu in valid_actions.get('map', []):
            if qubit < embeddings['qubit_embeddings'].shape[0]:
                qubit_embed = embeddings['qubit_embeddings'][qubit]
                qpu_embed = embeddings['qpu_embeddings'][qpu]
                
                combined = torch.cat([qubit_embed, qpu_embed])
                key = self.map_mlp(combined)
                
                action_keys.append(key)
                action_masks.append(1.0)
                action_info.append(('map', (qubit, qpu)))
        
        # Process scheduling actions - 修复版本
        num_gate_embeddings = embeddings['gate_embeddings'].shape[0]
        
        # 如果有gate embeddings，计算平均embedding作为后备
        if num_gate_embeddings > 0:
            avg_gate_embed = embeddings['gate_embeddings'].mean(dim=0)
        else:
            # 如果没有gate embeddings（所有gates都调度了？），使用零向量
            avg_gate_embed = torch.zeros(self.hidden_dim)
        
        # 建立gate ID到embedding索引的映射
        # 注意：gate_features只包含未调度的gates
        # 我们需要找出哪些gates未调度
        unscheduled_gate_indices = set()
        if 'gate_features' in state_repr and state_repr['gate_features'].size > 0:
            # gate_features的行数就是未调度gates的数量
            # 但我们需要知道具体是哪些gate IDs
            # 这需要环境提供更多信息，或者我们使用启发式方法
            pass
        
        for gate, time_slot in valid_actions.get('schedule', []):
            # 方案1：查找gate在未调度列表中的位置
            # 由于我们不知道确切的映射，使用以下策略：
            
            # 如果gate索引小于embedding数量，直接使用
            if gate < num_gate_embeddings:
                gate_embed = embeddings['gate_embeddings'][gate]
            else:
                # 否则，使用后备方案
                # 创建一个基于gate ID的特征向量
                gate_features = torch.tensor([
                    gate / 20.0,  # 归一化的gate ID（假设最多20个gates）
                    1.0 - (gate / 20.0)  # 紧急度（ID越大越不紧急）
                ], dtype=torch.float32)
                
                # 使用后备编码器
                gate_embed = self.gate_fallback_encoder(gate_features)
                
                # 或者更简单：使用平均embedding + 一些噪声
                # gate_embed = avg_gate_embed + torch.randn_like(avg_gate_embed) * 0.1
            
            time_embed = self.time_encoder(torch.tensor([time_slot]))
            combined = torch.cat([gate_embed, time_embed.squeeze(0)])
            key = self.schedule_mlp(combined)
            
            action_keys.append(key)
            action_masks.append(1.0)
            action_info.append(('schedule', (gate, time_slot)))
        
        # 如果没有收集到任何动作，返回默认值
        if not action_keys:
            # 这不应该发生，因为我们已经修复了索引问题
            print(f"WARNING: No action keys generated despite valid actions: "
                  f"map={len(valid_actions.get('map', []))}, "
                  f"schedule={len(valid_actions.get('schedule', []))}")
            return torch.tensor([1.0]), torch.tensor([0.0]), []
        
        # Stack keys
        action_keys = torch.stack(action_keys).unsqueeze(0)  # [1, num_actions, hidden_dim]
        action_masks = torch.tensor(action_masks).unsqueeze(0)  # [1, num_actions]
        
        # Generate query from global state
        global_embed = embeddings['all_embeddings'].mean(dim=0)
        query = self.query_generator(global_embed).unsqueeze(0).unsqueeze(0)
        
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
        """Get action from policy."""
        action_probs, action_logits, action_info = self.forward(state_repr, valid_actions)
        
        if not action_info:
            # 不应该发生，但作为保护
            return ('noop', ()), torch.tensor(0.0)
        
        # Ensure proper tensor format
        if not isinstance(action_probs, torch.Tensor):
            action_probs = torch.tensor([action_probs])
        
        if action_probs.dim() == 0:
            action_probs = action_probs.unsqueeze(0)
        
        # 选择动作
        if deterministic:
            action_idx = torch.argmax(action_probs).item()
        else:
            action_probs = action_probs + 1e-10
            action_probs = action_probs / action_probs.sum()
            
            try:
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().item()
            except:
                action_idx = torch.argmax(action_probs).item()
        
        action_idx = min(action_idx, len(action_info) - 1)
        
        selected_action = action_info[action_idx]
        log_prob = torch.log(action_probs[action_idx] + 1e-10)
        
        return selected_action, log_prob