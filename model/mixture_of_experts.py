"""
Module to create Mixture-of-Experts architecture for LLM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import Tensor

class MoEProjectionLayer(nn.Module):
    """
    Mixture-of-Experts projection layer that allows for modification of number of experts and
    number of selected experts. Designed to be fitted into projection layer of the model.

    Attributes:
        num_experts (int): number of experts to create (default 4)
        top_k (int): number of top experts to select (default 2)
        experts (ModuleList): list of projection modules that comprise the MoE architecture
        gate (nn.Linear): gating network for selecting which experts are active
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int=4, top_k: int=2, dropout: float=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Creates experts with own projection layers
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
                nn.LayerNorm(input_dim)
            ) for i in range(num_experts)
        ])

        # Gating network for choosing experts
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: Tensor) -> Tensor:
        """
        Modified forward pass that uses gate network to select k experts. Set up to use same MLP config as is in classification wrapper.

        Args:
            x (Tensor): Input tensor to the forward pass [batch, hidden_dim]
        
        Returns:
            Tensor: Output based on MoE [batch, hidden_dim]
        """
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Select top-k experts per sample. Sparse setup.
        top_k_vals, top_k_index = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_vals = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        for i in range(self.top_k):
            index = top_k_index[:, i]
            weight = top_k_vals[:, i].unsqueeze(-1)
            for expert_id in range(self.num_experts):
                mask = (index == expert_id).float().unsqueeze(-1) # Controlling which experts contribute
                if mask.sum() > 0:
                    expert_output = self.experts[expert_id](x)
                    output += expert_output * weight * mask

        return output

class MixedMoEProjectionLayer(nn.Module):
    """
    Mixture-of-Experts projection layer that allows for modification of number of experts and
    number of selected experts. Designed to be fitted into projection layer of the model.

    ## This version of the projection is intended to diversify experts (modules) in depth and activation types

    Attributes:
        num_experts (int): number of experts to create (default 4)
        top_k (int): number of top experts to select (default 2)
        experts (ModuleList): list of projection modules that comprise the MoE architecture
        gate (nn.Linear): gating network for selecting which experts are active
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Lists representing different potential combinations of expert attributes

        activations = [nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU] # types of activations
        depths = [1, 2, 3] # number of projection layers
        hidden_scales = [0.5, 1.0, 1.5] # scaling on hidden dimension

        self.experts = nn.ModuleList()
        for i in enumerate(num_experts):
            activation = activations[i % len(activations)]
            depth = random.choice(depths)
            scale = random.choice(hidden_scales)
            hid_dim = int(hidden_dim * scale)

            layers = []
            in_dim = input_dim

            # partially random construction
            for j in range(depth):
                layers.append(nn.Linear(in_dim, hid_dim))
                layers.append(activation())
                layers.append(nn.LayerNorm(hid_dim))
                in_dim = hid_dim
            
            # converting back to shape of input
            layers.append(nn.Linear(in_dim, input_dim))
            layers.append(nn.LayerNorm(input_dim))

            expert = nn.Sequential(*layers)
            self.experts.append(expert)
        
        self.gate = nn.Linear(input_dim, num_experts) # simple linear gating network
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Modified forward pass that uses gate network to select k experts. Set up to use same MLP config as is in classification wrapper.

        Args:
            x (Tensor): Input tensor to the forward pass [batch, hidden_dim]
        
        Returns:
            Tensor: Output based on MoE [batch, hidden_dim]
        """
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        top_k_vals, top_k_index = torch.topk(gate_scores, self.top_k, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter_(1, top_k_index, 1.0)
        gate_scores = gate_scores * mask
        gate_scores = gate_scores / (gate_scores.sum(dim=-1, keepdim=True) + 1e-9)

        outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            outputs.append(expert_out * gate_scores[:, i].unsqueeze(-1))
        
        return torch.stack(outputs, dim=0).sum(dim=0)