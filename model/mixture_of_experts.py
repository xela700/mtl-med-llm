"""
Module to create Mixture-of-Experts architecture for LLM training.

Prior version of this mudule created experts that were homogeneous in depth and activation type.
"""

import torch
import torch.nn as nn
from torch import Tensor

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
        for i in range(num_experts):
            activation = activations[i % len(activations)]
            depth = depths[i % len(depths)]
            scale = hidden_scales[i % len(hidden_scales)]
            hid_dim = int(hidden_dim * scale)

            layers = []
            in_dim = input_dim

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
        for b in range(x.size(0)):
            sample_output = 0
            for idx, g in zip(top_k_index[b], top_k_vals[b]):
                expert_output = self.experts[idx](x[b].unsqueeze(0))
                sample_output += g * expert_output
            outputs.append(sample_output)
        
        return torch.cat(outputs, dim=0)