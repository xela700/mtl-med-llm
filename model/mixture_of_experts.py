import torch
import torch.nn as nn
import torch.nn.functional as F
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