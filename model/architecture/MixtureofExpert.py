import torch
import torch.nn as nn
import torch.nn.functional as F
from .ExpertLayer import ExpertLayer
from .Conv1D import Conv1D

class MixtureOfExperts(nn.Module):
    """Simple MoE layer with top-k routing - no load balancing loss"""
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim)
            for _ in range(num_experts)
        ])

        # Create router
        self.router = Conv1D(num_experts, input_dim)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        device = x.device  # Get device from input tensor

        # Calculate routing probabilities (ensure on correct device)
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)

        # Get top-k routing weights and indices
        routing_weights_k, indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize weights
        routing_weights_k = routing_weights_k / routing_weights_k.sum(dim=-1, keepdim=True)

        # Process input once and cache
        expert_outputs_list = []
        for expert_idx in range(self.num_experts):
            # Process through expert - for all inputs initially
            expert_output = self.experts[expert_idx](x)
            expert_outputs_list.append(expert_output)  # [batch_size, seq_len, d_model]

        # Initialize output tensor
        final_output = torch.zeros_like(x, device=device)

        # No load balancing loss - just use a dummy zero tensor
        aux_loss = torch.tensor(0.0, device=device)

        # Assign outputs based on routing
        for k in range(self.top_k):
            # For each position in the top-k selection
            expert_idx = indices[:, :, k]  # [batch_size, seq_len]
            # Get the corresponding weight
            weight = routing_weights_k[:, :, k]  # [batch_size, seq_len]
            
            # Create a mask for each expert
            for e in range(self.num_experts):
                # Binary mask where this expert was selected
                mask = (expert_idx == e).unsqueeze(-1)  # [batch_size, seq_len, 1]
                # Get the output for this expert
                expert_output = expert_outputs_list[e]
                # Weight the output and add it where this expert was selected
                # Broadcasting handles the dimension expansion
                final_output = final_output + (mask * weight.unsqueeze(-1) * expert_output)

        return final_output, aux_loss