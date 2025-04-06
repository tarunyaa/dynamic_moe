import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert import ExpertLayer

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
        self.router = nn.Linear(input_dim, num_experts, bias=False)

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

        # Reshape input for expert processing
        x_flat = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]

        # Initialize output
        final_output = torch.zeros_like(x_flat)

        # No load balancing loss - just use a dummy zero tensor
        aux_loss = torch.tensor(0.0, device=device)

        # Process input with each expert
        for expert_idx in range(self.num_experts):
            # Create binary mask for this expert (ensure on correct device)
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # Get indices of tokens routed to this expert
            expert_indices = expert_mask.nonzero().squeeze(-1)

            # Get corresponding inputs
            expert_inputs = x_flat[expert_indices]

            # Process with expert
            expert_outputs = self.experts[expert_idx](expert_inputs)

            # Get routing weights for this expert
            weight_indices = (indices == expert_idx).long()  # Binary mask [batch, seq, top_k]
            weights = torch.zeros_like(routing_weights)
            weights = weights.scatter(-1, indices, routing_weights_k)
            expert_weights = weights[:, :, expert_idx].reshape(-1)  # [batch_size * seq_len]
            expert_weights = expert_weights[expert_indices]

            # Add weighted expert output to final output
            final_output[expert_indices] += expert_weights.unsqueeze(-1) * expert_outputs

        # Reshape back to original dimensions
        final_output = final_output.reshape(batch_size, seq_len, d_model)

        return final_output, aux_loss 