import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x):
        return self.mlp(x)


class TopPGatingMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_p=0.8):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            SimpleExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.top_p = top_p
        self.num_experts = num_experts

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_size)
        """
        B, T, H = x.shape
        logits = self.router(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)

        # Top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs < self.top_p
        mask[..., 0] = 1  # Always pick at least one

        # Sparse routing
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            # Find which tokens picked this expert
            routed_tokens = (sorted_indices == i) & mask
            if routed_tokens.any():
                weights = probs[routed_tokens]
                inputs = x[routed_tokens]
                out = self.experts[i](inputs)
                output[routed_tokens] += out * weights.unsqueeze(-1)
        return output
