import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    """Single expert layer that mimics LLaMa's SwiGLU FFN"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up) 