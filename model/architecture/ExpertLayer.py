import torch
import torch.nn as nn
from .Conv1D import Conv1D

class ExpertLayer(nn.Module):
    """Single expert layer that mimics GPT-2's MLP using Conv1D"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.c_fc = Conv1D(hidden_dim, input_dim)
        self.c_proj = Conv1D(input_dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")  # GPT-2 uses GELU activation

    def forward(self, x):
        # Apply each operation while preserving input dimensions
        hidden = self.c_fc(x)
        hidden = self.act(hidden)
        output = self.c_proj(hidden)
        return output