import torch
import torch.nn as nn

class Conv1D(nn.Module):
    """Copy of Hugging Face's Conv1D layer to ensure exact compatibility"""
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        # Make sure all operations happen on the same device
        x_view = x.view(-1, x.size(-1))
        # Use matmul instead of addmm to avoid device issues
        result = x_view @ self.weight + self.bias
        return result.view(*size_out)