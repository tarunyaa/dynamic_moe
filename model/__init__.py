import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Import classes
from .expert import ExpertLayer
from .moe import MixtureOfExperts
from .llama_moe import LLaMaMoEBlock

__all__ = ["ExpertLayer", "MixtureOfExperts", "LLaMaMoEBlock", "device"] 