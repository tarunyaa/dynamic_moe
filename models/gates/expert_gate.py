from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ExpertGate(ABC, nn.Module):
    """
    Abstract base class for routing policies in Mixture of Experts.
    Subclasses should implement the `forward` method that takes
    routing weights and returns selected weights and expert indices.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, routing_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            routing_weights (Tensor): Shape [batch, seq_len, num_experts], 
                                      softmax probabilities for each token.

        Returns:
            selected_weights (Tensor): Shape [batch, seq_len, k_or_p], normalized weights for selected experts
            selected_indices (Tensor): Shape [batch, seq_len, k_or_p], expert indices per token
        """
        pass
