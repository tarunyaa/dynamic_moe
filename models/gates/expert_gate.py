from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ExpertGate(nn.Module, ABC):
    """
    Abstract base class for routing policies in Mixture of Experts.
    Subclasses should implement the `forward` method that takes
    routing weights and expert outputs, and returns a combined final output.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        routing_weights: torch.Tensor,           # [batch, seq_len, num_experts]
        expert_outputs_list: list                # list of [batch, seq_len, hidden_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            routing_weights (Tensor): Softmax scores over experts per token [B, T, E]
            expert_outputs_list (List[Tensor]): List of length `num_experts`,
                each of shape [B, T, D], from each expert.

        Returns:
            final_output (Tensor): Combined output of shape [B, T, D]
            aux_loss (Tensor): Optional auxiliary loss (e.g., load balancing), scalar
        """
        pass