from gates.expert_gate import ExpertGate
import torch

class TopPGate(ExpertGate):
    def __init__(self, top_p=0.9):
        super().__init__()
        self.top_p = top_p

    def forward(self, routing_weights):
        sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_weights, dim=-1)
        mask = cumulative_probs <= self.top_p
        mask[..., 0] = True  # Ensure at least one expert
        selected_weights = sorted_weights * mask
        selected_indices = sorted_indices * mask
        normalized_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return normalized_weights, selected_indices
