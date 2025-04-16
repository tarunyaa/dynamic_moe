from gates.expert_gate import ExpertGate
import torch

class TopKGate(ExpertGate):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, routing_weights):
        topk_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        normalized_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return normalized_weights, indices