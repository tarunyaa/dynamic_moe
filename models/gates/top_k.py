# gates/top_k.py

from gates.expert_gate import ExpertGate
import torch
from gates.load_balance import load_balancing_loss

class TopKGate(ExpertGate):
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def forward(self, routing_weights: torch.Tensor, expert_outputs_list: list[torch.Tensor]):
        """
        Args:
            routing_weights: [B, T, E] — softmax probabilities
            expert_outputs_list: list of E tensors [B, T, D]
        Returns:
            final_output: [B, T, D]
            lb_loss: scalar load‑balancing loss
        """
        B, T, E = routing_weights.shape
        D = expert_outputs_list[0].shape[-1]
        device = routing_weights.device

        # 1) pick top‑k per token
        topk_w, topk_i = torch.topk(routing_weights, self.top_k, dim=-1)   # [B, T, k]
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-9)

        # 2) assemble output
        out = torch.zeros((B, T, D), device=device)
        for k in range(self.top_k):
            idx = topk_i[:, :, k]      # [B, T]
            w   = topk_w[:, :, k]      # [B, T]
            for e in range(E):
                mask = (idx == e).unsqueeze(-1)  # [B, T, 1]
                out += mask * w.unsqueeze(-1) * expert_outputs_list[e]

        # 3) build dispatch mask for LB
        # shape [B, T, E], True where token→expert
        expert_mask = torch.zeros_like(routing_weights, dtype=torch.bool)
        expert_mask.scatter_(2, topk_i, True)

        # 4) flatten to [B*T, E]
        gate_flat = routing_weights.view(-1, E)
        mask_flat = expert_mask.view(-1, E)

        # 5) compute LB loss
        lb_loss = load_balancing_loss(gate_flat, mask_flat)

        return out, lb_loss
