from gates.expert_gate import ExpertGate
import torch

class TopKGate(ExpertGate):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, routing_weights, expert_outputs_list):
        """
        Args:
            routing_weights: Tensor of shape [B, T, E] — softmax probabilities
            expert_outputs_list: List of expert outputs, each [B, T, D]

        Returns:
            final_output: Tensor [B, T, D]
            aux_loss: optional regularization (currently zero)
        """
        batch_size, seq_len, num_experts = routing_weights.shape
        device = routing_weights.device
        input_dim = expert_outputs_list[0].shape[-1]

        # Step 1: Select top-k experts per token
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)  # [B, T, k]

        # Step 2: Weighted combination
        final_output = torch.zeros((batch_size, seq_len, input_dim), device=device)

        for k in range(self.top_k):
            expert_idx = topk_indices[:, :, k]     # [B, T]
            weight = topk_weights[:, :, k]         # [B, T]

            for e in range(num_experts):
                mask = (expert_idx == e).unsqueeze(-1)       # [B, T, 1]
                expert_output = expert_outputs_list[e]       # [B, T, D]
                final_output += mask * weight.unsqueeze(-1) * expert_output

        aux_loss = torch.tensor(0.0, device=device)  # placeholder for load balancing
        return final_output, aux_loss
