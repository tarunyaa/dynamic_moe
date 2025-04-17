from gates.expert_gate import ExpertGate
import torch

class EntropyAdaptiveGate(ExpertGate):
    def __init__(self, num_experts, min_k, max_k):
        super().__init__()
        self.num_experts = num_experts
        self.min_k = min_k
        self.max_k = max_k

    def forward(self, routing_weights, expert_outputs_list):
        """
        Args:
            routing_weights: [B, T, E]
            expert_outputs_list: List of E tensors, each [B, T, D]

        Returns:
            final_output: [B, T, D]
            aux_loss: scalar (placeholder)
        """
        batch_size, seq_len, num_experts = routing_weights.shape
        input_dim = expert_outputs_list[0].shape[-1]
        device = routing_weights.device

        # Compute entropy of routing weights
        entropy = -(routing_weights * torch.log(routing_weights + 1e-9)).sum(dim=-1)  # [B, T]
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=device))
        norm_entropy = entropy / max_entropy

        # Determine dynamic k per token
        k_per_token = (norm_entropy * (self.max_k - self.min_k)).ceil().long() + self.min_k  # [B, T]

        # Sort expert weights to get top-k
        sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=True)  # [B, T, E]

        # Approximate by using top max_k and masking out extras
        selected_weights = sorted_weights[:, :, :self.max_k]  # [B, T, max_k]
        selected_indices = sorted_indices[:, :, :self.max_k]  # [B, T, max_k]

        # Create mask: [B, T, max_k]
        token_k = k_per_token.clamp(max=self.max_k).unsqueeze(-1)  # [B, T, 1]
        range_k = torch.arange(self.max_k, device=device).view(1, 1, -1)  # [1, 1, max_k]
        mask = (range_k < token_k)  # [B, T, max_k]

        # Mask and renormalize
        selected_weights = selected_weights * mask  # zero out unused
        selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1e-8)  # [B, T, max_k]

        # Final output aggregation
        final_output = torch.zeros(batch_size, seq_len, input_dim, device=device)

        for k in range(self.max_k):
            expert_ids = selected_indices[:, :, k]  # [B, T]
            weights = selected_weights[:, :, k]     # [B, T]
            mask_k = mask[:, :, k]                  # [B, T]

            for e in range(self.num_experts):
                select_mask = (expert_ids == e) & mask_k  # [B, T]
                weight_masked = weights * select_mask     # [B, T]
                expert_output = expert_outputs_list[e]    # [B, T, D]
                final_output += weight_masked.unsqueeze(-1) * expert_output

        aux_loss = torch.tensor(0.0, device=device)
        return final_output, aux_loss
