from gates.expert_gate import ExpertGate

class EntropyAdaptiveGate(ExpertGate):
    def __init__(self, num_experts, min_k=1, max_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.min_k = min_k
        self.max_k = max_k

    def forward(self, routing_weights):
        # Compute entropy per token
        entropy = -(routing_weights * torch.log(routing_weights + 1e-9)).sum(dim=-1)  # [batch, seq_len]
        
        # Map entropy to k per token
        max_entropy = torch.log(torch.tensor(self.num_experts, device=routing_weights.device))
        norm_entropy = entropy / max_entropy
        k_per_token = (norm_entropy * (self.max_k - self.min_k)).ceil().long() + self.min_k  # [batch, seq_len]

        # Now select top-k for each token (hard part)
        # This is dynamic top-k per token — requires a batched top-k function
        # Approximation: use fixed top-k=max_k and mask out unused weights later
        sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=True)

        # Build a mask to keep only top `k_per_token[i, j]` entries per token
        mask = torch.arange(self.max_k, device=routing_weights.device).unsqueeze(0).unsqueeze(0)  # [1, 1, max_k]
        k_clip = k_per_token.clamp(max=self.max_k).unsqueeze(-1)                                # [batch, seq_len, 1]
        mask = mask < k_clip                                                                     # [batch, seq_len, max_k]

        selected_weights = sorted_weights[:, :, :self.max_k] * mask
        selected_indices = sorted_indices[:, :, :self.max_k] * mask

        selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return selected_weights, selected_indices
