from gates.expert_gate import ExpertGate
import torch

class TopAnyGate(ExpertGate):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts

        # Expert keys (cosine sim)
        self.expert_keys = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(num_experts, model_dim)))
        # Learnable per-expert gating thresholds G ∈ ℝ^E
        self.thresholds = torch.nn.Parameter(torch.zeros(num_experts))

    def forward(self, token_inputs, expert_outputs_list):
        """
        Args:
            token_inputs: Tensor [B, T, D]
            expert_outputs_list: List of E tensors [B, T, D]

        Returns:
            final_output: Tensor [B, T, D]
            aux_loss: scalar (currently zero)
        """
        B, T, D = token_inputs.shape
        E = self.num_experts
        device = token_inputs.device

        # Normalize inputs and expert keys
        x_norm = F.normalize(token_inputs, dim=-1)                     # [B, T, D]
        w_norm = F.normalize(self.expert_keys, dim=-1)                 # [E, D]

        # Compute cosine similarity: [B, T, E]
        sim_scores = torch.einsum('btd,ed->bte', x_norm, w_norm)

        # Apply sigmoid and subtract learnable thresholds
        gate_scores = torch.sigmoid(sim_scores)
        thresholded_scores = F.relu(gate_scores - torch.sigmoid(self.thresholds))  # [B, T, E]

        # Binary mask: 1 if score > threshold
        mask = thresholded_scores > 0  # [B, T, E]

        # Top-1 fallback if no experts are activated
        none_selected = ~mask.any(dim=-1)                  # [B, T]
        top1_idx = gate_scores.argmax(dim=-1)              # [B, T]
        mask[none_selected, top1_idx[none_selected]] = True

        # Normalize: equal weights per activated expert
        k = mask.sum(dim=-1, keepdim=True).clamp(min=1)    # [B, T, 1]
        norm_weights = mask.float() / k.float()            # [B, T, E]

        # Aggregate outputs
        final_output = torch.zeros((B, T, D), device=device)

        for e in range(E):
            weight = norm_weights[:, :, e].unsqueeze(-1)     # [B, T, 1]
            expert_output = expert_outputs_list[e]           # [B, T, D]
            final_output += weight * expert_output

        aux_loss = torch.tensor(0.0, device=device)
        return final_output, aux_loss
