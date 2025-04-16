from gates.expert_gate import ExpertGate
import torch

class TopPGate(ExpertGate):
    def __init__(self, top_p):
        super().__init__()
        self.top_p = top_p

    def forward(self, routing_weights, expert_outputs_list):
        batch_size, seq_len, num_experts = routing_weights.shape
        device = routing_weights.device
        input_dim = expert_outputs_list[0].shape[-1]

        # Sort and compute cumulative probs
        sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_weights, dim=-1)
        mask = cumulative_probs <= self.top_p
        mask[..., 0] = True  # Always keep at least one expert

        # Apply mask
        selected_weights = sorted_weights * mask
        selected_indices = sorted_indices * mask  # still shape [B, T, E]

        # Normalize weights
        normalized_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True))

        final_output = torch.zeros((batch_size, seq_len, input_dim), device=device)

        for expert_id in range(num_experts):
            # Build mask where expert_id was selected
            mask_e = (selected_indices == expert_id)  # [B, T]
            weight_e = normalized_weights * mask_e    # [B, T]
            expert_output = expert_outputs_list[expert_id]  # [B, T, D]
            final_output += weight_e.sum(dim=-1).unsqueeze(-1) * expert_output

        aux_loss = torch.tensor(0.0, device=device)
        return final_output, aux_loss
