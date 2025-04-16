import torch
import torch.nn as nn
import torch.nn.functional as F

class RLGate(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1, baseline=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.baseline = baseline

        # Policy network: logits over experts
        self.policy_net = nn.Linear(input_dim, num_experts)
        self.baseline_value = nn.Parameter(torch.tensor(0.0)) if baseline else None

    def forward(self, x, expert_outputs_list, rewards=None):
        """
        Args:
            x: [B, T, D] input tokens
            expert_outputs_list: list of [B, T, D] expert outputs
            rewards: [B, T] reward tensor, optional (needed during training)

        Returns:
            final_output: [B, T, D]
            aux_loss: scalar loss to train gating policy
        """
        B, T, D = x.shape
        device = x.device
        logits = self.policy_net(x)  # [B, T, E]
        probs = F.softmax(logits, dim=-1)

        # Sample top_k experts per token
        expert_indices = torch.multinomial(probs.view(-1, self.num_experts), self.top_k, replacement=False)
        expert_indices = expert_indices.view(B, T, self.top_k)  # [B, T, k]

        # Prepare output buffer
        final_output = torch.zeros(B, T, D, device=device)

        # Accumulate expert outputs
        for k in range(self.top_k):
            idx = expert_indices[:, :, k]  # [B, T]
            for e in range(self.num_experts):
                mask = (idx == e).unsqueeze(-1)  # [B, T, 1]
                final_output += mask * expert_outputs_list[e] / self.top_k

        # Compute policy gradient loss if rewards are provided
        aux_loss = torch.tensor(0.0, device=device)
        if rewards is not None:
            log_probs = torch.log(probs + 1e-9)
            selected_log_probs = log_probs.gather(-1, expert_indices)  # [B, T, k]
            selected_log_probs = selected_log_probs.sum(dim=-1)        # [B, T]

            if self.baseline:
                advantage = rewards - self.baseline_value
            else:
                advantage = rewards

            aux_loss = - (advantage.detach() * selected_log_probs).mean()

        return final_output, aux_loss
