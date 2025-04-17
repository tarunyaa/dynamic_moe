# gates/load_balance.py

import torch

def load_balancing_loss(gate_probs: torch.Tensor,
                        expert_mask: torch.Tensor,
                        eps: float = 1e-9) -> torch.Tensor:
    """
    Switch‑Transformer load‑balancing loss.
    gate_probs  (T, E): softmaxed routing probabilities
    expert_mask (T, E): bool mask (1 where token was dispatched)
    """
    # Importance and load per expert
    importance = gate_probs.sum(0)           # P_i
    load       = expert_mask.float().sum(0)  # L_i

    # Normalize
    p_hat = importance / (importance.sum() + eps)
    l_hat = load       / (load.sum()       + eps)

    # E * sum_i p̂_i * l̂_i
    return gate_probs.size(1) * (p_hat * l_hat).sum()
