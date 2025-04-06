import torch
import torch.nn as nn
from .moe import MixtureOfExperts

class LLaMaMoEBlock(nn.Module):
    """Modified LLaMa block with MoE replacing the FFN, with all original parameters"""
    def __init__(self, original_block, num_experts=4, top_k=2):
        super().__init__()

        # Copy attention and normalization layers from original block
        self.self_attn = original_block.self_attn
        self.input_layernorm = original_block.input_layernorm
        self.post_attention_layernorm = original_block.post_attention_layernorm

        # Create MoE layer
        hidden_size = original_block.input_layernorm.weight.size(0)
        intermediate_size = original_block.mlp.gate_proj.out_features

        # Replace FFN with MoE
        self.mlp = MixtureOfExperts(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            num_experts=num_experts,
            top_k=top_k
        )

        # Initialize experts with original weights
        for expert in self.mlp.experts:
            # Initialize with the original weights but scaled down
            with torch.no_grad():
                expert.gate_proj.weight.copy_(original_block.mlp.gate_proj.weight / num_experts)
                expert.up_proj.weight.copy_(original_block.mlp.up_proj.weight / num_experts)
                expert.down_proj.weight.copy_(original_block.mlp.down_proj.weight / num_experts)

        # Initialize router with small random values
        nn.init.normal_(self.mlp.router.weight, mean=0.0, std=0.01)

        # Ensure all module parameters are on the same device as original block
        device = next(original_block.parameters()).device
        self.to(device)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):
        # Same forward pass as original LLaMa but with MoE
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )

        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # Add first residual connection
        hidden_states = attn_output + residual

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Use MoE layer - will return a tuple of (output, aux_loss)
        hidden_states, aux_loss = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs 