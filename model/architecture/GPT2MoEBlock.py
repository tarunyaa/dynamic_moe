import torch
import torch.nn as nn
from .MixtureofExpert import MixtureOfExperts

# Modified GPT-2 block with MoE that handles all required parameters
class GPT2MoEBlock(nn.Module):
    """Modified GPT-2 block with MoE replacing the MLP"""
    def __init__(self, original_block, num_experts=4, top_k=2):
        super().__init__()

        # Copy attention and normalization layers from original block
        self.attn = original_block.attn
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        
        # Get original dimensions
        hidden_size = original_block.ln_1.weight.size(0)  # 768 for base GPT-2
        intermediate_size = original_block.mlp.c_fc.weight.shape[1]  # 3072 for base GPT-2

        # Replace MLP with MoE
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
                # Ensure device consistency during copying
                expert.c_fc.weight.data = original_block.mlp.c_fc.weight.clone().to(expert.c_fc.weight.device) / num_experts
                expert.c_fc.bias.data = original_block.mlp.c_fc.bias.clone().to(expert.c_fc.bias.device) / num_experts
                expert.c_proj.weight.data = original_block.mlp.c_proj.weight.clone().to(expert.c_proj.weight.device) / num_experts
                expert.c_proj.bias.data = original_block.mlp.c_proj.bias.clone().to(expert.c_proj.bias.device) / num_experts

        # Initialize router with small random values
        with torch.no_grad():
            nn.init.normal_(self.mlp.router.weight, mean=0.0, std=0.01)

        # Ensure all module parameters are on the same device as original block
        device = next(original_block.parameters()).device
        self = self.to(device)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs  # Accept any additional arguments
    ):
        """Forward pass that handles all possible arguments from the Trainer"""
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Prepare arguments for attention layer
        attn_args = {
            "hidden_states": hidden_states,
            "layer_past": layer_past,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
        }
        
        # For encoder-decoder models if used
        if encoder_hidden_states is not None:
            attn_args["encoder_hidden_states"] = encoder_hidden_states
            attn_args["encoder_attention_mask"] = encoder_attention_mask
        
        # Add any kwargs that might be passed to the attention layer
        for key in kwargs:
            if key in ["position_ids", "position_embeddings"]:
                attn_args[key] = kwargs[key]
        
        attn_outputs = self.attn(**attn_args)
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        # First residual connection
        hidden_states = attn_output + residual
        
        # Second block - MoE instead of MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        
        # Use MoE layer
        hidden_states_moe, aux_loss = self.mlp(hidden_states)
        hidden_states = hidden_states_moe + residual
        
        # GPT-2 style outputs
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
            
        return outputs