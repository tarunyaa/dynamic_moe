import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.GPT2MoEBlock import GPT2MoEBlock


class ModelConverter:
    """Converts a GPT-2 model to use Mixture of Experts layers."""
    
    def __init__(self, model_name="gpt2-xl", num_experts=4, top_k=2, device=None):
        """
        Initialize the converter with model parameters.
        
        Args:
            model_name: The HuggingFace model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
            num_experts: Number of experts in each MoE layer
            top_k: Number of experts to route to for each token
            device: Device to use (defaults to CUDA if available)
        """
        self.model_name = model_name
        self.num_experts = num_experts
        self.top_k = top_k
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.conversion_status = {}
        self.successful_layers = []
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Using device: {self.device}")
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model, self.tokenizer
    
    def convert_layers(self, layer_stride=4):
        """
        Convert selected layers to MoE.
        
        Args:
            layer_stride: Convert every nth layer (default: 4)
            
        Returns:
            Tuple of (model, successful_layers)
        """
        if self.model is None:
            self.load_model()
            
        # Determine which layers to convert
        total_layers = len(self.model.transformer.h)
        print(f"\nTotal layers in model: {total_layers}")
        layers_to_convert = [i for i in range(0, total_layers, layer_stride)]
        print(f"Converting {len(layers_to_convert)} layers to MoE: {layers_to_convert}")
        
        # Create a dictionary to track conversion status
        self.conversion_status = {idx: False for idx in layers_to_convert}
        self.successful_layers = []
        
        # Convert only the selected layers
        for idx in layers_to_convert:
            print(f"\nConverting layer {idx} to MoE with {self.num_experts} experts and top-{self.top_k} routing")
            original_block = self.model.transformer.h[idx]
            
            try:
                # Create MoE block
                moe_block = GPT2MoEBlock(
                    original_block,
                    num_experts=self.num_experts,
                    top_k=self.top_k
                )
                
                # Replace with MoE block
                self.model.transformer.h[idx] = moe_block
                
                # Mark as successful
                self.conversion_status[idx] = True
                self.successful_layers.append(idx)
                print(f"✓ Layer {idx} conversion completed")
                
            except Exception as e:
                print(f"Error creating MoE block for layer {idx}: {e}")
        
        # Print summary of conversions
        print(f"\nConversion Summary:")
        print(f"Successfully converted {len(self.successful_layers)} out of {len(layers_to_convert)} layers")
        print(f"Successful layer conversions: {self.successful_layers}")
        
        return self.model, self.successful_layers
    
    def get_model(self):
        """Return the converted model."""
        if not self.successful_layers:
            print("Warning: Model has not been converted yet. Run convert_layers() first.")
        return self.model