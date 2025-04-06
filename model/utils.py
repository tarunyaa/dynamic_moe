import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import device

def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Load the specified model and tokenizer
    
    Args:
        model_name: Name of the Hugging Face model to load
        
    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer 