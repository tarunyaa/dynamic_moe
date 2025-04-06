import torch
from .utils import load_model
from .run_experiment import run_moe_conversion

def main():
    """Main entry point for the application"""
    print("Dynamic MoE LLM Adaptation")
    print("=" * 50)
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Run the MoE conversion experiment
    model, tokenizer = run_moe_conversion(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_experts=4,
        top_k=2,
        conversion_frequency=4
    )
    
    if model is not None and tokenizer is not None:
        print("Experiment completed successfully")
    else:
        print("Experiment failed")

if __name__ == "__main__":
    main() 