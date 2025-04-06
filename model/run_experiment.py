import torch
import os
from . import device
from .utils import load_model
from .llama_moe import LLaMaMoEBlock

def run_moe_conversion(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_experts=4,
    top_k=2,
    conversion_frequency=4,
    test_prompt="<|user|>\nWrite a short poem about artificial intelligence.\n<|assistant|>"
):
    """
    Run the MoE conversion experiment
    
    Args:
        model_name: Name of the model to load
        num_experts: Number of experts in each MoE layer
        top_k: Number of experts to route to for each token
        conversion_frequency: Convert every nth layer (e.g. 4 means layers 0, 4, 8, etc.)
        test_prompt: Prompt to test generation with
        
    Returns:
        model: The converted model
        tokenizer: The tokenizer
    """
    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    
    # Prepare inputs for testing
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    # Initial test with original model
    print("\nTesting inference with original model")
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
            orig_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Original model generated text:\n{orig_text}")
            print("Original model test passed!")
        except Exception as e:
            print(f"Error with original model: {e}")
            print("Exiting as original model has issues.")
            return None, None

    # Determine which layers to convert
    total_layers = len(model.model.layers)
    print(f"\nTotal layers in model: {total_layers}")
    layers_to_convert = [i for i in range(0, total_layers, conversion_frequency)]
    print(f"Converting {len(layers_to_convert)} layers to MoE: {layers_to_convert}")

    # Create a dictionary to track conversion status
    conversion_status = {idx: False for idx in layers_to_convert}
    successful_layers = []

    # Convert only the selected layers
    for idx in layers_to_convert:
        print(f"\nConverting layer {idx} to MoE with {num_experts} experts and top-{top_k} routing")
        original_block = model.model.layers[idx]
        
        try:
            # Create MoE block
            moe_block = LLaMaMoEBlock(
                original_block,
                num_experts=num_experts,
                top_k=top_k
            )
            
            # Replace with MoE block
            model.model.layers[idx] = moe_block
            
            # Test after conversion
            print(f"Testing after converting layer {idx}")
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                    )
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"✓ Layer {idx} conversion successful!")
                    print(f"Sample output: {generated_text}")
                    conversion_status[idx] = True
                    successful_layers.append(idx)
                except Exception as e:
                    print(f"✗ Error after converting layer {idx}: {e}")
                    print(f"Reverting layer {idx} to original configuration")
                    model.model.layers[idx] = original_block
        except Exception as e:
            print(f"Error creating MoE block for layer {idx}: {e}")

    # Print conversion summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"Successfully converted {len(successful_layers)}/{len(layers_to_convert)} layers to MoE")
    print(f"Successful layers: {successful_layers}")
    print(f"New parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Final test with partially converted model
    print("\nFinal testing with converted model")
    with torch.no_grad():
        try:
            # Add detailed generation info
            print("Running final generation test with these parameters:")
            print("  - max_new_tokens: 100")
            print("  - do_sample: True")
            print("  - temperature: 0.7")
            print("  - top_p: 0.9")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Final model generated text:\n{final_text}")
            print("Final model test passed!")
            
            # Optional: Save the model
            if successful_layers:
                output_dir = f"tiny-llama-moe-sparse-{num_experts}experts-no-balancing"
                print(f"\nSaving model to {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print("Model saved successfully")
            else:
                print("\nNo layers were successfully converted, skipping model save.")
        except Exception as e:
            print(f"Error with final model: {e}")
            print("Final model has issues, consider using fewer MoE layers.")
    
    return model, tokenizer

if __name__ == "__main__":
    # Run the experiment with default parameters
    model, tokenizer = run_moe_conversion() 