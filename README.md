# Dynamic MoE

A framework for dynamically converting Transformer-based models into Mixtures of Experts (MoE) architectures.

## Model Directory

The `model/` directory contains the core implementation for converting standard transformer models (specifically LLaMA-based models) into dynamic Mixture of Experts architectures.

### Files Overview

- **__init__.py**: Exports the main classes and sets up the device (CUDA if available, CPU otherwise).
- **main.py**: Entry point for the application, with a simple interface to run MoE conversion experiments.
- **moe.py**: Contains the `MixtureOfExperts` class, a general-purpose MoE implementation with top-k routing.
- **expert.py**: Implements `ExpertLayer`, the individual expert modules used within MoE layers.
- **llama_moe.py**: Contains `LLaMaMoEBlock`, a modified LLaMA transformer block with MoE replacing the standard FFN.
- **run_experiment.py**: Orchestrates experiments for converting standard models to MoE, with layer-by-layer testing.
- **utils.py**: Utility functions for loading models and other helper functions.
- **pyproject.toml** & **poetry.lock**: Package dependency management for the project.

### Key Components

#### MixtureOfExperts

The core MoE implementation that:
- Uses a gating/routing mechanism to direct tokens to the most relevant experts
- Supports top-k routing (can select multiple experts per token)
- Maintains the same interface as a standard feed-forward neural network

#### LLaMaMoEBlock

A drop-in replacement for standard LLaMA transformer blocks that:
- Preserves the original attention mechanism
- Replaces the feed-forward network with an MoE layer
- Maintains compatibility with the original model's interface
- Initializes expert weights from the original model's parameters

#### Experiment Framework

The `run_experiment.py` file provides:
- Automatic conversion of standard models to MoE architectures
- Layer-by-layer testing to ensure model stability
- Options to control how many and which layers to convert
- Fallback mechanisms to revert problematic conversions

### Usage

To convert a model to use MoE:

```python
from model.run_experiment import run_moe_conversion

model, tokenizer = run_moe_conversion(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_experts=4,        # Number of experts per MoE layer
    top_k=2,              # How many experts to route each token to
    conversion_frequency=4  # Convert every 4th layer
)
```

The framework automatically handles model loading, conversion, testing, and can optionally save the converted model.

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)

### Environment Setup

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dynamic-moe.git
   cd dynamic-moe
   ```

3. Install dependencies:
   ```bash
   cd model
   poetry install
   ```

### Running the Project

#### Infrastructure

It is recommended to run this project in a GPU-acceleratd environment. See this guide to launch and SSH into a GPU-accelerated EC2 instance: https://docs.google.com/document/d/1Hky7NQRuBwpyDnrvl4j9ksPi_ORabHdHzkXRHhqhxZg/edit?tab=t.0


#### Running from Command Line

To run the default conversion experiment:

```bash
cd model
poetry run python -m main
```

This will:
- Load the TinyLlama-1.1B-Chat model
- Convert layers to MoE architecture
- Run tests to ensure stability
- Save the converted model

#### Customizing the Conversion

To customize the conversion, you can modify the parameters in `model/main.py` or create your own script:

```python
import torch
from model.run_experiment import run_moe_conversion

# Configure your experiment
model, tokenizer = run_moe_conversion(
    model_name="facebook/opt-350m",  # Choose your model
    num_experts=8,                  # Increase number of experts
    top_k=3,                        # Use top-3 routing
    conversion_frequency=2,         # Convert every 2nd layer
    test_prompt="Your custom prompt for testing"
)

# Use the converted model
if model and tokenizer:
    # Your inference code here
    inputs = tokenizer("Hello, I'm a converted MoE model!", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Running from Python

You can also import and use the framework in your own Python code:

```bash
cd model
poetry run python
```

```python
from model.run_experiment import run_moe_conversion

# Run your custom conversion
model, tokenizer = run_moe_conversion(
    model_name="gpt2",
    num_experts=4,
    top_k=2
)
