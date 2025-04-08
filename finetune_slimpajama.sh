#!/bin/bash

# Install dependencies if needed
cd model
poetry install

# Run fine-tuning
poetry run python -m finetune \
    --model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dataset_name="cerebras/SlimPajama-627B" \
    --subset="arxiv" \
    --train_size=10000 \
    --output_dir="./finetuned-moe-slimpajama" \
    --num_experts=4 \
    --top_k=2 \
    --conversion_frequency=4 \
    --learning_rate=5e-5 \
    --batch_size=8 \
    --epochs=3 \
    --max_length=1024 \
    --save_steps=1000 