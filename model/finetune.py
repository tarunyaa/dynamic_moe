import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from .utils import load_model
from .run_experiment import run_moe_conversion
import logging
import math
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_function(examples, tokenizer, max_length=1024):
    """Preprocess the examples by tokenizing and formatting them for training"""
    # Tokenize the texts
    tokenized_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Dynamic MoE model on SlimPajama")
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model to load and convert to MoE",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cerebras/SlimPajama-627B",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="arxiv",
        help="Subset of the SlimPajama dataset (arxiv, books, c4, etc.)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of examples to use for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned-moe-slimpajama",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4,
        help="Number of experts in each MoE layer",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of experts to route to for each token",
    )
    parser.add_argument(
        "--conversion_frequency",
        type=int,
        default=4,
        help="Convert every nth layer to MoE",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps",
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 1: First convert the model to MoE architecture
    logger.info("Converting model to MoE architecture...")
    model, tokenizer = run_moe_conversion(
        model_name=args.model_name,
        num_experts=args.num_experts,
        top_k=args.top_k,
        conversion_frequency=args.conversion_frequency
    )
    
    if model is None or tokenizer is None:
        logger.error("Failed to convert model to MoE architecture")
        return
    
    # Step 2: Load and preprocess SlimPajama dataset
    logger.info(f"Loading SlimPajama dataset ({args.subset} subset)...")
    try:
        dataset = load_dataset(args.dataset_name, args.subset, split="train", streaming=True)
        # Take a subset for training (SlimPajama is very large)
        dataset = dataset.take(args.train_size)
        
        # Convert to regular (non-streaming) dataset for easier use with Trainer
        regular_dataset = list(dataset)
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": [item["text"] for item in regular_dataset]})
        
        # Split into train/validation
        train_val = dataset.train_test_split(test_size=0.05)
        train_dataset = train_val["train"]
        val_dataset = train_val["test"]
        
        logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Trying alternate approach with specific subset loading...")
        try:
            # Try a different approach for specific subsets
            dataset = load_dataset(args.dataset_name, split=f"train[:{args.train_size}]")
            train_val = dataset.train_test_split(test_size=0.05)
            train_dataset = train_val["train"]
            val_dataset = train_val["test"]
            logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
    
    # Step 3: Tokenize the dataset
    logger.info("Tokenizing dataset...")
    tokenized_train = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_val = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    
    # Step 4: Set up training arguments
    logger.info("Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Use CLM (Causal Language Modeling) instead of MLM
    )
    
    # Step 5: Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Step 6: Train the model
    logger.info("Starting fine-tuning...")
    try:
        trainer.train()
        
        # Step 7: Save the model
        logger.info(f"Fine-tuning complete! Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main() 