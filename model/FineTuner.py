import os
import torch
from transformers import Trainer, TrainingArguments
import json


class FineTuner:
    """
    Fine-tuning handler for GPT-2 models with Mixture of Experts.
    Provides optimized parameters for GPT-2 XL and manages the training process.
    """
    
    def __init__(
        self,
        model=None,
        train_dataset=None,
        eval_dataset=None,
        output_dir="./results",
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        report_to="none",
        seed=42
    ):
        """
        Initialize the fine-tuner with model and datasets.
        
        Args:
            model: Model to fine-tune (typically from ModelConverter)
            train_dataset: Training dataset (typically from DataLoader)
            eval_dataset: Evaluation dataset (typically from DataLoader)
            output_dir: Directory to save models and checkpoints
            logging_dir: Directory for logs
            fp16: Whether to use 16-bit floating point precision
            bf16: Whether to use bfloat16 precision (if supported)
            report_to: Logging backend ("none" by default)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.fp16 = fp16
        self.bf16 = bf16
        self.report_to = report_to
        self.seed = seed
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        
        self.trainer = None
        self.training_args = None
        self.training_successful = False
    
    def create_training_args(
        self,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        max_grad_norm=0.3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        **kwargs
    ):
        """
        Create training arguments optimized for GPT-2 XL.
        
        Returns:
            TrainingArguments object with optimized parameters
        """
        # Default arguments optimized for GPT-2 XL
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            logging_dir=self.logging_dir,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            max_grad_norm=max_grad_norm,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to=self.report_to,
            seed=self.seed,
            **kwargs
        )
        
        return self.training_args
    
    def setup_trainer(self, training_args=None):
        """
        Set up the Trainer with model, args, and datasets.
        
        Args:
            training_args: Custom training arguments or None to use defaults
            
        Returns:
            Configured Trainer instance
        """
        if self.model is None:
            raise ValueError("Model not provided. Set model before setting up trainer.")
            
        if self.train_dataset is None:
            raise ValueError("Training dataset not provided. Set train_dataset before setting up trainer.")
        
        # Use provided args or create default ones
        if training_args is None:
            if self.training_args is None:
                self.create_training_args()
            training_args = self.training_args
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        
        return self.trainer
    
    def train(self, resume_from_checkpoint=None):
        """
        Start training the model.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from or None
            
        Returns:
            Training result or None if training failed
        """
        if self.trainer is None:
            self.setup_trainer()
        
        print("\n========== STARTING FINETUNING ==========")
        print("Finetuning GPT-2 XL MoE model...")
        try:
            result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            self.training_successful = True
            print("\n========== FINETUNING COMPLETE ==========")
            return result
        except Exception as e:
            print(f"Training error: {e}")
            self.training_successful = False
            return None
    
    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Run setup_trainer() first.")
            
        if self.eval_dataset is None:
            raise ValueError("Evaluation dataset not provided.")
        
        print("\n========== EVALUATING MODEL ==========")
        return self.trainer.evaluate()
    
    def save_model(self, output_dir=None):
        """
        Save the fine-tuned model with MoE layers.
        
        Args:
            output_dir: Directory to save the model or None for default
            
        Returns:
            Path to saved model
        """
        if not self.training_successful:
            print("Warning: Saving model that didn't complete training successfully.")
            
        if self.trainer is None:
            raise ValueError("Trainer not set up. Nothing to save.")
        
        output_dir = output_dir or os.path.join(self.output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving model to {output_dir}")
        
        # 1. Save the model state dict directly using PyTorch
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # 2. Save model config (if available)
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
        
        # 3. Save tokenizer separately (if available via trainer)
        if hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer is not None:
            self.trainer.tokenizer.save_pretrained(output_dir)
        
        # 4. Save MoE configuration info for reloading
        moe_config = self._get_moe_config()
        with open(os.path.join(output_dir, "moe_config.json"), 'w') as f:
            json.dump(moe_config, f, indent=2)
        
        return output_dir
    
    def _get_moe_config(self):
        """
        Extract MoE configuration from the model.
        
        Returns:
            Dictionary containing MoE configuration
        """
        # Default MoE configuration
        moe_config = {
            "base_model": "gpt2-xl",  # Default assumption
            "moe_layers": []
        }
        
        # Try to extract MoE information
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            moe_layers = []
            for i, layer in enumerate(self.model.transformer.h):
                if hasattr(layer, 'mlp') and 'MixtureOfExperts' in str(type(layer.mlp)):
                    # Found MoE layer
                    moe_layer_info = {
                        "layer_idx": i,
                        "num_experts": layer.mlp.num_experts,
                        "top_k": layer.mlp.top_k
                    }
                    moe_layers.append(moe_layer_info)
            
            moe_config["moe_layers"] = moe_layers
            
            # Try to determine original model name
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
                moe_config["base_model"] = self.model.config.model_type
        
        return moe_config
    
    def load_model(self, model_path):
        """
        Load a saved model with MoE layers.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        from architecture.GPT2MoEBlock import GPT2MoEBlock
        from ModelConverter import ModelConverter
        
        # Check if MoE config exists
        moe_config_path = os.path.join(model_path, "moe_config.json")
        if not os.path.exists(moe_config_path):
            raise ValueError(f"MoE configuration not found at {moe_config_path}")
        
        # Load MoE configuration
        with open(moe_config_path, 'r') as f:
            moe_config = json.load(f)
        
        # Initialize a base model first
        base_model_name = moe_config.get("base_model", "gpt2-xl")
        converter = ModelConverter(model_name=base_model_name)
        model, _ = converter.load_model()
        
        # Load the state dict
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            raise ValueError(f"Model weights not found at {state_dict_path}")
        
        state_dict = torch.load(state_dict_path, map_location=converter.device)
        model.load_state_dict(state_dict)
        
        print(f"Successfully loaded model from {model_path}")
        self.model = model
        return model
    
    def set_model(self, model):
        """Set the model to be fine-tuned."""
        self.model = model
        if self.trainer is not None:
            self.trainer.model = model
    
    def set_datasets(self, train_dataset, eval_dataset=None):
        """Set the datasets for training and evaluation."""
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if self.trainer is not None:
            self.trainer.train_dataset = train_dataset
            self.trainer.eval_dataset = eval_dataset


# Example usage when run directly
if __name__ == "__main__":
    from ModelConverter import ModelConverter
    from DataLoader import DataLoader
    
    # Load and convert model
    converter = ModelConverter(model_name="gpt2-xl")
    model, _ = converter.convert_layers()
    
    # Load datasets in streaming mode
    data_loader = DataLoader(model_name="gpt2-xl")
    train_dataset, eval_dataset = data_loader.prepare_datasets()
    
    # Create fine-tuner with reduced memory usage
    tuner = FineTuner(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        fp16=False  # Disable fp16 to reduce memory usage
    )
    
    # Custom training args - adjusted for memory constraints
    tuner.create_training_args(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64, 
        eval_steps=500,
        save_steps=500,
        max_steps=5000,  # Reduced from 15000 to 5000
    )
    
    # Train and save model
    tuner.train()
    tuner.save_model()
