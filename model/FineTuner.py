import os
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter


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
        report_to="tensorboard",
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
            report_to: Logging backend ("tensorboard", "wandb", etc.)
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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
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
        dataloader_num_workers=2,
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
            evaluation_strategy=evaluation_strategy,
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
        Save the fine-tuned model.
        
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
        print(f"\nSaving model to {output_dir}")
        self.trainer.save_model(output_dir)
        return output_dir
    
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
            
    def launch_tensorboard(self, port=6006):
        """
        Launch TensorBoard to visualize training metrics.
        
        Args:
            port: Port to serve TensorBoard on (default: 6006)
            
        Returns:
            TensorBoard instance
        """
        try:
            from tensorboard import program
            import webbrowser
            import threading
            import time
            
            # Start TensorBoard server
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', self.logging_dir, '--port', str(port)])
            url = tb.launch()
            print(f"\n========== TENSORBOARD ==========")
            print(f"TensorBoard started at {url}")
            print(f"View training metrics at: http://localhost:{port}")
            
            # Open browser automatically
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            
            return tb
        except ImportError:
            print("\nTo use TensorBoard, install it with: pip install tensorboard")
            print("\nAfter installation, you can view training metrics with:")
            print(f"tensorboard --logdir {self.logging_dir}")
            return None


# Example usage when run directly
if __name__ == "__main__":
    from ModelConverter import ModelConverter
    from DataLoader import DataLoader
    
    # Load and convert model
    converter = ModelConverter(model_name="gpt2-xl")
    model, _ = converter.convert_layers()
    
    # Load datasets
    data_loader = DataLoader(model_name="gpt2-xl", subset_size=1000000)
    train_dataset, eval_dataset = data_loader.prepare_datasets()
    
    # Create fine-tuner with lower epochs for testing
    tuner = FineTuner(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Custom training args for testing
    tuner.create_training_args(
        num_train_epochs=1,
        eval_steps=500,
        save_steps=500
    )
    
    # Start TensorBoard before training (optional)
    tuner.launch_tensorboard()
    
    # Train and save model
    tuner.train()
    tuner.save_model()
