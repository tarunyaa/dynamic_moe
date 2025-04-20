import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, IterableDataset
import numpy as np
from tqdm import tqdm
import math
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch.nn.functional as F
from scipy import stats
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class C4Evaluator:
    def __init__(
        self,
        models: Dict[str, str],
        num_samples: int = 1000,
        max_length: int = 512,
        batch_size: int = 8,
        device: Optional[str] = None
    ):
        """
        Initialize the C4Evaluator with models and evaluation parameters.
        
        Args:
            models: Dictionary mapping model names to their HuggingFace model IDs
            num_samples: Number of samples to evaluate on
            max_length: Maximum sequence length for evaluation
            batch_size: Batch size for evaluation
            device: Device to run evaluation on (cuda/cpu)
        """
        self.models = models
        self.num_samples = num_samples
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results = defaultdict(dict)
        
        # Load C4 dataset
        logger.info("Loading C4 dataset...")
        self.dataset = load_dataset('c4', 'en', split='train', streaming=True)
        self.samples = list(self.dataset.take(num_samples))
        logger.info(f"Loaded {len(self.samples)} samples from C4 dataset")

    def _prepare_model(self, model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load and prepare model and tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            model.eval()
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Using EOS token as padding token for {model_id}")
            
            tokenizer.padding_side = "right"
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise

    def _get_expert_usage(self, model_outputs: Dict) -> Dict[str, torch.Tensor]:
        """Extract expert usage statistics from model outputs."""
        expert_usage = {}
        
        try:
            # Check for expert weights in different possible locations
            if hasattr(model_outputs, 'expert_weights'):
                expert_weights = model_outputs.expert_weights
            elif hasattr(model_outputs, 'last_hidden_state') and hasattr(model_outputs.last_hidden_state, 'expert_weights'):
                expert_weights = model_outputs.last_hidden_state.expert_weights
            elif isinstance(model_outputs, dict) and 'expert_weights' in model_outputs:
                expert_weights = model_outputs['expert_weights']
            else:
                logger.warning("No expert weights found in model outputs")
                return expert_usage
            
            if isinstance(expert_weights, torch.Tensor):
                expert_usage['weights'] = expert_weights
                expert_usage['num_experts'] = expert_weights.shape[-1]
                expert_usage['active_experts'] = (expert_weights > 0.1).sum(dim=-1)
                logger.debug(f"Found {expert_usage['num_experts']} experts")
            
        except Exception as e:
            logger.warning(f"Error extracting expert usage: {str(e)}")
        
        return expert_usage

    def evaluate_model(self, model_id: str) -> Dict[str, float]:
        """Evaluate a single model on all metrics."""
        tokenizer, model = self._prepare_model(model_id)
        metrics = {}
        
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        inference_times = []
        expert_usage_per_sample = []
        input_difficulties = []
        
        for i in tqdm(range(0, len(self.samples), self.batch_size),
                     desc=f"Evaluating model"):
            batch = self.samples[i:min(i + self.batch_size, len(self.samples))]
            texts = [item['text'] for item in batch]
            
            try:
                # Tokenize inputs
                inputs = tokenizer(
                    texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                ).to(self.device)
                
                # Time inference
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    
                    # Compute loss and accuracy
                    loss = outputs.loss
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Get expert usage
                    expert_usage = self._get_expert_usage(outputs)
                    if expert_usage:
                        expert_usage_per_sample.extend(
                            expert_usage['active_experts'].mean(dim=1).cpu().numpy()
                        )
                    
                    # Compute input difficulty
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    input_difficulties.extend(entropy.mean(dim=1).cpu().numpy())
                    
                    # Update metrics
                    n_tokens = inputs['input_ids'].numel()
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
                    total_correct += (predictions[:, :-1] == inputs['input_ids'][:, 1:]).sum().item()
                    
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
            except Exception as e:
                logger.warning(f"Error processing batch: {str(e)}")
                continue
        
        # Calculate final metrics
        metrics['perplexity'] = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        metrics['token_accuracy'] = total_correct / total_tokens if total_tokens > 0 else 0.0
        metrics['avg_inference_time'] = np.mean(inference_times) if inference_times else 0.0
        
        if expert_usage_per_sample and input_difficulties:
            correlation = stats.pearsonr(input_difficulties, expert_usage_per_sample)[0]
            metrics['difficulty_expert_correlation'] = correlation
            metrics['avg_experts_per_token'] = np.mean(expert_usage_per_sample)
            metrics['expert_usage_std'] = np.std(expert_usage_per_sample)
        
        return metrics

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Run evaluation for all models."""
        for model_name, model_id in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            try:
                metrics = self.evaluate_model(model_id)
                self.results[model_name].update(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        return self.results

    def print_results(self):
        """Print evaluation results in a formatted table."""
        if not self.results:
            logger.warning("No results to print. Run evaluate_all() first.")
            return
        
        metrics = [
            'perplexity',
            'token_accuracy',
            'avg_inference_time',
            'difficulty_expert_correlation',
            'avg_experts_per_token',
            'expert_usage_std'
        ]
        
        # Print header
        header = "Model".ljust(20) + " | " + " | ".join(metric.ljust(15) for metric in metrics)
        print("\nEvaluation Results")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        
        # Print results for each model
        for model_name in self.models:
            row = model_name.ljust(20) + " | "
            row += " | ".join(
                f"{self.results[model_name].get(metric, 'N/A'):>15.4f}"
                if isinstance(self.results[model_name].get(metric), (int, float))
                else "N/A".ljust(15)
                for metric in metrics
            )
            print(row)
        
        print("=" * len(header))

    def plot_results(self, save_path: Optional[str] = None):
        """Plot evaluation results."""
        if not self.results:
            logger.warning("No results to plot. Run evaluate_all() first.")
            return
        
        metric_groups = {
            'Performance': ['perplexity', 'token_accuracy'],
            'Efficiency': ['avg_inference_time', 'avg_experts_per_token'],
            'Adaptation': ['difficulty_expert_correlation', 'expert_usage_std']
        }
        
        fig, axes = plt.subplots(len(metric_groups), 1, figsize=(10, 5*len(metric_groups)))
        if len(metric_groups) == 1:
            axes = [axes]
        
        for i, (group, metrics) in enumerate(metric_groups.items()):
            data = []
            for metric in metrics:
                values = [self.results[model].get(metric, 0) for model in self.models]
                data.append(values)
            
            ax = axes[i]
            x = np.arange(len(self.models))
            width = 0.35
            
            for j, (metric, values) in enumerate(zip(metrics, data)):
                ax.bar(x + j*width, values, width, label=metric)
            
            ax.set_title(group)
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(list(self.models.keys()), rotation=45)
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
