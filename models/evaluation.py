import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
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

class ModelEvaluator:
    def __init__(
        self,
        models: Dict[str, str],
        dataset: Dataset,
        max_length: int = 512,
        batch_size: int = 8,
        device: Optional[str] = None,
        task_type: str = "language_modeling"  # or "qa"
    ):
        """
        Initialize the ModelEvaluator with models and dataset.
        
        Args:
            models: Dictionary mapping model names to their HuggingFace model IDs
            dataset: Dataset to evaluate on
            max_length: Maximum sequence length for evaluation
            batch_size: Batch size for evaluation
            device: Device to run evaluation on (cuda/cpu)
            task_type: Type of task ("language_modeling" or "qa")
        """
        self.models = models
        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task_type = task_type
        self.results = defaultdict(dict)
        
    def _prepare_model(self, model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load and prepare model and tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            model.eval()
            
            # Set up padding token if not already set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Using EOS token as padding token for {model_id}")
                else:
                    # Add a new padding token
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    # Resize model's token embeddings
                    model.resize_token_embeddings(len(tokenizer))
                    logger.info(f"Added new padding token for {model_id}")
            
            # Verify padding token is set
            if tokenizer.pad_token is None:
                raise ValueError(f"Failed to set padding token for {model_id}")
            
            # Set padding side to right for causal language models
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
            
            # Process expert weights
            if isinstance(expert_weights, torch.Tensor):
                expert_usage['weights'] = expert_weights
                expert_usage['num_experts'] = expert_weights.shape[-1]
                # Count experts with weight > 0.1 as active
                expert_usage['active_experts'] = (expert_weights > 0.1).sum(dim=-1)
                logger.debug(f"Found {expert_usage['num_experts']} experts")
            else:
                logger.warning(f"Expert weights is not a tensor: {type(expert_weights)}")
            
        except Exception as e:
            logger.warning(f"Error extracting expert usage: {str(e)}")
        
        return expert_usage

    def compute_perplexity(self, model_id: str) -> float:
        """Compute perplexity score for a model."""
        tokenizer, model = self._prepare_model(model_id)
        total_loss = 0.0
        total_tokens = 0
        
        for i in tqdm(range(0, len(self.dataset), self.batch_size), 
                     desc=f"Computing perplexity for {model_id}"):
            batch = self.dataset[i:i + self.batch_size]
            texts = batch['text']
            
            try:
                inputs = tokenizer(
                    texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    n_tokens = inputs['input_ids'].numel()
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
                    
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {str(e)}")
                continue
        
        avg_neg_log_likelihood = total_loss / total_tokens
        perplexity = math.exp(avg_neg_log_likelihood)
        return perplexity

    def compute_accuracy_metrics(self, model_id: str) -> Dict[str, float]:
        """Compute accuracy metrics based on task type."""
        tokenizer, model = self._prepare_model(model_id)
        metrics = {}
        
        if self.task_type == "qa":
            # Implement QA-specific metrics (F1, exact match)
            pass
        else:
            # Language modeling metrics
            total = 0
            correct = 0
            
            for i in tqdm(range(0, len(self.dataset), self.batch_size),
                         desc=f"Computing accuracy for {model_id}"):
                batch = self.dataset[i:i + self.batch_size]
                texts = batch['text']
                
                try:
                    encoded = tokenizer(
                        texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                        padding=True
                    ).to(self.device)
                    
                    input_ids = encoded["input_ids"]
                    labels = input_ids.clone()
                    
                    with torch.no_grad():
                        outputs = model(**encoded)
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        
                        correct += (predictions[:, 1:] == labels[:, 1:]).sum().item()
                        total += labels[:, 1:].numel()
                        
                except Exception as e:
                    logger.warning(f"Error processing batch {i}: {str(e)}")
                    continue
            
            metrics['token_accuracy'] = correct / total if total > 0 else 0.0
        
        return metrics

    def compute_efficiency_metrics(self, model_id: str) -> Dict[str, float]:
        """Compute efficiency metrics including expert usage and inference time."""
        tokenizer, model = self._prepare_model(model_id)
        metrics = {}
        
        # Track expert usage
        total_experts_used = 0
        total_tokens = 0
        inference_times = []
        memory_usage = []
        
        for i in tqdm(range(0, len(self.dataset), self.batch_size),
                     desc=f"Computing efficiency metrics for {model_id}"):
            batch = self.dataset[i:i + self.batch_size]
            texts = batch['text']
            
            try:
                # Measure memory before inference
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.memory_allocated()
                else:
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss
                
                # Time inference
                start_time = time.time()
                
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**encoded)
                    
                    # Get expert usage if available
                    expert_usage = self._get_expert_usage(outputs)
                    if expert_usage:
                        total_experts_used += expert_usage['active_experts'].sum().item()
                        total_tokens += expert_usage['active_experts'].numel()
                
                # Measure memory after inference
                if torch.cuda.is_available():
                    memory_after = torch.cuda.max_memory_allocated()
                else:
                    memory_after = process.memory_info().rss
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                memory_usage.append(memory_after - memory_before)
                
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {str(e)}")
                continue
        
        # Calculate metrics
        if total_tokens > 0:
            metrics['avg_experts_per_token'] = total_experts_used / total_tokens
        
        if inference_times:
            metrics['avg_inference_time'] = np.mean(inference_times)
            metrics['std_inference_time'] = np.std(inference_times)
        
        if memory_usage:
            metrics['avg_memory_usage'] = np.mean(memory_usage)
            metrics['max_memory_usage'] = np.max(memory_usage)
        
        return metrics

    def compute_adaptive_behavior_metrics(self, model_id: str) -> Dict[str, float]:
        """Compute metrics related to model's adaptive behavior."""
        tokenizer, model = self._prepare_model(model_id)
        metrics = {}
        
        # Track expert usage and input difficulty
        expert_usage_per_sample = []
        input_difficulties = []
        
        for i in tqdm(range(0, len(self.dataset), self.batch_size),
                     desc=f"Computing adaptive behavior metrics for {model_id}"):
            batch = self.dataset[i:i + self.batch_size]
            texts = batch['text']
            
            try:
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**encoded)
                    
                    # Get expert usage
                    expert_usage = self._get_expert_usage(outputs)
                    if expert_usage:
                        expert_usage_per_sample.extend(
                            expert_usage['active_experts'].mean(dim=1).cpu().numpy()
                        )
                    
                    # Estimate input difficulty (e.g., using entropy of predictions)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    input_difficulties.extend(entropy.mean(dim=1).cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {str(e)}")
                continue
        
        # Calculate correlation between input difficulty and expert usage
        if expert_usage_per_sample and input_difficulties:
            correlation = stats.pearsonr(input_difficulties, expert_usage_per_sample)[0]
            metrics['difficulty_expert_correlation'] = correlation
        
        # Calculate expert usage distribution metrics
        if expert_usage_per_sample:
            expert_usage = np.array(expert_usage_per_sample)
            metrics['expert_usage_mean'] = expert_usage.mean()
            metrics['expert_usage_std'] = expert_usage.std()
            metrics['expert_usage_entropy'] = stats.entropy(np.histogram(expert_usage, bins=20)[0])
        
        return metrics

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Run all evaluations for all models."""
        for model_name, model_id in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Compute accuracy metrics
            try:
                accuracy_metrics = self.compute_accuracy_metrics(model_id)
                self.results[model_name].update(accuracy_metrics)
            except Exception as e:
                logger.error(f"Error computing accuracy metrics for {model_name}: {str(e)}")
            
            # Compute efficiency metrics
            try:
                efficiency_metrics = self.compute_efficiency_metrics(model_id)
                self.results[model_name].update(efficiency_metrics)
            except Exception as e:
                logger.error(f"Error computing efficiency metrics for {model_name}: {str(e)}")
            
            # Compute adaptive behavior metrics
            try:
                adaptive_metrics = self.compute_adaptive_behavior_metrics(model_id)
                self.results[model_name].update(adaptive_metrics)
            except Exception as e:
                logger.error(f"Error computing adaptive behavior metrics for {model_name}: {str(e)}")
        
        return self.results

    def plot_results(self, save_path: Optional[str] = None):
        """Plot evaluation results."""
        if not self.results:
            logger.warning("No results to plot. Run evaluate_all() first.")
            return
        
        # Group metrics by category
        metric_categories = {
            'Accuracy & Performance': ['token_accuracy', 'perplexity'],
            'Efficiency': ['avg_experts_per_token', 'avg_inference_time', 'avg_memory_usage'],
            'Adaptive Behavior': ['difficulty_expert_correlation', 'expert_usage_entropy']
        }
        
        # Create subplots for each category
        fig, axes = plt.subplots(1, len(metric_categories), figsize=(5*len(metric_categories), 5))
        if len(metric_categories) == 1:
            axes = [axes]
        
        for i, (category, metrics) in enumerate(metric_categories.items()):
            # Filter metrics that exist in results
            available_metrics = [m for m in metrics if any(m in model_results for model_results in self.results.values())]
            
            if not available_metrics:
                continue
                
            # Create subplots for each metric in the category
            subfig, subaxes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 5))
            if len(available_metrics) == 1:
                subaxes = [subaxes]
            
            for j, metric in enumerate(available_metrics):
                values = [self.results[model].get(metric, 0) for model in self.models]
                sns.barplot(x=list(self.models.keys()), y=values, ax=subaxes[j])
                subaxes[j].set_title(metric)
                subaxes[j].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_{category.lower().replace(' ', '_')}.png")
            plt.show()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def print_results(self):
        """Print evaluation results in a formatted table."""
        if not self.results:
            logger.warning("No results to print. Run evaluate_all() first.")
            return
        
        # Group metrics by category
        metric_categories = {
            'Accuracy & Performance': ['token_accuracy', 'perplexity'],
            'Efficiency': ['avg_experts_per_token', 'avg_inference_time', 'avg_memory_usage'],
            'Adaptive Behavior': ['difficulty_expert_correlation', 'expert_usage_entropy']
        }
        
        for category, metrics in metric_categories.items():
            print(f"\n{category}")
            print("=" * 50)
            
            # Filter metrics that exist in results
            available_metrics = [m for m in metrics if any(m in model_results for model_results in self.results.values())]
            
            if not available_metrics:
                continue
            
            # Print header
            header = "Model".ljust(20) + " | " + " | ".join(metric.ljust(15) for metric in available_metrics)
            print(header)
            print("-" * len(header))
            
            # Print results for each model
            for model_name in self.models:
                row = model_name.ljust(20) + " | "
                row += " | ".join(f"{self.results[model_name].get(metric, 'N/A'):.4f}".ljust(15) 
                                for metric in available_metrics)
                print(row)
            
            print("=" * 50)
