import os
import time
import random
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
from huggingface_hub.utils import HfHubHTTPError


class DataLoader:
    """
    Data loader for language model training that handles downloading, processing,
    and tokenizing data from the SlimPajama dataset.
    """
    
    def __init__(
        self,
        model_name="gpt2-xl",
        max_length=512,
        subset_size=1_000_000,
        test_size=0.05,
        seed=42,
        data_dir="./data"
    ):
        """
        Initialize the data loader.
        
        Args:
            model_name: Name of the model for tokenization
            max_length: Maximum sequence length for tokenization
            subset_size: Number of samples to collect from SlimPajama
            test_size: Proportion of data to use for validation
            seed: Random seed for reproducible splits
            data_dir: Directory to store data
        """
        self.model_name = model_name
        self.max_length = max_length
        self.subset_size = subset_size
        self.test_size = test_size
        self.seed = seed
        self.data_dir = data_dir
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_tokenizer(self):
        """Load tokenizer for the specified model."""
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token for GPT models
        return self.tokenizer
    
    def get_slimpajama_data(self):
        """
        Load and prepare SlimPajama dataset.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        print("Loading SlimPajama subset from Hugging Face...")
        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True
        )

        samples = []
        print(f"Collecting {self.subset_size} samples...")
        
        max_retries = 5
        i = 0
        pbar = tqdm(total=self.subset_size, desc="Streaming dataset")
        
        while i < self.subset_size:
            try:
                sample_iter = iter(dataset)
                while i < self.subset_size:
                    sample = next(sample_iter)
                    samples.append(sample)
                    i += 1
                    pbar.update(1)
            except HfHubHTTPError as e:
                if "429" in str(e) and max_retries > 0:
                    retry_time = 2 ** (5 - max_retries) * (1 + random.random())
                    max_retries -= 1
                    print(f"\nRate limit hit. Retrying in {retry_time:.1f} seconds. Retries left: {max_retries}")
                    time.sleep(retry_time)
                else:
                    pbar.close()
                    raise
            except Exception as e:
                pbar.close()
                raise
        
        pbar.close()

        print("Converting to Hugging Face Dataset...")
        collected_dataset = Dataset.from_dict({
            k: [sample[k] for sample in samples]
            for k in samples[0].keys()
        })

        print("Splitting into train and validation...")
        splits = collected_dataset.train_test_split(test_size=self.test_size, seed=self.seed)
        self.train_dataset = splits['train']
        self.eval_dataset = splits['test']
        return self.train_dataset, self.eval_dataset

    def preprocess_function(self, examples):
        """
        Tokenize text examples for model training.
        
        Args:
            examples: Batch of examples to tokenize
            
        Returns:
            Tokenized examples with input_ids and labels
        """
        texts = examples["text"]
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    def prepare_datasets(self):
        """
        Prepare and tokenize datasets for training.
        
        Returns:
            Tuple of (tokenized_train_dataset, tokenized_eval_dataset)
        """
        if self.tokenizer is None:
            self.load_tokenizer()
            
        if self.train_dataset is None or self.eval_dataset is None:
            self.get_slimpajama_data()
        
        print("Tokenizing datasets...")
        tokenized_train = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=16,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        tokenized_eval = self.eval_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=16,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing validation data"
        )
        
        print(f"Prepared {len(tokenized_train)} training examples and {len(tokenized_eval)} validation examples")
        
        self.train_dataset = tokenized_train
        self.eval_dataset = tokenized_eval
        
        return tokenized_train, tokenized_eval
    
    def get_datasets(self):
        """
        Get the prepared datasets.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.train_dataset is None or "input_ids" not in self.train_dataset.features:
            print("Datasets not prepared yet. Preparing now...")
            self.prepare_datasets()
            
        return self.train_dataset, self.eval_dataset

# Run data preparation when script is executed directly
if __name__ == "__main__":
    loader = DataLoader(
        model_name="gpt2-xl",
        max_length=512,
        subset_size=1_000_000,
        test_size=0.05,
    )
    
    train_dataset, eval_dataset = loader.prepare_datasets()
    print("Data preparation complete!")
