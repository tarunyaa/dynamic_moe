import os
import time
import random
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, IterableDataset
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
        test_size=0.05,
        seed=42,
        data_dir="./data"
    ):
        """
        Initialize the data loader.
        
        Args:
            model_name: Name of the model for tokenization
            max_length: Maximum sequence length for tokenization
            test_size: Proportion of data to use for validation
            seed: Random seed for reproducible splits
            data_dir: Directory to store data
        """
        self.model_name = model_name
        self.max_length = max_length
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
        Load and prepare SlimPajama dataset in streaming mode.
        
        Returns:
            Tuple of (train_dataset, eval_dataset) as streaming datasets
        """
        print("Loading SlimPajama dataset in streaming mode...")
        
        # Handle rate limiting with retries
        max_retries = 5
        for attempt in range(max_retries):
            try:
                dataset = load_dataset(
                    "cerebras/SlimPajama-627B", 
                    split="train", 
                    streaming=True
                )
                break
            except HfHubHTTPError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    retry_time = 2 ** attempt * (1 + random.random())
                    print(f"\nRate limit hit. Retrying in {retry_time:.1f} seconds. Retries left: {max_retries - attempt - 1}")
                    time.sleep(retry_time)
                else:
                    raise
        
        # For streaming datasets, we can use the IterableDatasets.shuffle method with a buffer_size
        shuffled_dataset = dataset.shuffle(
            buffer_size=10000,
            seed=self.seed
        )
        
        # Create train/validation split for streaming data
        # Since we can't easily split a streaming dataset with exact proportions,
        # we'll use a deterministic function based on a hash of the examples
        def train_validation_split(example):
            import hashlib
            # Create a deterministic hash based on the content and our seed
            text_hash = hashlib.md5((str(example['text']) + str(self.seed)).encode()).hexdigest()
            # Convert first 8 chars of hash to int and decide split
            hash_int = int(text_hash[:8], 16)
            # If hash_int % 100 is less than test_size*100, it goes to validation
            return 'validation' if hash_int % 100 < (self.test_size * 100) else 'train'
        
        # Split the dataset
        train_dataset = shuffled_dataset.filter(lambda example: train_validation_split(example) == 'train')
        eval_dataset = shuffled_dataset.filter(lambda example: train_validation_split(example) == 'validation')
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
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
        
        print("Tokenizing datasets in streaming mode...")
        tokenized_train = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=16,
            remove_columns=["text"],
        )
        
        tokenized_eval = self.eval_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=16,
            remove_columns=["text"],
        )
        
        print("Datasets prepared in streaming mode")
        
        self.train_dataset = tokenized_train
        self.eval_dataset = tokenized_eval
        
        return tokenized_train, tokenized_eval
    
    def get_datasets(self):
        """
        Get the prepared datasets.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.train_dataset is None:
            print("Datasets not prepared yet. Preparing now...")
            self.prepare_datasets()
            
        return self.train_dataset, self.eval_dataset

# Run data preparation when script is executed directly
if __name__ == "__main__":
    loader = DataLoader(
        model_name="gpt2-xl",
        max_length=512,
        test_size=0.05,
    )
    
    train_dataset, eval_dataset = loader.prepare_datasets()
    print("Data preparation complete!")
