import torch 
import numpy as np 
import random 
from transformers import set_seed
import os

def set_seeds(seed: int = 66370) -> None:
    """
    Fix every possible seed for reproducibility 
    """

    os.environ["PYTHONHASHSEED"] = str(seed)   # deterministic hash
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"    All seeds set to {seed}.")

def print_gpu_memory():
    """Prints amount of GPU memory currently allocated."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated | {reserved:.2f} GB reserved")

def clear_cuda_cache():
    """Releases GPU memory."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def calculate_pos_weights(dataset, num_classes):
    """
    neg_count / pos_count
    """
    
    print("Calculating class weights from training data...")
    all_labels = np.array(dataset["labels"]) # Shape: (N, num_classes)

    pos_counts = all_labels.sum(axis=0)
    total_counts = len(all_labels)
    neg_counts = total_counts - pos_counts
    
    pos_counts = np.maximum(pos_counts, 1.0)
    
    pos_weights = neg_counts / pos_counts
    return torch.FloatTensor(pos_weights)