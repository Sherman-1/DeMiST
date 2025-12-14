import os
import yaml
import json
import pandas as pd
import numpy as np
from Bio import SeqIO
from transformers import T5Tokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch

def load_config(config_path="assets/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

AAs = set("ACDEFGHIKLMNPQRSTVWY")
MAX_AA_LENGTH = 100

def clean_sequence(seq):
    """
    Filters out : 
        - Non standard amino acids sequences 
        - Longer than max length sequences

    >>> clean_sequence("ACDEFGHIKLMNPQRSTVWY")  # Valid
    'ACDEFGHIKLMNPQRSTVWY'
    >>> clean_sequence("ACDXYZ")  # Invalid characters
    ''
    >>> clean_sequence("A" * 101)  # Too long
    ''
    """
    if not isinstance(seq, str):
        seq = str(seq)
    seq = seq.replace(" ", "").replace("\n", "").upper()
    if len(seq) > MAX_AA_LENGTH:
        return ""
    if any(aa not in AAs for aa in seq):
        return ""
    return seq


def build_dataset():
    """
    Reads FASTAs, tokenizes, generates ONE-HOT labels for BCE, 
    and saves Parquet files.
    """
    config = load_config()["dataset"]
    output_dir = config["output_dir"]
    raw_files_map = config["raw_files"]
    model_checkpoint = config["tokenizer_model"]
    max_len = config["max_length"]

    print(f"Loading Tokenizer: {model_checkpoint}...")
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, do_lower_case=False)

    all_records = []
    
    label_list = sorted(list(raw_files_map.keys()))
    label2id = {label: i for i, label in enumerate(label_list)}
    num_classes = len(label_list)

    print(f"Processing {num_classes} classes: {label2id}")

    for str_label, filepath in raw_files_map.items():
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Skipping.")
            continue
            
        class_id = label2id[str_label]
        one_hot = np.zeros(num_classes, dtype=float)
        one_hot[class_id] = 1.0
        one_hot_list = one_hot.tolist()

        for record in SeqIO.parse(filepath, "fasta"):
            clean_seq = clean_sequence(record.seq)
            
            # ProtT5 Tokenization
            tokenized = tokenizer(
                clean_seq,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

            all_records.append({
                "seq_id": record.id,
                "str_label": str_label,
                "class_id_int": class_id, 
                "labels": one_hot_list,
                "raw_sequence": str(record.seq),
                "clean_sequence": clean_seq,
                "input_ids": tokenized["input_ids"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist()
            })

    df = pd.DataFrame(all_records)
    print(f"Total records: {len(df)}")

    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["class_id_int"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["class_id_int"], random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "validation.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    id2label = {i: l for l, i in label2id.items()}
    with open(os.path.join(output_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f)

    print(f"Dataset saved to {output_dir}")

def load_processed_data():
    """
    Loads Parquet files and formats 'labels' as FloatTensor for BCE.
    """
    config = load_config()["dataset"]
    data_dir = config["output_dir"]
    
    data_files = {
        "train": os.path.join(data_dir, "train.parquet"),
        "validation": os.path.join(data_dir, "validation.parquet"),
        "test": os.path.join(data_dir, "test.parquet")
    }

    # Load
    dataset = load_dataset("parquet", data_files=data_files)

    # Columns strictly needed for Model/Trainer
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in columns_to_keep])

    # CRITICAL: Set format to PyTorch
    # labels must be 'float' (float32) for BCEWithLogitsLoss
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Explicit casting to ensure FloatTensor (Dataset sometimes defaults lists to doubles/float64)
    # We map over the dataset to cast, or rely on the model to cast, 
    # but strictly speaking set_format usually handles this if the data is float compatible.
    # To be extremely safe for BCE:
    def cast_labels(batch):
        batch["labels"] = batch["labels"].float()
        return batch

    dataset = dataset.map(lambda x: {"labels": x["labels"].to(torch.float32)}, batched=True, batch_size=None)

    # Load label info
    with open(os.path.join(data_dir, "id2label.json"), "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
        
    return dataset, id2label

if __name__ == "__main__":
    build_dataset()