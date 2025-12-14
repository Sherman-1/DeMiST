import os
import yaml
import json
import pandas as pd
import numpy as np
import random
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
    Reads FASTAs, downsamples per class, tokenizes, generates ONE-HOT labels, 
    and saves Parquet files.
    """
    config = load_config()["dataset"]
    output_dir = config["output_dir"]
    raw_files_map = config["raw_files"]
    model_checkpoint = config["tokenizer_model"]
    max_len = config["max_length"]
    
    max_samples = config.get("max_samples_per_class", None) 

    print(f"Loading Tokenizer: {model_checkpoint}...")
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, do_lower_case=False)

    all_records = []
    
    label_list = sorted(list(raw_files_map.keys()))
    label2id = {label: i for i, label in enumerate(label_list)}
    num_classes = len(label_list)

    print(f"Processing {num_classes} classes: {label2id}")
    if max_samples:
        print(f"Downsampling active: Max {max_samples} sequences per class.")

    for str_label, filepath in raw_files_map.items():
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Skipping.")
            continue
            
        class_id = label2id[str_label]
        one_hot = np.zeros(num_classes, dtype=float)
        one_hot[class_id] = 1.0
        one_hot_list = one_hot.tolist()

        valid_records = []
        print(f"Parsing {str_label}...")
        
        for record in SeqIO.parse(filepath, "fasta"):
            clean_seq = clean_sequence(record.seq)
            if clean_seq:  
                prot_t5_seq = " ".join(list(clean_seq))
                valid_records.append({
                    "id": record.id,
                    "raw": str(record.seq),
                    "clean": clean_seq,
                    "t5_input": prot_t5_seq
                })

        count_before = len(valid_records)
        if max_samples is not None and count_before > max_samples:
            random.seed(66370)
            valid_records = random.sample(valid_records, max_samples)

        for item in valid_records:
            tokenized = tokenizer(
                item["t5_input"],
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

            all_records.append({
                "seq_id": item["id"],
                "str_label": str_label,
                "class_id_int": class_id, 
                "labels": one_hot_list,
                "raw_sequence": item["raw"],
                "clean_sequence": item["clean"],
                "input_ids": tokenized["input_ids"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist()
            })

    df = pd.DataFrame(all_records)
    print(f"Total records processed: {len(df)}")

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

    config = load_config()["dataset"]
    data_dir = config["output_dir"]
    
    data_files = {
        "train": os.path.join(data_dir, "train.parquet"),
        "validation": os.path.join(data_dir, "validation.parquet"),
        "test": os.path.join(data_dir, "test.parquet")
    }

    dataset = load_dataset("parquet", data_files=data_files)

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in columns_to_keep])

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    def cast_labels(batch):
        batch["labels"] = batch["labels"].float()
        return batch

    dataset = dataset.map(lambda x: {"labels": x["labels"].to(torch.float32)}, batched=True, batch_size=None)

    with open(os.path.join(data_dir, "id2label.json"), "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
        
    return dataset, id2label

if __name__ == "__main__":
    build_dataset()