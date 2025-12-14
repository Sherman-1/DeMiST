#!/usr/bin/env python3
"""
Prediction script for DeMiST.
"""

import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import scipy.special
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
from transformers import T5Tokenizer


current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from model import ProtT5Classifier
from utils import set_seeds, print_gpu_memory

MAX_LEN_AA = 100 

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_thresholds(thresholds_path):
    """Parses the generated thresholds file."""
    thresholds = {}
    with open(thresholds_path, 'r') as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":")
                thresholds[key.strip()] = float(val.strip())
    
    return thresholds

def read_fasta(path):
    records = []
    print(f"Reading FASTA: {path}")
    for record in SeqIO.parse(str(path), "fasta"):
        seq = str(record.seq).upper().replace(" ", "").replace("\n", "")
        if 20 <= len(seq) <= MAX_LEN_AA:
            records.append({
                "id": record.id,
                "sequence": seq,
                "t5_input": " ".join(list(seq))
            })
    print(f"Loaded {len(records)} valid sequences (20-100 AA).")
    return records

def predict(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    id2label_path = Path(args.model_dir).parent.parent / "data/processed/id2label.json"
    if not id2label_path.exists():
 
        print("Warning: id2label.json not found in data/processed. Using default class order.")

        classes = ["globular", "molten", "transmembrane", "disordered", "shuffled_globular", "shuffled_molten"]
        id2label = {i: c for i, c in enumerate(classes)}
    else:
        import json
        with open(id2label_path, 'r') as f:
            id2label = {int(k): v for v, k in json.load(f).items()} 

            
    num_classes = len(id2label)
    label_list = [id2label[i] for i in range(num_classes)]
    
    
    config_mock = {
        "model": {
            "checkpoint": "Rostlab/prot_t5_xl_half_uniref50-enc",
            "use_lora": (Path(args.model_dir) / "adapter_config.json").exists()
        }
    }
    
    print("Initializing Model...")
    model = ProtT5Classifier(config_mock, num_classes=num_classes)
    

    if config_mock["model"]["use_lora"]:
        from peft import PeftModel
        print("Loading LoRA adapters...")
        model.encoder = PeftModel.from_pretrained(model.encoder, args.model_dir)
        
        head_path = Path(args.model_dir) / "pytorch_model.bin"
        if head_path.exists():
            state_dict = torch.load(head_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            
    else:

        weights_path = Path(args.model_dir) / "pytorch_model.bin"
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()


    tokenizer = T5Tokenizer.from_pretrained(config_mock["model"]["checkpoint"], do_lower_case=False)

    thresholds_map = load_thresholds(args.thresholds)
    thresholds_arr = np.array([thresholds_map.get(c, 0.5) for c in label_list])
    print(f"Loaded Thresholds: {thresholds_map}")

    sequences = read_fasta(args.input)
    results = []
    
    batch_size = args.batch_size
    
    print(f"Predicting on {len(sequences)} sequences...")
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i:i+batch_size]
            inputs = [b["t5_input"] for b in batch]
            
            encodings = tokenizer(
                inputs, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LEN_AA, 
                return_tensors="pt"
            ).to(device)

            logits = model(encodings.input_ids, encodings.attention_mask)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy() 
            
            for j, p in enumerate(probs):
                passed_indices = np.where(p >= thresholds_arr)[0]
                
                decision = "Unclassified"
                if len(passed_indices) == 1:
                    decision = label_list[passed_indices[0]]
                elif len(passed_indices) > 1:
                    decision = label_list[np.argmax(p)]
                
                res = {
                    "id": batch[j]["id"],
                    "sequence": batch[j]["sequence"],
                    "prediction": decision
                }
                for k, cls_name in enumerate(label_list):
                    res[cls_name] = float(p[k])
                
                results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input FASTA")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--model_dir", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--thresholds", required=True, help="Path to best_thresholds.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    predict(args)