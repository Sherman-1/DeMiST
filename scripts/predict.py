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

# --- Constants ---
MAX_LEN_AA = 100 # Consistent with training

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
    
    # Sort by the standard class order if possible, or return dict
    return thresholds

def read_fasta(path):
    records = []
    print(f"Reading FASTA: {path}")
    for record in SeqIO.parse(str(path), "fasta"):
        seq = str(record.seq).upper().replace(" ", "").replace("\n", "")
        if 20 <= len(seq) <= MAX_LEN_AA:
             # Add spaces for ProtT5
            records.append({
                "id": record.id,
                "sequence": seq,
                "t5_input": " ".join(list(seq))
            })
    print(f"Loaded {len(records)} valid sequences (20-100 AA).")
    return records

def predict(args):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Config & Resources
    # We need the id2label mapping to map logits to class names
    id2label_path = Path(args.model_dir).parent.parent / "data/processed/id2label.json"
    if not id2label_path.exists():
        # Fallback: try to find it in the model directory or assume standard order
        print("Warning: id2label.json not found in data/processed. Using default class order.")
        # Make sure these match your training EXACTLY
        classes = ["globular", "molten", "transmembrane", "disordered", "shuffled_globular", "shuffled_molten"]
        id2label = {i: c for i, c in enumerate(classes)}
    else:
        import json
        with open(id2label_path, 'r') as f:
            id2label = {int(k): v for v, k in json.load(f).items()} # Inverted or direct? Check json structure.
            # Usually id2label is "0": "globular". 
            # If your file is label2id, invert it.
            # Assuming standard id2label JSON format {0: "label", ...}
            
    num_classes = len(id2label)
    label_list = [id2label[i] for i in range(num_classes)]
    
    # 3. Load Model
    # We construct a dummy config dict to satisfy ProtT5Classifier init
    # The actual weights will be overwritten by load_state_dict/peft
    config_mock = {
        "model": {
            "checkpoint": "Rostlab/prot_t5_xl_half_uniref50-enc",
            "use_lora": (Path(args.model_dir) / "adapter_config.json").exists()
        }
    }
    
    print("Initializing Model...")
    model = ProtT5Classifier(config_mock, num_classes=num_classes)
    
    # Load Weights
    # If LoRA was used, the model directory contains 'adapter_model.bin' and we use PEFT
    if config_mock["model"]["use_lora"]:
        from peft import PeftModel
        print("Loading LoRA adapters...")
        # Base model is already loaded in __init__, we just wrap it
        # Note: ProtT5Classifier wraps the encoder. We need to be careful with PEFT wrapping.
        # Actually, simpler way for inference:
        # The Trainer saved the whole thing or just adapters?
        # If standard HF Trainer + LoRA: it saves adapters.
        # If you called model.save_pretrained(), it saves adapters.
        
        # Load the base model weights if they were fine-tuned (unlikely with LoRA)
        # OR just load adapters onto the encoder.
        model.encoder = PeftModel.from_pretrained(model.encoder, args.model_dir)
        
        # Load the classifier head (Trainer saves this separately usually if it's not part of LoRA modules)
        # Check if pytorch_model.bin exists (custom head weights)
        head_path = Path(args.model_dir) / "pytorch_model.bin"
        if head_path.exists():
            state_dict = torch.load(head_path, map_location="cpu")
            # Filter for classifier keys only if needed, or load strict=False
            model.load_state_dict(state_dict, strict=False)
            
    else:
        # Standard full finetuning loading
        weights_path = Path(args.model_dir) / "pytorch_model.bin"
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config_mock["model"]["checkpoint"], do_lower_case=False)

    # 5. Load Thresholds
    thresholds_map = load_thresholds(args.thresholds)
    thresholds_arr = np.array([thresholds_map.get(c, 0.5) for c in label_list])
    print(f"Loaded Thresholds: {thresholds_map}")

    # 6. Inference Loop
    sequences = read_fasta(args.input)
    results = []
    
    batch_size = args.batch_size
    
    print(f"Predicting on {len(sequences)} sequences...")
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i:i+batch_size]
            inputs = [b["t5_input"] for b in batch]
            
            # Tokenize
            encodings = tokenizer(
                inputs, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LEN_AA, 
                return_tensors="pt"
            ).to(device)
            
            # Forward
            logits = model(encodings.input_ids, encodings.attention_mask)["logits"]
            probs = torch.sigmoid(logits).cpu().numpy() # BCE -> Sigmoid
            
            # Decode
            for j, p in enumerate(probs):
                # Apply thresholds
                passed_indices = np.where(p >= thresholds_arr)[0]
                
                decision = "Unclassified"
                if len(passed_indices) == 1:
                    decision = label_list[passed_indices[0]]
                elif len(passed_indices) > 1:
                    # Ambiguous: pick max relative to threshold or raw max
                    # Simple approach: Max raw probability
                    decision = label_list[np.argmax(p)]
                
                res = {
                    "id": batch[j]["id"],
                    "sequence": batch[j]["sequence"],
                    "prediction": decision
                }
                # Add individual probs
                for k, cls_name in enumerate(label_list):
                    res[cls_name] = float(p[k])
                
                results.append(res)

    # 7. Save
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