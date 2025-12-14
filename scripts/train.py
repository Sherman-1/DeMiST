import argparse
import torch
import wandb
import yaml
import sys, os 
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, T5Tokenizer
from dataset import load_processed_data
from model import ProtT5Classifier
from metrics import compute_metrics_bce, find_best_thresholds
from utils import set_seeds, calculate_pos_weights, print_gpu_memory

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="assets/config.yaml", help="Path to config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    hyperparams = config.get("hyperparameters", {})
    
    set_seeds(hyperparams.get("seed", 66370))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cpu"):
        raise EnvironmentError("CUDA device not found. A GPU is required to run this training script.")
    
    print_gpu_memory()

    print("Loading datasets...")
    dataset, id2label = load_processed_data()
    num_classes = len(id2label)
    
    pos_weights = calculate_pos_weights(dataset["train"], num_classes).to(device)

    model = ProtT5Classifier(
        config=config,          
        num_classes=num_classes,
        loss_weights=pos_weights
    ).to(device)
    
    tokenizer_model = config["model"]["checkpoint"]
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_model, do_lower_case=False)
    
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=hyperparams.get("epochs", 4),
        per_device_train_batch_size=hyperparams.get("batch_size", 32),
        learning_rate=float(hyperparams.get("lr", 1e-4)),
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",          
        eval_strategy="epoch",          
        load_best_model_at_end=True,    
        metric_for_best_model="roc_auc",
        greater_is_better=True,         
        save_total_limit=1,            
        report_to="wandb",
        run_name="demist_lora_run",
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_bce
    )

    wandb.init(project="DeMiST")
    trainer.train()
    
    print("\nOptimizing decision thresholds on Validation set...")
    
    val_preds = trainer.predict(dataset["validation"])
    val_logits = val_preds.predictions

    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]  
        
    val_labels = val_preds.label_ids
    val_probs = scipy.special.expit(val_logits)

    best_thresholds = find_best_thresholds(val_probs, val_labels)
    thresholds_path = os.path.join(training_args.output_dir, "best_thresholds.txt")
    
    print("\nOptimal Thresholds per Class:")
    with open(thresholds_path, "w") as f:
        for class_idx, threshold in enumerate(best_thresholds):
            class_name = id2label[class_idx] if 'id2label' in locals() else f"Class_{class_idx}"
            log_line = f"{class_name}: {threshold:.4f}"
            print(f"  {log_line}")
            f.write(log_line + "\n")

    print(f"Thresholds saved to {thresholds_path}")

    print("\nGenerating Confusion Matrix on TEST Set...")

    test_preds = trainer.predict(dataset["test"])
    test_logits = test_preds.predictions
    if isinstance(test_logits, tuple): test_logits = test_logits[0]
    test_probs = scipy.special.expit(test_logits)
    test_labels_idx = np.argmax(test_preds.label_ids, axis=1) 

    final_preds = []
    for i in range(len(test_probs)):
        probs = test_probs[i]
        
        passed_indices = [c for c, p in enumerate(probs) if p >= best_thresholds[c]]
        
        if len(passed_indices) == 0:

            final_preds.append(np.argmax(probs)) 
        elif len(passed_indices) == 1:

            final_preds.append(passed_indices[0])
        else:

            final_preds.append(np.argmax(probs))

    class_names = [id2label[i] for i in range(len(id2label))]
    cm = confusion_matrix(test_labels_idx, final_preds, labels=range(len(class_names)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Thresholded)\nModel: {config["model"]["checkpoint"]}')

    save_path = os.path.join(training_args.output_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")

    wandb.finish()

if __name__ == "__main__":
    main()