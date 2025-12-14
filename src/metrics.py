import numpy as np
import scipy.special
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

def compute_metrics_bce(eval_pred):
    """
    Metrics function for Hugging Face Trainer (Multi-label/BCE context).
    Uses default 0.5 threshold for immediate logging.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple): 
        logits = logits[0]
        
    probs = scipy.special.expit(logits)
    preds = (probs >= 0.5).astype(int)
    
    preds_sl = np.argmax(probs, axis=-1)
    true_sl  = np.argmax(labels, axis=-1)

    return {
        "accuracy": accuracy_score(true_sl, preds_sl),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probs, average="macro", multi_class="ovr")
    }

def find_best_thresholds(probs: np.ndarray, labels: np.ndarray, n_bins: int = 101):
    """
    Optimizes decision thresholds per class based on F1 score.
    """
    thresholds = np.linspace(0, 1, n_bins)
    best_thresholds = []
    
    num_classes = probs.shape[1]
    
    for c in range(num_classes):
        best_f1 = 0.0
        best_t = 0.5

        y_prob = probs[:, c]
        y_true = labels[:, c]
        
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds.append(best_t)
        
    return best_thresholds