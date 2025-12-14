# DeMiST: Deciphering Microprotein Structural Transitions

<p align="center">
  <img src="assets/logo.png" alt="DeMiST Logo" width="300"/> 
</p>

### Scope and Technical Stack
This repository contains the complete pipeline for sequence-based microprotein classification:

## 🚀 Quick Start (Development)

1.  **Setup Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate demist
    pip install -e . 
    ```

2.  **Prepare Data:**
    * Place raw FASTA files in `data/raw/` as specified in `assets/config.yaml`.
    * Run the dataset builder (which performs tokenization and train/test/val splitting):
    ```bash
    python src/dataset.py 
    ```

3.  **Train Model:**
    ```bash
    python scripts/train.py --config configs/default.yaml
    ```