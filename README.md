# DeMiST: Deep Learning for Microprotein Sequence Transformers

<p align="center">
  <img src="assets/logo.png" alt="DeMiST Logo" width="300"/> 
</p>

**DeMiST** stands for **D**eciphering **Mi**croprotein **S**tructural **T**ransitions.

### Scope and Technical Stack
This repository contains the complete pipeline for sequence-based microprotein classification:

## 🚀 Quick Start (Development)

1.  **Setup Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate demist
    pip install -e . # Install the project locally
    ```

2.  **Prepare Data:**
    * Place raw FASTA files in `data/raw/` as specified in `assets/config.yaml`.
    * Run the dataset builder (which performs tokenization and train/test/val splitting):
    ```bash
    python src/dataset.py # Runs the __main__ function to generate Parquet files
    ```

3.  **Train Model:**
    ```bash
    python scripts/train.py --config configs/default.yaml
    ```