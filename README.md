# SDBformer

PyTorch implementation of **SDBformer** (Spiking Dual-Branch Transformer) for EEG-Based Auditory Attention Detection.

## Introduction

We propose **SDBformer**, a Spiking Dual-Branch Transformer designed for high-fidelity wearable BCI applications. It addresses the challenges of high energy consumption and signal distortion in traditional methods.

The network features:

- **Dual-Branch Architecture**: Effectively extracts both spectral and spatial features from EEG signals using parallel spiking streams.
    
- **Spike-Driven Self-Attention (SDSA)**: A novel attention mechanism that operates entirely in the spike domain, avoiding energy-intensive multiplication operations.
    
- **Spike Firing Approximation (SFA)**: A training paradigm that decouples training and inference, eliminating the need for expensive Backpropagation Through Time (BPTT).
    

Experimental results on the KUL and DTU benchmark datasets demonstrate that SDBformer achieves state-of-the-art (SOTA) accuracy. Notably, it operates with extremely low energy consumption (**~0.0206 mJ/sample**) and requires only **0.13 M** parameters, making it highly suitable for resource-constrained wearable devices.

## Dataset & Preprocess

This code is designed for **Within-Subject (WS)** analysis using **Differential Entropy (DE)** features.

1. **Download Dataset**: Please acquire the [DTU dataset](https://www.google.com/search?q=https://zenodo.org/record/1199011&authuser=2) (or your target AAD dataset).
    
2. **Feature Extraction**:
    
    - The code expects pre-processed `.npz` files containing DE features.
        
    - Expected file naming format: `S{subject_id}_DE_Features_1s.npz`.
        
    - Data shape within `.npz`: `DE` key with shape `(N_samples, 66, 5)` (Channels, Bands).
        
3. Directory Structure:
    
    Place your data files in the data directory defined in the config (default: ./data/DE_Features).
    

## Requirements

- Python 3.12+
    

To install the dependencies, run:

Bash

```
pip install -r requirements.txt
```

**`requirements.txt` content:**

Plaintext

```
matplotlib==3.10.8
mne==1.11.0
numpy==2.4.1
pandas==2.3.3
scikit_learn==1.8.0
torch==2.9.1
tqdm==4.66.5
```

_Note: The versions listed above match the development environment. Standard stable versions (e.g., `torch>=2.0`, `numpy>=1.24`) should also work compatible._

## Run

1. **Configuration**:
    
    - Modify the hyperparameters and file paths in the `TrainingConfig` class within `main.py`.
        
    - You can select specific subjects to run by modifying `self.subjects` in the config.
        
2. Training & Evaluation:
    
    Run the main script to perform 5-Fold Cross-Validation, CSP preprocessing, and Energy Analysis.
    
    Bash
    
    ```
    python main.py
    ```
    
3. **Output**:
    
    - Training logs and model checkpoints will be saved in `./Results/`.
        
    - A summary CSV report (`final_performance_summary.csv`) containing Mean/Std accuracy for all subjects will be generated.
        
    - Detailed energy consumption reports (FLOPs/SOPs/Energy) for the representative model (e.g., S1-F1) will be saved in the respective folder.
        