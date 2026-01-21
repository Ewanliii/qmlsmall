# Source Code (`src/`)

This directory contains the reproducible implementation used in the paper.

## Structure

- `preprocessing/`  
  Scripts for data preprocessing and representation learning (e.g., DAE training and latent feature extraction).

- `classical_ml/`  
  Baseline classical machine learning experiments (Logistic Regression, Random Forest, MLP).

- `qml/`  
  Quantum machine learning experiments (VQC variants and evaluation).

## How to run

Each submodule provides its own `README.md` with example commands. In general, the workflow is:

1. Run preprocessing (DAE) to obtain latent features / processed datasets.
2. Run classical baselines on the same processed datasets.
3. Run VQC experiments on the same processed datasets.



```python

```
