# Quantum Machine Learning for Small-Sample Financial Fraud Detection

This repository contains the implementation for my first-author manuscript:

**Exploring the Role of Structural Inductive Bias in Variational Quantum Circuits under Imbalanced Low-Data Conditions**

**Author:** Jiahua Li  
**Status:** Manuscript submitted to *Quantum Machine Intelligence*, 2026

## Overview

This project studies how variational quantum circuit (VQC) design choices affect model behavior under small-sample and class-imbalanced financial fraud detection settings. The experiments compare multiple VQC circuit variants against classical machine learning baselines under unified training budgets and fixed evaluation protocols.

The project is intended as a research code sample. It demonstrates:

- controlled experimental design under limited-data settings;
- baseline benchmarking against Logistic Regression, Random Forest, and MLP;
- VQC implementation and evaluation with PennyLane;
- analysis of architecture choices, entanglement structure, readout design, and generalization behavior;
- reproducible organization of code, notebooks, figures, and data instructions.

## Research Question

Under highly sample-limited and class-imbalanced conditions, do circuit-level structural choices in VQC models affect generalization more clearly than simply increasing circuit depth or training effort?

## Repository Structure

```text
.
├── data/                 # Data description and small processed artifacts
├── figures/              # Main and appendix figures used in the manuscript
├── notebook/             # Exploratory notebooks and plotting notebooks
├── src/
│   ├── preprocessing/    # DAE training and latent feature extraction
│   ├── classical_ml/     # Logistic Regression, Random Forest, and MLP baselines
│   └── qml/              # VQC variants and evaluation scripts
├── .gitignore
└── README.md
```

## Workflow

1. **Preprocessing**
   - Download the IEEE-CIS Fraud Detection dataset from Kaggle.
   - Train a denoising autoencoder (DAE) to produce latent feature representations.

2. **Small-sample construction**
   - Construct fixed small datasets such as `N=100` and `N=200`.
   - Use the same sampled data across classical and quantum models for fair comparison.

3. **Classical baselines**
   - Evaluate Logistic Regression, Random Forest, and MLP under the same train/test protocol.

4. **Quantum model evaluation**
   - Evaluate VQC variants under repeated stratified cross-validation.
   - Compare circuit variants with different rotation, entanglement, embedding, and readout choices.

5. **Analysis and visualization**
   - Generate AUC summaries and figures for main experiments and appendix analyses.

## Data Availability

The original IEEE-CIS Fraud Detection dataset cannot be redistributed due to Kaggle licensing restrictions. Please download it from:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

Large generated artifacts such as raw CSV files, full latent matrices, model weights, and local outputs are intentionally excluded from GitHub. See [`data/README.md`](data/README.md) for details.

## Notes for Reviewers

This repository is organized as a coding sample for PhD applications. The code reflects an independent research project from problem formulation and experiment design to implementation, evaluation, manuscript preparation, and repository organization.

The project is not presented as a production software package. It is a research repository designed to make the experimental workflow, scripts, and results transparent.
