# Quantum Machine Learning for Small-Sample Fraud Detection

Code and data for the paper ：Exploring the Role of Structural Induction Bias in Variational Quantum Circuits under Imbalanced Low-Data Conditions

> **Title:** Exploring the Role of Structural Induction Bias in Variational Quantum Circuits under Imbalanced Low-Data Conditions
> 
> **Author:** Jiahua Li
> 
> **Status:** Under review / Preprint

The project investigates the performance of variational quantum circuits (VQCs)
under small-sample, high-dimensional settings, and compares them with classical
machine learning baselines.

## Repository Structure
- `src/` : Reproducible implementation (preprocessing, ML, QML)
- `data/` : Dataset descriptions and processed small-sample datasets
- `notebook/` : Exploratory and analysis notebooks
- `figures/` : figures used in the paper



## Workflow Overview

1. **Preprocessing**  
   Train a denoising autoencoder (DAE) to obtain latent representations.

2. **Small-sample construction**  
   Construct fixed small datasets (e.g., N=100/200) for fair comparison.

3. **Classical baselines**  
   Evaluate Logistic Regression, Random Forest, and MLP.

4. **Quantum models**  
   Evaluate VQC variants under repeated stratified cross-validation.

## Reproducibility

All reproducible experiments are implemented as standalone Python scripts under `src/`.
Jupyter notebooks are provided for transparency and visualization purposes.

## Data Availability

The original IEEE-CIS Fraud Detection dataset cannot be redistributed due to license
restrictions. Please download the dataset from Kaggle and follow the preprocessing
instructions provided in `data/`.

## Citation

If you use this code, please cite the corresponding paper.



```python

```
