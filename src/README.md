# Source Code

This directory contains the main scripts used in the project.

## Structure

- `preprocessing/`
  - Denoising autoencoder training and latent feature extraction.

- `classical_ml/`
  - Classical machine learning baselines, including Logistic Regression, Random Forest, and MLP.

- `qml/`
  - Variational quantum circuit experiments and architecture variants.

## General Workflow

1. Run preprocessing to obtain latent representations and fixed small-sample datasets.
2. Run classical baselines on the same processed datasets.
3. Run VQC experiments on the same processed datasets.
4. Compare model behavior under unified evaluation settings.

Each subdirectory contains a short README with its role in the workflow.
