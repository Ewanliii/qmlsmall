# Quantum ML Experiments

This folder contains variational quantum circuit (VQC) experiments.

## Script

- `VQC_work.py`
  - Runs VQC variants under repeated stratified cross-validation.
  - Reports AUC summaries and optionally saves per-run results.

## Inputs

Recommended input:

- a fixed small dataset `.npz` containing `X` and `y`.

The experiments are designed to compare circuit-level choices under controlled data and training settings.
