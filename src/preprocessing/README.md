# Preprocessing

This folder contains scripts for preprocessing and representation learning.

## Script

- `DAE_trainer.py`
  - Trains a denoising autoencoder (DAE) on numeric transaction features.
  - Exports latent representations and preprocessing artifacts.

## Expected Input

- `train_transaction.csv` from the IEEE-CIS Fraud Detection dataset.

The raw dataset is not included in this repository due to license restrictions.

## Generated Outputs

Generated artifacts are stored locally and ignored by Git. They may include latent matrices, trained encoder weights, imputation artifacts, scalers, and fixed small-sample `.npz` datasets.

## Example

```bash
python src/preprocessing/DAE_trainer.py --csv_path /path/to/train_transaction.csv --latent_dim 32 --out_dir outputs/dae_d32
```
