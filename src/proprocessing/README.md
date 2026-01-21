# Preprocessing

This folder contains scripts for preprocessing and representation learning.

## Scripts

- `DAE_trainer.py`  
  Trains a denoising autoencoder (DAE) on the numeric features and exports latent representations
  and preprocessing artifacts (e.g., imputer/scaler).

## Expected inputs

- `train_transaction.csv` from the IEEE-CIS Fraud Detection dataset (Kaggle).  
  The raw dataset is not included in this repository due to license restrictions.

## Outputs

The script produces latent representations and related artifacts locally (not tracked by Git),
which can then be used to generate small experimental datasets.

## Example

```bash
python src/preprocessing/DAE_trainer.py --csv_path /path/to/train_transaction.csv --latent_dim 32 --out_dir outputs/dae_d32



```python

```
