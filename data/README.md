# Data Description

This directory contains subsampled datasets derived from a large-scale original dataset 
(approximately 600,000 samples), which cannot be redistributed due to license restrictions.

Dataset source:
- IEEE-CIS Fraud Detection Dataset (Kaggle)

Please download the dataset from https://www.kaggle.com/competitions/ieee-fraud-detection/data
and follow the preprocessing instructions provided in `data/processed/README.md`.

These subsampled datasets are used consistently across all experiments to ensure
fair comparison between different models and circuit architectures.


All required experimental content can be generated via `src/
Preprocessing
/README.md`. The generated content includes:

`y_labels.npy`: Labels corresponding to each row of `Zdae_encoder`.pt: Weights for the DAE encoder

`imputer.joblib`: Rules for imputing missing values

`scaler.joblib`: Rules for standardization

`Xy_small_N100_seed42` contains samples from the N=100 experiment

`Xy_small_N200_seed42` contains samples from the N=200 experiment

`Z_latent_D16.npy` contains nearly 600,000 data samples and their corresponding dimensions. Due to file size limitations, we will not upload it. It can be trained following the instructions in `DAE_trainer`.



```python

```
