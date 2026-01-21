"""
Train a Denoising Autoencoder (DAE) on IEEE-CIS Fraud Detection tabular features
and export latent representations + preprocessing artifacts.

Notes:
- The original dataset is NOT included in the repository due to license restrictions.
- This script expects you to download `train_transaction.csv` from Kaggle and provide its path.
- It trains a DAE on numeric features (categorical features are ignored).
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Model
# -----------------------------
class DAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# -----------------------------
# Data / Preprocessing
# -----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found at: {csv_path}\n"
            "Please download `train_transaction.csv` from Kaggle (IEEE-CIS Fraud Detection)."
        )
    df = pd.read_csv(csv_path)
    if "isFraud" not in df.columns:
        raise ValueError("Column `isFraud` not found in the dataset.")
    return df


def build_numeric_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep numeric features only, and drop the label column from X.
    """
    y = df["isFraud"].astype(int).to_numpy()
    X = df.select_dtypes(include=[np.number]).drop(columns=["isFraud"])
    print(f"Original numeric feature dimension: {X.shape[1]}")
    return X.to_numpy(), y


def fit_transform_preprocessing(X: np.ndarray) -> Tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """
    Median imputation + standardization.
    Returns transformed X and fitted preprocessors.
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, imputer, scaler


# -----------------------------
# Training / Encoding
# -----------------------------
@dataclass
class TrainConfig:
    latent_dim: int = 16
    batch_size: int = 1024
    num_epochs: int = 20
    learning_rate: float = 1e-3
    noise_std: float = 0.1
    seed: int = 42


def train_dae(
    X_tensor: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> DAE:
    input_dim = X_tensor.shape[1]
    model = DAE(input_dim=input_dim, latent_dim=cfg.latent_dim).to(device)

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)

            # Add Gaussian noise for denoising training
            noise = cfg.noise_std * torch.randn_like(x_batch)
            x_noisy = x_batch + noise

            # Forward
            x_recon = model(x_noisy)
            loss = criterion(x_recon, x_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] - Recon Loss: {avg_loss:.6f}")

    return model


@torch.no_grad()
def encode_latents(
    model: DAE,
    X_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    Z = model.encoder(X_tensor.to(device)).cpu().numpy()
    print(f"Latent representation shape: {Z.shape}")
    return Z


# -----------------------------
# Saving
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_artifacts(
    out_dir: str,
    cfg: TrainConfig,
    Z: np.ndarray,
    y: np.ndarray,
    model: DAE,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> None:
    """
    Save:
    - Z latent representations
    - y labels
    - encoder weights
    - preprocessing objects (imputer/scaler)
    """
    ensure_dir(out_dir)

    # Prefer compressed format for large arrays
    z_path = os.path.join(out_dir, f"Z_latent_D{cfg.latent_dim}.npz")
    y_path = os.path.join(out_dir, "y_labels.npy")
    enc_path = os.path.join(out_dir, f"dae_encoder_D{cfg.latent_dim}.pt")
    imp_path = os.path.join(out_dir, "imputer.joblib")
    sca_path = os.path.join(out_dir, "scaler.joblib")

    np.savez_compressed(z_path, Z=Z)
    np.save(y_path, y)

    torch.save(model.encoder.state_dict(), enc_path)
    joblib.dump(imputer, imp_path)
    joblib.dump(scaler, sca_path)

    print("Saved files:")
    for p in [z_path, y_path, enc_path, imp_path, sca_path]:
        print(" -", p)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DAE and export latent representations.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to `train_transaction.csv` (downloaded from Kaggle IEEE-CIS Fraud Detection).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/dae",
        help="Output directory to save latent representations and artifacts.",
    )
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension (e.g., 8/16/32).")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data
    df = load_data(args.csv_path)

    overall_fraud_rate = df["isFraud"].mean()
    print(f"Overall fraud rate (full dataset): {overall_fraud_rate:.4f}")

    X_raw, y = build_numeric_features(df)
    X_scaled, imputer, scaler = fit_transform_preprocessing(X_raw)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Train
    model = train_dae(X_tensor=X_tensor, cfg=cfg, device=device)

    # Encode
    Z = encode_latents(model=model, X_tensor=X_tensor, device=device)

    # Save
    save_artifacts(
        out_dir=args.out_dir,
        cfg=cfg,
        Z=Z,
        y=y,
        model=model,
        imputer=imputer,
        scaler=scaler,
    )


if __name__ == "__main__":
    main()
