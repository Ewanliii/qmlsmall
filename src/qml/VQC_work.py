"""
VQC experiment runner on small-sample latent features (IEEE-CIS Fraud Detection).

This script:
1) Loads a fixed small dataset (recommended) from .npz containing X and y.
   Alternatively, it can sample from full latent arrays Z and y (not recommended for strict reproducibility).
2) Runs repeated stratified CV for one or multiple circuit variants.
3) Trains a simple VQC using PennyLane (lightning.qubit) with weighted BCE loss.
4) Reports AUC mean±std and optionally saves per-run results to CSV.

IMPORTANT:
- Do NOT commit raw Kaggle CSV or full latent matrices (Z_latent_*.npy) to GitHub.
- Commit only this script and configuration/README files.
"""

from __future__ import annotations

import os
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

import pennylane as qml
from pennylane import numpy as pnp


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)


# -----------------------------
# Data utilities
# -----------------------------
def load_small_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Small dataset not found: {npz_path}")

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"].astype(int)

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError(f"Invalid shapes: X={X.shape}, y={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y row mismatch: {X.shape[0]} vs {y.shape[0]}")

    print(f"Loaded small dataset: X={X.shape}, y={y.shape}, pos={int(y.sum())}, pos_rate={y.mean():.4f}")
    return X, y


def load_full_latents(z_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Z not found: {z_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y not found: {y_path}")

    Z = np.load(z_path)
    y = np.load(y_path).astype(int)
    print(f"Loaded full latents: Z={Z.shape}, y={y.shape}, pos_rate={y.mean():.6f}")
    return Z, y


def stratified_subsample_fixed_pos(Z: np.ndarray, y: np.ndarray, N: int, min_pos: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified subsample ensuring at least `min_pos` positives and at most N/2 positives.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    target_pos = max(min_pos, int(round(N * y.mean())))
    target_pos = min(target_pos, N // 2)
    target_pos = min(target_pos, len(pos_idx))
    target_neg = N - target_pos

    if target_neg > len(neg_idx):
        raise ValueError("Not enough negatives for the requested N.")

    sel_pos = rng.choice(pos_idx, size=target_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=target_neg, replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)

    X_small = Z[sel]
    y_small = y[sel]
    print(f"Subsampled: N={len(y_small)}, pos={int(y_small.sum())}, pos_rate={y_small.mean():.4f}")
    return X_small, y_small


# -----------------------------
# Preprocessing for angle encoding
# -----------------------------
def scale_to_pi(X_train: np.ndarray, X_test: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize using training statistics -> tanh squashing -> map to [-pi, pi].
    """
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + eps

    Xtr = (X_train - mu) / std
    Xte = (X_test - mu) / std

    Xtr = np.tanh(Xtr) * np.pi
    Xte = np.tanh(Xte) * np.pi
    return Xtr, Xte


# -----------------------------
# VQC definition
# -----------------------------
@dataclass
class VQCConfig:
    n_qubits: int = 4
    n_layers: int = 1
    steps: int = 30
    lr: float = 0.05
    n_splits: int = 2
    n_repeats: int = 30


def make_device(n_qubits: int):
    # lightning.qubit is fast if installed; fallback to default.qubit otherwise.
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)


def build_circuit(cfg: VQCConfig, variant: str):
    n_qubits = cfg.n_qubits
    dev = make_device(n_qubits)

    def entangle_ring():
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    def entangle_chain():
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

    def entangle():
        if "B1" in variant:
            entangle_chain()
        else:
            entangle_ring()

    def embed_block(x4):
        # Baseline: RY only
        if "C1" in variant:
            # Enhanced embedding: RY + RZ
            for q in range(n_qubits):
                qml.RY(x4[q], wires=q)
                qml.RZ(x4[q], wires=q)
        else:
            for q in range(n_qubits):
                qml.RY(x4[q], wires=q)

    def trainable_block(weights, l):
        if "A2" in variant:
            # A2: fewer parameters (RY only), weights shape: (L, Q, 1)
            for q in range(n_qubits):
                qml.RY(weights[l, q, 0], wires=q)
        elif "A1" in variant:
            # A1: Rot (3 params), weights shape: (L, Q, 3)
            for q in range(n_qubits):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        else:
            # Baseline: RY + RZ (2 params), weights shape: (L, Q, 2)
            for q in range(n_qubits):
                qml.RY(weights[l, q, 0], wires=q)
                qml.RZ(weights[l, q, 1], wires=q)

    def readout():
        # D1/D2: multi-qubit readout (return tuple of expvals)
        if ("D1" in variant) or ("D2" in variant):
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))
        # Baseline: single-qubit expval
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        # Data re-uploading in chunks of n_qubits
        num_chunks = x.shape[0] // n_qubits
        for l in range(cfg.n_layers):
            for chunk in range(num_chunks):
                xq = x[chunk * n_qubits:(chunk + 1) * n_qubits]
                embed_block(xq)

                # B2: entangle only once per layer (skip for chunk != 0)
                if ("B2" in variant) and (chunk != 0):
                    pass
                else:
                    entangle()

                trainable_block(weights, l)
        return readout()

    return circuit


def sigmoid(z):
    return 1.0 / (1.0 + pnp.exp(-z))


def init_weights(cfg: VQCConfig, variant: str, seed: int):
    rng = np.random.default_rng(seed)
    if "A2" in variant:
        shape = (cfg.n_layers, cfg.n_qubits, 1)
    elif "A1" in variant:
        shape = (cfg.n_layers, cfg.n_qubits, 3)
    else:
        shape = (cfg.n_layers, cfg.n_qubits, 2)
    w0 = 0.01 * rng.standard_normal(size=shape)
    return pnp.array(w0, requires_grad=True)


def predict_proba(circuit, X: np.ndarray, weights, bias, variant: str, w_read=None):
    probs = []
    for i in range(X.shape[0]):
        out = circuit(X[i], weights)

        if isinstance(out, (tuple, list)):
            z = pnp.array(out)  # shape: (n_qubits,)
            if w_read is not None:      # D2: trainable weighted readout
                out_scalar = pnp.dot(w_read, z)
            else:                        # D1: fixed mean readout
                out_scalar = pnp.mean(z)
        else:
            out_scalar = out             # baseline: scalar

        logit = 2.0 * out_scalar + bias
        probs.append(sigmoid(logit))
    return pnp.array(probs)


def weighted_bce_loss(y_true, y_prob, pos_weight: float, eps: float = 1e-8):
    y_true = pnp.array(y_true)
    y_prob = pnp.clip(y_prob, eps, 1 - eps)
    loss = -(pos_weight * y_true * pnp.log(y_prob) + (1 - y_true) * pnp.log(1 - y_prob))
    return pnp.mean(loss)


def train_vqc(circuit, X_train: np.ndarray, y_train: np.ndarray, cfg: VQCConfig, variant: str, seed: int):
    weights = init_weights(cfg, variant, seed)
    bias = pnp.array(0.0, requires_grad=True)

    # D2: trainable weighted readout head
    if "D2" in variant:
        w_read = pnp.array(np.zeros(cfg.n_qubits), requires_grad=True)
    else:
        w_read = None

    pos = int(np.sum(y_train))
    neg = len(y_train) - pos
    pos_weight = float(neg / max(pos, 1))

    opt = qml.AdamOptimizer(stepsize=cfg.lr)

    def objective(w, b, wr):
        probs = predict_proba(circuit, X_train, w, b, variant, wr)
        return weighted_bce_loss(y_train, probs, pos_weight=pos_weight)

    for _ in range(cfg.steps):
        if "D2" in variant:
            weights, bias, w_read = opt.step(objective, weights, bias, w_read)
        else:
            weights, bias = opt.step(lambda w_, b_: objective(w_, b_, None), weights, bias)

    return weights, bias, w_read


# -----------------------------
# Experiment loop
# -----------------------------
def run_experiment(
    X_small: np.ndarray,
    y_small: np.ndarray,
    variants: List[str],
    cv_seeds: List[int],
    cfg: VQCConfig,
) -> pd.DataFrame:
    all_rows: List[Dict] = []
    t_start = time.time()

    for variant in variants:
        print("\n==============================")
        print("Running VARIANT:", variant)
        print("==============================")

        circuit = build_circuit(cfg, variant)

        for cv_seed in cv_seeds:
            print(f"\n--- CV seed = {cv_seed} ---")
            rskf = RepeatedStratifiedKFold(
                n_splits=cfg.n_splits,
                n_repeats=cfg.n_repeats,
                random_state=cv_seed,
            )

            auc_scores = []
            skipped = 0

            for fold_id, (train_idx, test_idx) in enumerate(rskf.split(X_small, y_small), 1):
                X_train_raw, X_test_raw = X_small[train_idx], X_small[test_idx]
                y_train, y_test = y_small[train_idx], y_small[test_idx]

                # Skip single-class folds (rare but safer)
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    skipped += 1
                    continue

                # Fold-wise scaling to [-pi, pi]
                X_train, X_test = scale_to_pi(X_train_raw, X_test_raw)

                # Train VQC (fold-specific init seed for reproducibility)
                weights, bias, w_read = train_vqc(
                    circuit=circuit,
                    X_train=X_train,
                    y_train=y_train,
                    cfg=cfg,
                    variant=variant,
                    seed=1000 * cv_seed + fold_id,
                )

                # Predict probabilities and compute AUC
                y_prob = np.array(predict_proba(circuit, X_test, weights, bias, variant, w_read), dtype=float)
                auc_scores.append(roc_auc_score(y_test, y_prob))

            mean_auc = float(np.mean(auc_scores)) if len(auc_scores) > 0 else np.nan
            std_auc = float(np.std(auc_scores)) if len(auc_scores) > 0 else np.nan

            print(f"[{variant}] seed={cv_seed}  AUC={mean_auc:.4f} ± {std_auc:.4f}  folds={len(auc_scores)}  skipped={skipped}")

            all_rows.append({
                "VARIANT": variant,
                "cv_seed": cv_seed,
                "n_splits": cfg.n_splits,
                "n_repeats": cfg.n_repeats,
                "folds_used": len(auc_scores),
                "skipped": skipped,
                "steps": cfg.steps,
                "lr": cfg.lr,
                "n_layers": cfg.n_layers,
                "n_qubits": cfg.n_qubits,
                "AUC_mean": mean_auc,
                "AUC_std": std_auc,
            })

    df_results = pd.DataFrame(all_rows)

    print("\n======== SUMMARY (by VARIANT) ========")
    if len(df_results) > 0:
        print(df_results.groupby("VARIANT")[["AUC_mean", "AUC_std"]].mean())
    print(f"\nTotal elapsed: {(time.time() - t_start)/60:.1f} minutes")
    print("pos:", int(y_small.sum()), "pos_rate:", y_small.mean())

    return df_results


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VQC variants with repeated stratified CV.")
    data_grp = p.add_mutually_exclusive_group(required=True)
    data_grp.add_argument("--small_npz", type=str, help="Path to small dataset .npz with keys X,y (recommended).")
    data_grp.add_argument("--full_latents", action="store_true", help="Load full Z,y and subsample (not recommended).")

    p.add_argument("--z_path", type=str, default="", help="Path to full latent Z (.npy), required if --full_latents.")
    p.add_argument("--y_path", type=str, default="", help="Path to labels y (.npy), required if --full_latents.")
    p.add_argument("--N", type=int, default=100, help="Subsample size when using --full_latents.")
    p.add_argument("--min_pos", type=int, default=5, help="Minimum positives in subsample when using --full_latents.")
    p.add_argument("--seed", type=int, default=42, help="Seed for subsampling (when using --full_latents).")

    p.add_argument("--variants", type=str, default="baseline",
                   help='Comma-separated list, e.g. "baseline,A1,B1,C1,D1,B1+C1,D2,B1+D2+B2"')
    p.add_argument("--cv_seeds", type=str, default="42", help='Comma-separated CV seeds, e.g. "42,7,13"')

    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n_splits", type=int, default=2)
    p.add_argument("--n_repeats", type=int, default=30)

    p.add_argument("--save_csv", type=str, default="", help="If set, save results to this CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = VQCConfig(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        steps=args.steps,
        lr=args.lr,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
    )

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    cv_seeds = [int(s.strip()) for s in args.cv_seeds.split(",") if s.strip()]

    # Load data
    if args.small_npz:
        X_small, y_small = load_small_npz(args.small_npz)
    else:
        if not args.z_path or not args.y_path:
            raise ValueError("When using --full_latents, you must provide --z_path and --y_path.")
        Z, y = load_full_latents(args.z_path, args.y_path)
        X_small, y_small = stratified_subsample_fixed_pos(Z, y, N=args.N, min_pos=args.min_pos, seed=args.seed)

    # Run
    df_results = run_experiment(X_small, y_small, variants, cv_seeds, cfg)

    # Save
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        df_results.to_csv(args.save_csv, index=False)
        print(f"\nSaved -> {args.save_csv}")


if __name__ == "__main__":
    main()
