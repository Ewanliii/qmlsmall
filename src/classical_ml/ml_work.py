"""
Classical ML baselines (LR / RF / MLP) on small-sample latent features.

This script:
1) Loads latent features Z and labels y (from local files, not tracked by Git).
2) Draws a stratified small sample (N=100/200/...) with a fixed seed.
3) Optionally saves the sampled dataset to .npz for reuse.
4) Evaluates LR / RF / MLP with RepeatedStratifiedKFold and reports mean±std AUC.
5) Optionally runs a single train/test split for quick sanity checking.

Notes:
- Do NOT commit raw Kaggle data or full latent matrices to GitHub.
- Commit only this script and documentation. Keep data artifacts locally.
"""

from __future__ import annotations

import os
import argparse
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# Utilities
# -----------------------------
def load_arrays(z_path: str, y_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"Z not found: {z_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y not found: {y_path}")

    Z = np.load(z_path)
    y = np.load(y_path)

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D array. Got shape: {Z.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array. Got shape: {y.shape}")
    if Z.shape[0] != y.shape[0]:
        raise ValueError(f"Z and y must have the same number of rows. Got {Z.shape[0]} vs {y.shape[0]}")

    print(f"Z shape: {Z.shape} | y shape: {y.shape} | positive rate: {y.mean():.6f}")
    return Z, y


def stratified_subsample(
    Z: np.ndarray,
    y: np.ndarray,
    N: int,
    seed: int,
    min_pos: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw a stratified subsample to approximately match the population positive rate,
    while ensuring at least `min_pos` positive examples.

    Returns:
        X_small, y_small, selected_indices
    """
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Target positives: proportional to base rate but at least min_pos
    target_pos = max(min_pos, int(round(N * y.mean())))
    target_pos = min(target_pos, len(pos_idx))
    target_neg = N - target_pos
    if target_neg > len(neg_idx):
        raise ValueError("Not enough negative samples for the requested N.")

    sel_pos = rng.choice(pos_idx, size=target_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=target_neg, replace=False)

    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)

    X_small = Z[sel]
    y_small = y[sel]

    print(f"Subsampled: N={len(y_small)} | pos={int(y_small.sum())} | pos_rate={y_small.mean():.4f}")
    return X_small, y_small, sel


def save_small_npz(out_path: str, X_small: np.ndarray, y_small: np.ndarray, sel_idx: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, X=X_small, y=y_small, idx=sel_idx)
    print(f"Saved subsample to: {out_path} (compressed .npz)")


def load_small_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Small-sample npz not found: {npz_path}")
    data = np.load(npz_path)
    X_small = data["X"]
    y_small = data["y"]
    idx = data["idx"] if "idx" in data.files else None
    print(f"Loaded subsample: X={X_small.shape}, y={y_small.shape}, pos={int(y_small.sum())}, pos_rate={y_small.mean():.4f}")
    return X_small, y_small, idx


def build_models(seed: int) -> tuple[LogisticRegression, RandomForestClassifier, MLPClassifier]:
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        alpha=1e-2,
        max_iter=800,
        random_state=seed,
    )
    return lr, rf, mlp


def evaluate_repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
) -> None:
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    lr_scores, rf_scores, mlp_scores = [], [], []
    skipped = 0

    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip folds with a single class (rare but possible in tiny samples)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            skipped += 1
            continue

        lr, rf, mlp = build_models(seed)

        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        mlp.fit(X_train, y_train)

        lr_scores.append(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
        rf_scores.append(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
        mlp_scores.append(roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1]))

    print("Skipped folds:", skipped)
    print(f"LR : {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
    print(f"RF : {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
    print(f"MLP: {np.mean(mlp_scores):.4f} ± {np.std(mlp_scores):.4f}")


def evaluate_single_split(X: np.ndarray, y: np.ndarray, seed: int, test_size: float) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    lr, rf, mlp = build_models(seed)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    auc_mlp = roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

    print(f"LR  AUC: {auc_lr:.4f}")
    print(f"RF  AUC: {auc_rf:.4f}")
    print(f"MLP AUC: {auc_mlp:.4f}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical ML baselines on small-sample latent features.")
    parser.add_argument("--z_path", type=str, required=True, help="Path to latent feature matrix Z (.npy).")
    parser.add_argument("--y_path", type=str, required=True, help="Path to labels y (.npy).")

    # Sampling
    parser.add_argument("--N", type=int, default=200, help="Subsample size (e.g., 100/200).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling and CV.")
    parser.add_argument("--min_pos", type=int, default=2, help="Minimum number of positive examples in subsample.")

    # Option: save or load a fixed subsample
    parser.add_argument("--save_npz", type=str, default="", help="If set, save the subsample to this .npz path.")
    parser.add_argument("--load_npz", type=str, default="", help="If set, load subsample from this .npz path instead of sampling.")

    # Evaluation
    parser.add_argument("--mode", type=str, choices=["cv", "single"], default="cv",
                        help="Evaluation mode: repeated CV or a single train/test split.")
    parser.add_argument("--n_splits", type=int, default=2, help="Number of splits for RepeatedStratifiedKFold.")
    parser.add_argument("--n_repeats", type=int, default=50, help="Number of repeats for RepeatedStratifiedKFold.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test size for single split mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load or sample
    if args.load_npz:
        X_small, y_small, _ = load_small_npz(args.load_npz)
    else:
        Z, y = load_arrays(args.z_path, args.y_path)
        X_small, y_small, sel = stratified_subsample(Z, y, N=args.N, seed=args.seed, min_pos=args.min_pos)

        if args.save_npz:
            save_small_npz(args.save_npz, X_small, y_small, sel)

    # Evaluate
    if args.mode == "cv":
        evaluate_repeated_cv(
            X=X_small,
            y=y_small,
            seed=args.seed,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
        )
    else:
        evaluate_single_split(
            X=X_small,
            y=y_small,
            seed=args.seed,
            test_size=args.test_size,
        )


if __name__ == "__main__":
    main()
