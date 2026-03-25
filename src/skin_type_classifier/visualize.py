"""Visualization utilities for learning curve results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_learning_curve(
    results_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "test_macro_f1",
    target_metric: float | None = None,
) -> Path:
    """Plot learning curve: metric vs. number of training images.

    Args:
        results_df: DataFrame with columns ``[fraction, seed, n_train_images, ...]``.
        output_dir: Directory to save the plot.
        metric: Which metric column to plot on the y-axis.
        target_metric: If provided, draw a horizontal target line.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = results_df.groupby("n_train_images")[metric]
    means = grouped.mean()
    stds = grouped.std()
    x = means.index.values
    y_mean = means.values
    y_std = stds.fillna(0).values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_mean, "o-", color="#2563eb", linewidth=2, markersize=6, label=f"Mean {metric}")
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color="#2563eb", label="\u00b11 std dev")

    if target_metric is not None:
        ax.axhline(y=target_metric, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Target: {target_metric}")

    ax.set_xlabel("Number of Training Images", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Learning Curve: Performance vs. Training Set Size", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig_path = output_dir / f"learning_curve_{metric}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_per_class_f1_heatmap(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot per-class F1 scores across training fractions as a heatmap.

    Args:
        results_df: DataFrame with columns including ``test_per_class_f1_0..5``.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    f1_cols = [f"test_per_class_f1_{i}" for i in range(6)]
    fst_labels = ["FST 1", "FST 2", "FST 3", "FST 4", "FST 5", "FST 6"]

    pivot = results_df.groupby("n_train_images")[f1_cols].mean()
    pivot.columns = fst_labels
    data = pivot.values

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index.astype(int), rotation=45, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(fst_labels)
    ax.set_xlabel("Number of Training Images", fontsize=12)
    ax.set_ylabel("Skin Type", fontsize=12)
    ax.set_title("Per-Class F1 Score vs. Training Set Size", fontsize=14)
    fig.colorbar(im, ax=ax, label="F1 Score")

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            val = data[row_idx, col_idx]
            color = "white" if val > 0.5 else "black"
            ax.text(row_idx, col_idx, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig_path = output_dir / "per_class_f1_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path
