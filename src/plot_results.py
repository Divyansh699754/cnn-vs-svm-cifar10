"""
Generate comparison plots from saved results.

Usage:
    python src/plot_results.py                  # uses results/results.npy
    python src/plot_results.py --results path   # custom results file
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def plot_cnn_history(history_path, save_dir):
    """Plot CNN accuracy and loss curves (requires history saved as .npy)."""
    if not os.path.exists(history_path):
        print(f"[!] CNN history file not found at {history_path}, skipping curve plots.")
        return

    h = np.load(history_path, allow_pickle=True).item()

    # --- Standalone accuracy plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h["accuracy"], label="Training Accuracy", linewidth=2)
    ax.plot(h["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax.set_title("CNN Model Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "cnn_accuracy_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # --- Combined accuracy + loss plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(h["accuracy"], label="Training Accuracy", linewidth=2)
    ax1.plot(h["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax1.set_title("CNN Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(h["loss"], label="Training Loss", linewidth=2)
    ax2.plot(h["val_loss"], label="Validation Loss", linewidth=2)
    ax2.set_title("CNN Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "cnn_loss_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_svm_performance(results, save_dir):
    """Bar chart for SVM accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar("SVM", results["svm_accuracy"], color="#5b7fb5", width=0.5)
    ax1.set_title("SVM Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar("SVM", results["svm_loss"], color="#6ab187", width=0.5)
    ax2.set_title("SVM Model Loss")
    ax2.set_ylabel("Loss")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "svm_performance_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_comparison(results, save_dir):
    """Side-by-side comparison bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["CNN", "SVM"]
    colors = ["#5b7fb5", "#6ab187"]

    # Accuracy
    axes[0].bar(labels, [results["cnn_accuracy"], results["svm_accuracy"]], color=colors)
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    # Training time
    axes[1].bar(labels, [results["cnn_train_time"], results["svm_train_time"]], color=colors)
    axes[1].set_title("Training Time Comparison")
    axes[1].set_ylabel("Time (seconds)")
    axes[1].grid(axis="y", alpha=0.3)

    # Memory
    axes[2].bar(labels, [results["cnn_memory_usage"], results["svm_memory_usage"]], color=colors)
    axes[2].set_title("Memory Usage Comparison")
    axes[2].set_ylabel("Memory (MB)")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot CNN vs SVM results")
    parser.add_argument("--results", default=os.path.join(ROOT_DIR, "results", "results.npy"))
    args = parser.parse_args()

    results = np.load(args.results, allow_pickle=True).item()
    plot_svm_performance(results, ASSETS_DIR)
    plot_comparison(results, ASSETS_DIR)

    # Try plotting CNN history if available
    history_path = os.path.join(ROOT_DIR, "results", "cnn_history.npy")
    plot_cnn_history(history_path, ASSETS_DIR)

    print("\nAll plots saved to assets/")


if __name__ == "__main__":
    main()
