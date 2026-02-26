"""
CNN vs SVM Image Recognition Comparison on CIFAR-10
====================================================
Compares Convolutional Neural Network (CNN) and Support Vector Machine (SVM)
performance on the CIFAR-10 image classification dataset.

Metrics tracked: accuracy, training time, memory usage, and loss.
"""

import os
import time
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Must be set before importing TF

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import psutil

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Utilities ──────────────────────────────────────────────────────────────────
def get_memory_usage_mb():
    """Return current process RSS in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ── CNN ────────────────────────────────────────────────────────────────────────
def build_cnn(input_shape, num_classes=10):
    """Build a simple CNN for CIFAR-10 classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def train_cnn(x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
    """Train CNN and return metrics dict + Keras history."""
    input_shape = x_train.shape[1:]
    model = build_cnn(input_shape)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n[CNN] Training started...")
    mem_start = get_memory_usage_mb()
    t_start = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    train_time = time.time() - t_start
    mem_usage = get_memory_usage_mb() - mem_start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[CNN] Done — accuracy: {test_acc:.4f} | time: {train_time:.1f}s | mem: {mem_usage:.1f} MB")

    metrics = {
        "cnn_accuracy": test_acc,
        "cnn_loss": test_loss,
        "cnn_train_time": train_time,
        "cnn_memory_usage": mem_usage,
    }
    return metrics, history


# ── SVM ────────────────────────────────────────────────────────────────────────
def train_svm(x_train, y_train, x_test, y_test, n_train=3000, pca_variance=0.90):
    """Train linear SVM with PCA and return metrics dict."""
    # Flatten
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)

    # PCA
    pca = PCA(n_components=pca_variance)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    print(f"[SVM] PCA reduced features to {x_train_pca.shape[1]} components")

    # Subset for tractable training
    x_sub = x_train_pca[:n_train]
    y_sub = y_train[:n_train].ravel()

    model = svm.SVC(kernel="linear", probability=True, verbose=False)

    print(f"[SVM] Training on {n_train} samples...")
    mem_start = get_memory_usage_mb()
    t_start = time.time()
    model.fit(x_sub, y_sub)
    train_time = time.time() - t_start
    mem_usage = get_memory_usage_mb() - mem_start

    y_pred = model.predict(x_test_pca)
    y_prob = model.predict_proba(x_test_pca)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)

    print(f"[SVM] Done — accuracy: {acc:.4f} | time: {train_time:.1f}s | mem: {mem_usage:.1f} MB")

    metrics = {
        "svm_accuracy": acc,
        "svm_loss": loss,
        "svm_train_time": train_time,
        "svm_memory_usage": mem_usage,
    }
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CNN vs SVM on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10, help="CNN training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="CNN batch size")
    parser.add_argument("--svm-samples", type=int, default=3000, help="Number of training samples for SVM")
    parser.add_argument("--pca-variance", type=float, default=0.90, help="PCA explained variance ratio")
    args = parser.parse_args()

    # Load CIFAR-10
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Train both models
    cnn_metrics, history = train_cnn(x_train, y_train, x_test, y_test,
                                     epochs=args.epochs, batch_size=args.batch_size)
    svm_metrics = train_svm(x_train, y_train, x_test, y_test,
                            n_train=args.svm_samples, pca_variance=args.pca_variance)

    # Save CNN training history for plotting
    np.save(os.path.join(RESULTS_DIR, "cnn_history.npy"), history.history)

    # Merge & save
    results = {**cnn_metrics, **svm_metrics}

    np.save(os.path.join(RESULTS_DIR, "results.npy"), results)

    with open(os.path.join(RESULTS_DIR, "results.txt"), "w") as f:
        f.write(f"CNN Accuracy:      {results['cnn_accuracy']:.4f}\n")
        f.write(f"CNN Loss:          {results['cnn_loss']:.4f}\n")
        f.write(f"CNN Training Time: {results['cnn_train_time']:.2f} seconds\n")
        f.write(f"CNN Memory Usage:  {results['cnn_memory_usage']:.2f} MB\n")
        f.write(f"SVM Accuracy:      {results['svm_accuracy']:.4f}\n")
        f.write(f"SVM Loss:          {results['svm_loss']:.4f}\n")
        f.write(f"SVM Training Time: {results['svm_train_time']:.2f} seconds\n")
        f.write(f"SVM Memory Usage:  {results['svm_memory_usage']:.2f} MB\n")

    # Summary
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    print(f"  {'Metric':<20} {'CNN':>12} {'SVM':>12}")
    print("-" * 55)
    print(f"  {'Accuracy':<20} {results['cnn_accuracy']:>11.4f} {results['svm_accuracy']:>11.4f}")
    print(f"  {'Loss':<20} {results['cnn_loss']:>11.4f} {results['svm_loss']:>11.4f}")
    print(f"  {'Train Time (s)':<20} {results['cnn_train_time']:>11.1f} {results['svm_train_time']:>11.1f}")
    print(f"  {'Memory (MB)':<20} {results['cnn_memory_usage']:>11.1f} {results['svm_memory_usage']:>11.1f}")
    print("=" * 55)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
