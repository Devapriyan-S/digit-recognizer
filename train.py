#!/usr/bin/env python3
"""Train the digit classifier and export it for the browser.

Produces web/model/weights.json, which holds int8-quantised weights plus the
per-tensor scales needed to reconstruct them. Quantisation is what keeps the
model at ~110 KB instead of ~440 KB of float32 — a meaningful difference for a
page that must load instantly.

    python train.py            # downloads MNIST on first run, then trains

The test-set metrics printed here are the ones quoted in the README. They come
from the 10,000 held-out images, never from the training split.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).parent
CACHE = ROOT / ".cache" / "mnist.npz"
OUT = ROOT / "web" / "model"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

HIDDEN = (128, 64)
SEED = 0


def load_mnist() -> tuple[np.ndarray, ...]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MNIST to {CACHE} …")
        urllib.request.urlretrieve(MNIST_URL, CACHE)
    d = np.load(CACHE)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def quantise(matrix: np.ndarray) -> tuple[str, float]:
    """Symmetric per-tensor int8 quantisation, base64-encoded.

    Symmetric (zero maps to zero) rather than asymmetric, because it needs no
    zero-point term in the browser's inner loop and the weights are already
    roughly centred on zero.

    The bytes are base64'd rather than written as a JSON array of integers:
    "-127," costs five characters per weight, so the array form is 320 KB
    against 147 KB base64 — and 101 KB against 96 KB even after gzip.
    """
    scale = float(np.abs(matrix).max()) / 127.0
    if scale == 0:
        return base64.b64encode(np.zeros(matrix.size, dtype=np.int8).tobytes()).decode(), 1.0
    q = np.clip(np.round(matrix / scale), -127, 127).astype(np.int8)
    return base64.b64encode(q.flatten().tobytes()).decode(), scale


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"train {x_train.shape}   test {x_test.shape}")

    # Scale to [0, 1]. The browser applies exactly the same scaling.
    X = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    Xt = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=6,
        validation_fraction=0.1,
        random_state=SEED,
        verbose=True,
    )
    model.fit(X, y_train)

    train_acc = model.score(X, y_train)
    test_acc = model.score(Xt, y_test)
    print(f"\ntrain accuracy {train_acc:.4f}")
    print(f"test  accuracy {test_acc:.4f}")

    pred = model.predict(Xt)
    print("\n" + classification_report(y_test, pred, digits=4))
    cm = confusion_matrix(y_test, pred)
    print("confusion matrix (rows = truth, cols = predicted):")
    print(cm)

    # The pairs the model actually confuses, worst first — worth telling the
    # user about, since it explains most misreads in the demo.
    off = [(int(cm[i, j]), int(i), int(j))
           for i in range(10) for j in range(10) if i != j]
    off.sort(reverse=True)
    print("\ntop confusions:", [f"{t}->{p}: {n}" for n, t, p in off[:6]])

    OUT.mkdir(parents=True, exist_ok=True)
    layers = []
    for W, b in zip(model.coefs_, model.intercepts_):
        qw, scale = quantise(W.astype(np.float32))
        layers.append({
            "shape": list(W.shape),
            "w": qw,          # base64 int8
            "scale": scale,
            # Biases stay float32: there are only a few hundred of them, so
            # quantising saves nothing and costs accuracy.
            "b": [float(v) for v in b],
        })

    bundle = {
        "architecture": [784, *HIDDEN, 10],
        "activation": "relu",
        "input_scale": 1.0 / 255.0,
        "layers": layers,
        "metrics": {
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_size": int(len(y_test)),
            "top_confusions": [{"truth": t, "predicted": p, "count": n} for n, t, p in off[:6]],
        },
    }
    path = OUT / "weights.json"
    path.write_text(json.dumps(bundle, separators=(",", ":")))
    print(f"\nwrote {path}  ({path.stat().st_size / 1024:.0f} KB)")

    # Verify the quantised model still works before shipping it: a silent
    # accuracy collapse from quantisation would be invisible until someone
    # tried the demo.
    acc_q = quantised_accuracy(bundle, Xt, y_test)
    print(f"quantised test accuracy {acc_q:.4f}  "
          f"(lost {100 * (test_acc - acc_q):.2f} points to int8)")

    (OUT / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=2))


def quantised_accuracy(bundle: dict, X: np.ndarray, y: np.ndarray) -> float:
    """Run the exported int8 model exactly as the browser will."""
    a = X
    for i, layer in enumerate(bundle["layers"]):
        raw = np.frombuffer(base64.b64decode(layer["w"]), dtype=np.int8)
        W = raw.astype(np.float32).reshape(layer["shape"]) * layer["scale"]
        a = a @ W + np.array(layer["b"], dtype=np.float32)
        if i < len(bundle["layers"]) - 1:
            a = np.maximum(a, 0.0)
    return float((a.argmax(axis=1) == y).mean())


if __name__ == "__main__":
    main()
