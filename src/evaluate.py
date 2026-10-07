import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score


def evaluate_model(model, X_test, y_test, output_dir: str, label: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    # Labels: 1 = positive, 0 = negative.
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    tp, fn, fp, tn = cm.ravel()

    positive_precision = tp / (tp + fp) if (tp + fp) else 0.0
    negative_precision = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title(f"Confusion Matrix\n{label}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Positive", "Negative"])
    ax.set_yticks([0, 1], ["Positive", "Negative"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    fig.savefig(output_dir / f"confusion_{safe}.png", dpi=160)
    plt.close(fig)

    return {
        "positive_precision": positive_precision,
        "negative_precision": negative_precision,
        "accuracy": accuracy,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
