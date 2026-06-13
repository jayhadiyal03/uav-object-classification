"""
common.py
---------
Shared helpers used by every model script: loading a split, loading images at
a fixed size, label encoding, and a single metrics/reporting function so all
models are evaluated identically (accuracy, macro & weighted F1, per-class
report, confusion matrix figure).
"""
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)

CROPS = "dataset/crops"
# fixed class order (by frequency) -> integer labels
CLASS_ORDER = ["car", "pedestrian", "motor", "people", "bus", "van", "bicycle"]
CLS2IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
os.makedirs("results", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


def load_split(split_csv, split):
    df = pd.read_csv(split_csv)
    df = df[df.split == split]
    paths = [os.path.join(CROPS, p) for p in df["path"]]
    labels = np.array([CLS2IDX[c] for c in df["class_name"]])
    return paths, labels


def load_images(paths, size, gray=False):
    """Return uint8 array (N,size,size,3) or (N,size,size) if gray."""
    out = []
    for p in paths:
        im = cv2.imread(p)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        if gray:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out.append(im)
    return np.array(out)


def evaluate(model_name, y_true, y_pred, extra=None):
    """Print + save metrics, save a confusion-matrix figure, return dict."""
    acc = accuracy_score(y_true, y_pred)
    res = {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred,
                                            average="weighted")), 4),
        "macro_precision": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
    }
    if extra:
        res.update(extra)

    report = classification_report(y_true, y_pred, target_names=CLASS_ORDER,
                                   zero_division=0, output_dict=True)
    res["per_class"] = {c: {"precision": round(report[c]["precision"], 3),
                            "recall": round(report[c]["recall"], 3),
                            "f1": round(report[c]["f1-score"], 3),
                            "support": int(report[c]["support"])}
                        for c in CLASS_ORDER}

    with open(f"results/{model_name}.json", "w") as f:
        json.dump(res, f, indent=2)

    # confusion matrix (row-normalised)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(5.5, 4.6))
    plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(7), CLASS_ORDER, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(7), CLASS_ORDER, fontsize=8)
    for i in range(7):
        for j in range(7):
            if cm[i, j] > 0.01:
                plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                         fontsize=7,
                         color="white" if cm[i, j] > 0.5 else "black")
    plt.ylabel("true"); plt.xlabel("predicted")
    plt.title(f"{model_name}\nacc={res['accuracy']:.3f}  "
              f"macro-F1={res['macro_f1']:.3f}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/cm_{model_name}.png", dpi=130)
    plt.close()

    print(f"[{model_name}]  acc={res['accuracy']:.4f}  "
          f"macroF1={res['macro_f1']:.4f}  wF1={res['weighted_f1']:.4f}")
    return res
