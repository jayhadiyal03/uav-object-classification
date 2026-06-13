"""
06_compare.py
-------------
Reads every results/*.json produced by the model scripts and builds:
  - a printed comparison table (accuracy, macro-F1, weighted-F1, ...)
  - reports/figures/model_comparison.png  (grouped bar chart)
  - reports/results_summary.csv
Run this LAST, after the model scripts.
"""
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for fn in sorted(os.listdir("results")):
    if not fn.endswith(".json"):
        continue
    d = json.load(open(os.path.join("results", fn)))
    rows.append({"model": d["model"], "accuracy": d["accuracy"],
                 "macro_f1": d["macro_f1"], "weighted_f1": d["weighted_f1"],
                 "macro_precision": d.get("macro_precision"),
                 "macro_recall": d.get("macro_recall")})

df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
df.to_csv("reports/results_summary.csv", index=False)
print(df.to_string(index=False))

# bar chart of the main "stratified" models (no _track suffix)
main = df[~df.model.str.contains("_track")].set_index("model")
if len(main):
    ax = main[["accuracy", "macro_f1", "weighted_f1"]].plot.bar(
        figsize=(9, 4.5), ylim=(0, 1))
    ax.set_ylabel("score"); ax.set_title("Model comparison (test set)")
    ax.legend(loc="lower right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison.png", dpi=130)
    print("\nSaved reports/figures/model_comparison.png and "
          "reports/results_summary.csv")
