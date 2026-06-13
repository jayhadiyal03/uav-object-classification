"""
02_analyze_split.py
-------------------
1. Computes and plots dataset statistics (class distribution, crop sizes).
2. Saves a sample montage of crops per class.
3. Creates two reproducible train/val/test splits:
     - stratified  : instance-level, stratified by class (standard practice)
     - track_aware : groups all crops of one physical object (track_id) into a
                     single split, so the same object never appears in both
                     train and test (honest, harder, exposes leakage).
Outputs go to  reports/figures/  and  dataset/splits_*.csv
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RNG = 42
IDX = "dataset/index.csv"
CROPS = "dataset/crops"
FIG = "reports/figures"
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(IDX)
order = df["class_name"].value_counts().index.tolist()

# ---------- 1. class distribution ----------
counts = df["class_name"].value_counts()
plt.figure(figsize=(7, 4))
bars = plt.bar(counts.index, counts.values, color="#3b6ea5")
plt.bar_label(bars)
plt.ylabel("number of crops")
plt.title("Class distribution (8,903 crops, 7 classes)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG}/class_distribution.png", dpi=130)
plt.close()

# ---------- 2. crop size distribution ----------
df["area"] = df["w"] * df["h"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(df["w"], df["h"], s=4, alpha=0.2, color="#3b6ea5")
ax[0].set_xlabel("width (px)"); ax[0].set_ylabel("height (px)")
ax[0].set_title("Crop width vs height")
for c in order:
    ax[1].hist(np.sqrt(df[df.class_name == c]["area"]), bins=40,
               histtype="step", label=c)
ax[1].set_xlabel("sqrt(area) (px)"); ax[1].set_ylabel("count")
ax[1].set_title("Object scale per class"); ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{FIG}/size_distribution.png", dpi=130)
plt.close()

# ---------- 3. sample montage ----------
fig, axes = plt.subplots(len(order), 6, figsize=(9, 1.4 * len(order)))
for i, c in enumerate(order):
    sub = df[df.class_name == c].sample(min(6, (df.class_name == c).sum()),
                                        random_state=RNG)
    for j, (_, row) in enumerate(sub.iterrows()):
        im = cv2.cvtColor(cv2.imread(os.path.join(CROPS, row["path"])),
                          cv2.COLOR_BGR2RGB)
        axes[i, j].imshow(cv2.resize(im, (64, 64)))
        axes[i, j].axis("off")
    axes[i, 0].set_ylabel(c, rotation=0, ha="right", va="center", fontsize=9)
    axes[i, 0].axis("on"); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
plt.suptitle("Sample crops per class")
plt.tight_layout()
plt.savefig(f"{FIG}/sample_montage.png", dpi=130)
plt.close()

# ---------- 4a. stratified instance-level split (70/15/15) ----------
tr, tmp = train_test_split(df, test_size=0.30, stratify=df["class_name"],
                           random_state=RNG)
va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["class_name"],
                          random_state=RNG)
strat = df.copy()
strat["split"] = "train"
strat.loc[va.index, "split"] = "val"
strat.loc[te.index, "split"] = "test"
strat.to_csv("dataset/splits_stratified.csv", index=False)

# ---------- 4b. track-aware split (group by class+track) ----------
rng = np.random.default_rng(RNG)
assign = {}
for c in order:
    tracks = df[df.class_name == c]["track_id"].unique()
    rng.shuffle(tracks)
    n = len(tracks)
    n_tr = max(1, int(round(0.70 * n)))
    n_va = max(1, int(round(0.15 * n))) if n - n_tr >= 2 else 0
    for t in tracks[:n_tr]:
        assign[(c, t)] = "train"
    for t in tracks[n_tr:n_tr + n_va]:
        assign[(c, t)] = "val"
    for t in tracks[n_tr + n_va:]:
        assign[(c, t)] = "test"
track = df.copy()
track["split"] = [assign[(r.class_name, r.track_id)]
                  for r in df.itertuples()]
track.to_csv("dataset/splits_track_aware.csv", index=False)

# ---------- summary table ----------
print("=== Stratified split (instances) ===")
print(pd.crosstab(strat.class_name, strat.split).loc[order])
print("\n=== Track-aware split (instances) ===")
print(pd.crosstab(track.class_name, track.split).loc[order])
print("\n=== Track-aware split (unique tracks) ===")
tk = track.drop_duplicates(["class_name", "track_id"])
print(pd.crosstab(tk.class_name, tk.split).loc[order])
print("\nFigures saved to reports/figures/")
