"""
run_all.py -- runs the full pipeline in order.

Usage (from the repository root):
    python src/run_all.py

Each script can also be run individually, e.g.:
    python src/03_classical_ml.py
Outputs (dataset/, results/, reports/) are created in the repo root.
You can pass a split file to the model scripts, e.g.:
    python src/03_classical_ml.py dataset/splits_track_aware.csv
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../src
ROOT = os.path.dirname(HERE)                          # repo root

STEPS = [
    "01_prepare_dataset.py",
    "02_analyze_split.py",
    "03_classical_ml.py",
    "04_cnn_scratch.py",
    "05_transfer_resnet.py",
    "06_compare.py",
]

for s in STEPS:
    print("\n" + "=" * 70 + f"\nRUNNING {s}\n" + "=" * 70)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=ROOT)
    if r.returncode != 0:
        print(f"\n!! {s} failed (exit {r.returncode}). Stopping.")
        sys.exit(r.returncode)

print("\nAll steps finished. See results/ and reports/figures/.")
