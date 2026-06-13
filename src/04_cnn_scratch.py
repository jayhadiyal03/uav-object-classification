"""
04_cnn_scratch.py  -- MODEL B
A compact CNN (3 conv blocks + FC head) trained from scratch on 64x64 RGB
crops. Uses class-weighted cross-entropy + light augmentation to handle
imbalance. Saves training curves and selects the epoch with best val macro-F1.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from common import load_split, load_images, evaluate, CLASS_ORDER

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dataset/splits_stratified.csv"
TAG = "" if "stratified" in SPLIT else "_track"
SIZE, BS = 64, 64
EPOCHS = int(os.environ.get("EPOCHS", 18))
torch.manual_seed(42); np.random.seed(42)
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def to_tensor(paths):
    x = load_images(paths, SIZE).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)


class SmallCNN(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Dropout(0.4), nn.Linear(128, n))

    def forward(self, x):
        return self.head(self.features(x))


def main():
    ptr, ytr = load_split(SPLIT, "train")
    pva, yva = load_split(SPLIT, "val")
    pte, yte = load_split(SPLIT, "test")
    Xtr, Xva, Xte = to_tensor(ptr), to_tensor(pva), to_tensor(pte)
    ytr_t, yva_t = torch.tensor(ytr), torch.tensor(yva)

    cw = compute_class_weight("balanced", classes=np.arange(7), y=ytr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(Xtr, ytr_t), batch_size=BS,
                        shuffle=True)

    hist = {"train_loss": [], "val_f1": []}
    best_f1, best_state = -1, None
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0
        for xb, yb in loader:
            # light augmentation: random horizontal flip
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=[3])
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vp = model(Xva).argmax(1).numpy()
        vf1 = f1_score(yva, vp, average="macro")
        hist["train_loss"].append(tot / len(Xtr))
        hist["val_f1"].append(vf1)
        if vf1 > best_f1:
            best_f1, best_state = vf1, {k: v.clone()
                                        for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d}  loss={tot/len(Xtr):.3f}  val_macroF1={vf1:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    evaluate(f"B_cnn{TAG}", yte, pred,
             extra={"params": sum(p.numel() for p in model.parameters())})

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(hist["train_loss"], "o-", color="#d1495b", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="#d1495b")
    ax2 = ax1.twinx()
    ax2.plot(hist["val_f1"], "s-", color="#3b6ea5", label="val macro-F1")
    ax2.set_ylabel("val macro-F1", color="#3b6ea5")
    plt.title("CNN-from-scratch training")
    plt.tight_layout(); plt.savefig(f"reports/figures/cnn_curve{TAG}.png",
                                    dpi=130)


if __name__ == "__main__":
    main()
