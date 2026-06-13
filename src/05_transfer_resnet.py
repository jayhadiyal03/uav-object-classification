"""
05_transfer_resnet.py  -- MODEL C (transfer learning)
ImageNet-pretrained ResNet18. Images are streamed from disk in batches via a
Dataset (never all loaded into RAM at once), so this runs even on modest PCs.

Two modes (set MODE below):
  "features" : freeze backbone, extract 512-d features, train a small head.
               Fast on CPU.  (default)
  "finetune" : train the whole network end-to-end at a low LR. Best accuracy;
               use a GPU if available.

Run:  python 05_transfer_resnet.py [split_csv]
"""
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import ResNet18_Weights
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from common import load_split, evaluate

MODE = "features"          # "features" (CPU-friendly) or "finetune" (use GPU)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dataset/splits_stratified.csv"
TAG = "" if "stratified" in SPLIT else "_track"
SIZE = 224
BS = 64
EPOCHS = int(os.environ.get("EPOCHS", 25 if MODE == "features" else 8))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CropDataset(Dataset):
    """Loads + normalises one image at a time (low memory)."""
    def __init__(self, paths, labels, augment=False):
        self.paths, self.labels, self.augment = paths, labels, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            im = im[:, ::-1, :].copy()          # horizontal flip
        im = (im - MEAN) / STD
        return (torch.from_numpy(im.transpose(2, 0, 1)),
                torch.tensor(self.labels[i]))


def backbone():
    return torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


@torch.no_grad()
def extract_features(net, loader):
    net.eval()
    feats, ys = [], []
    for xb, yb in loader:
        f = net(xb.to(DEVICE)).squeeze(-1).squeeze(-1)
        feats.append(f.cpu()); ys.append(yb)
    return torch.cat(feats), torch.cat(ys)


def main():
    print(f"Device: {DEVICE}  |  MODE: {MODE}  |  EPOCHS: {EPOCHS}")
    ptr, ytr = load_split(SPLIT, "train")
    pva, yva = load_split(SPLIT, "val")
    pte, yte = load_split(SPLIT, "test")
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(7),
                                           y=ytr), dtype=torch.float32)

    if MODE == "features":
        net = backbone(); net.fc = nn.Identity(); net.to(DEVICE)
        for p in net.parameters():
            p.requires_grad = False
        Ftr, ytr_t = extract_features(
            net, DataLoader(CropDataset(ptr, ytr), batch_size=BS))
        Fva, _ = extract_features(
            net, DataLoader(CropDataset(pva, yva), batch_size=BS))
        Fte, _ = extract_features(
            net, DataLoader(CropDataset(pte, yte), batch_size=BS))

        head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
                             nn.Linear(256, 7)).to(DEVICE)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=cw.to(DEVICE))
        loader = DataLoader(list(zip(Ftr, ytr_t)), batch_size=BS, shuffle=True)
        best_f1, best = -1, None
        for ep in range(1, EPOCHS + 1):
            head.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(head(xb.to(DEVICE)), yb.to(DEVICE))
                loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                vp = head(Fva.to(DEVICE)).argmax(1).cpu().numpy()
            vf1 = f1_score(yva, vp, average="macro")
            if vf1 > best_f1:
                best_f1, best = vf1, {k: v.clone()
                                      for k, v in head.state_dict().items()}
            print(f"epoch {ep:2d}  val_macroF1={vf1:.3f}")
        head.load_state_dict(best); head.eval()
        with torch.no_grad():
            pred = head(Fte.to(DEVICE)).argmax(1).cpu().numpy()

    else:  # finetune
        net = backbone(); net.fc = nn.Linear(512, 7); net.to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=cw.to(DEVICE))
        tr_loader = DataLoader(CropDataset(ptr, ytr, augment=True),
                               batch_size=BS, shuffle=True)
        va_loader = DataLoader(CropDataset(pva, yva), batch_size=BS)
        best_f1, best = -1, None
        for ep in range(1, EPOCHS + 1):
            net.train(); tot = 0
            for xb, yb in tr_loader:
                opt.zero_grad()
                loss = crit(net(xb.to(DEVICE)), yb.to(DEVICE))
                loss.backward(); opt.step(); tot += loss.item() * len(xb)
            net.eval(); vp = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    vp.append(net(xb.to(DEVICE)).argmax(1).cpu())
            vf1 = f1_score(yva, torch.cat(vp).numpy(), average="macro")
            if vf1 > best_f1:
                best_f1, best = vf1, {k: v.clone()
                                      for k, v in net.state_dict().items()}
            print(f"epoch {ep:2d}  loss={tot/len(ptr):.3f}  "
                  f"val_macroF1={vf1:.3f}")
        net.load_state_dict(best); net.eval()
        te_loader = DataLoader(CropDataset(pte, yte), batch_size=BS)
        preds = []
        with torch.no_grad():
            for xb, _ in te_loader:
                preds.append(net(xb.to(DEVICE)).argmax(1).cpu())
        pred = torch.cat(preds).numpy()

    evaluate(f"C_resnet18_{MODE}{TAG}", yte, pred)


if __name__ == "__main__":
    main()
