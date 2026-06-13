"""
03_classical_ml.py  -- MODEL A (baseline)
Handcrafted features (HOG shape descriptor + colour-mean/std) fed to three
classical classifiers: Logistic Regression, Linear SVM, Random Forest.
All use class_weight='balanced' to counter the heavy class imbalance.
"""
import sys
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from common import load_split, load_images, evaluate

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dataset/splits_stratified.csv"
TAG = "" if "stratified" in SPLIT else "_track"
SIZE = 48


def featurize(imgs_rgb):
    feats = []
    for im in imgs_rgb:
        g = np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        h = hog(g, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys")
        colour = np.concatenate([im.reshape(-1, 3).mean(0),
                                 im.reshape(-1, 3).std(0)]) / 255.0
        feats.append(np.concatenate([h, colour]))
    return np.array(feats, dtype=np.float32)


def main():
    ptr, ytr = load_split(SPLIT, "train")
    Xtr = featurize(load_images(ptr, SIZE))
    pte, yte = load_split(SPLIT, "test")
    Xte = featurize(load_images(pte, SIZE))
    print(f"Feature dim={Xtr.shape[1]}  train={len(ytr)}  test={len(yte)}")

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    models = {
        f"A_logreg{TAG}": LogisticRegression(max_iter=2000,
                                             class_weight="balanced"),
        f"A_linsvm{TAG}": LinearSVC(class_weight="balanced", max_iter=5000),
        f"A_rforest{TAG}": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42),
    }
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        evaluate(name, yte, clf.predict(Xte))


if __name__ == "__main__":
    main()
