"""
01_prepare_dataset.py
----------------------
Builds a multi-class image-classification dataset from the VisDrone-MOT
sequence `uav0000297_00000_v`.

For every annotated object (a row in the .txt file) we crop the object out of
its frame using the bounding box and save it into a folder named after its
class. We also record the object's track_id and frame so we can later create a
"track-aware" split that avoids leaking near-duplicate crops of the same
physical object across train/test.

Annotation columns (VisDrone-MOT):
  frame, track_id, x, y, w, h, score, class, truncation, occlusion
"""
import os
import csv
import cv2
import glob
import numpy as np

# --- auto-detect where the data lives (handles a few common layouts) ---
def _find_data():
    cands = [
        ("Corsework_data/uav0000297_00000_v",
         "Corsework_data/uav0000297_00000_v.txt"),
        ("Coursework_data/uav0000297_00000_v",
         "Coursework_data/uav0000297_00000_v.txt"),
        ("uav0000297_00000_v", "uav0000297_00000_v.txt"),
    ]
    for d, a in cands:
        if os.path.isdir(d) and os.path.isfile(a):
            return d, a
    # last resort: search the tree
    hits = glob.glob("**/uav0000297_00000_v.txt", recursive=True)
    if hits:
        a = hits[0]
        return a[:-4], a
    raise FileNotFoundError(
        "Could not find uav0000297_00000_v folder + .txt. Run this script "
        "from the folder that contains the unzipped coursework data.")

SEQ_DIR, ANN_FILE = _find_data()
OUT_DIR   = "dataset/crops"           # crops/<class_name>/<id>.png
INDEX_CSV = "dataset/index.csv"       # metadata for every saved crop
MIN_SIDE  = 8                         # discard degenerate / tiny boxes

CLASS_NAMES = {1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
               5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
               9: "bus", 10: "motor"}


def main():
    ann = np.loadtxt(ANN_FILE, delimiter=",")
    H_img, W_img = cv2.imread(os.path.join(SEQ_DIR, "0000001.jpg")).shape[:2]

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    saved, dropped_ignored, dropped_small = 0, 0, 0

    # group annotations by frame so each frame image is read only once
    frames = np.unique(ann[:, 0]).astype(int)
    for frame in frames:
        img_path = os.path.join(SEQ_DIR, f"{frame:07d}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        fr = ann[ann[:, 0] == frame]
        for r in fr:
            cls = int(r[7])
            if cls not in CLASS_NAMES:        # class 0 = ignored regions
                dropped_ignored += 1
                continue
            x, y, w, h = int(r[2]), int(r[3]), int(r[4]), int(r[5])
            # clip box to image bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W_img, x + w), min(H_img, y + h)
            if (x2 - x1) < MIN_SIDE or (y2 - y1) < MIN_SIDE:
                dropped_small += 1
                continue
            crop = img[y1:y2, x1:x2]
            name = CLASS_NAMES[cls]
            cdir = os.path.join(OUT_DIR, name)
            os.makedirs(cdir, exist_ok=True)
            track_id = int(r[1])
            fname = f"{name}_t{track_id}_f{frame}.png"
            cv2.imwrite(os.path.join(cdir, fname), crop)
            rows.append([os.path.join(name, fname), name, cls, track_id,
                         frame, x2 - x1, y2 - y1])
            saved += 1

    os.makedirs(os.path.dirname(INDEX_CSV), exist_ok=True)
    with open(INDEX_CSV, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["path", "class_name", "class_id", "track_id",
                       "frame", "w", "h"])
        wcsv.writerows(rows)

    print(f"Saved crops          : {saved}")
    print(f"Dropped (ignored cls): {dropped_ignored}")
    print(f"Dropped (too small)  : {dropped_small}")
    print(f"Index written to     : {INDEX_CSV}")


if __name__ == "__main__":
    main()
