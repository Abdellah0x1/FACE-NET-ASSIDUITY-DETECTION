"""
preprocess.py
dataset/raw/person_XX/*.jpg  ->  dataset/processed/person_XX/*.png (160x160 faces)

- Photos with no face, or with confidence below MIN_CONFIDENCE, are NOT put in
  processed/. Low-confidence crops go to dataset/rejected/ so you can look at them.
- processed/ and rejected/ are rebuilt from scratch at every run: never put
  hand-made files there. Raw photos are never touched.

Run from the project root:   python scripts/preprocess.py
"""
from pathlib import Path
import csv
import shutil
import numpy as np
from PIL import Image
from face_utils import (EXTS, MIN_CONFIDENCE, get_mtcnn, load_image,
                        detect_and_crop, crop_blur)

RAW_DIR = Path("dataset/raw")
OUT_DIR = Path("dataset/processed")
REJECT_DIR = Path("dataset/rejected")
LOG = Path("reports/preprocess_log.csv")


def reset(folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def main():
    people = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if not people:
        raise SystemExit(f"No person folders in {RAW_DIR}")

    mtcnn = get_mtcnn()
    reset(OUT_DIR)
    reset(REJECT_DIR)
    rows = []

    for person_dir in people:
        pid = person_dir.name
        (OUT_DIR / pid).mkdir()
        photos = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in EXTS)
        counts = {"ok": 0, "no_face": 0, "low_confidence": 0, "unreadable": 0}
        blurs = []

        for photo in photos:
            try:
                img = load_image(photo)
            except Exception:
                counts["unreadable"] += 1
                rows.append([pid, photo.name, "unreadable", "", ""])
                continue

            crop, prob = detect_and_crop(mtcnn, img)
            if crop is None:
                status, blur = "no_face", ""
            else:
                blur = round(crop_blur(crop), 1)
                if prob < MIN_CONFIDENCE:
                    status = "low_confidence"
                    (REJECT_DIR / pid).mkdir(exist_ok=True)
                    Image.fromarray(crop).save(REJECT_DIR / pid / f"{photo.stem}.png")
                else:
                    status = "ok"
                    Image.fromarray(crop).save(OUT_DIR / pid / f"{photo.stem}.png")
                    blurs.append(blur)

            counts[status] += 1
            rows.append([pid, photo.name, status, round(prob, 4), blur])

        med = f"{np.median(blurs):.0f}" if blurs else "-"
        print(f"{pid}: {len(photos)} photos -> {counts['ok']} ok, "
              f"{counts['no_face']} no face, {counts['low_confidence']} low confidence, "
              f"{counts['unreadable']} unreadable | median crop blur {med}")

    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "filename", "status", "confidence", "crop_blur"])
        w.writerows(rows)
    print(f"\nLog saved to {LOG}")


if __name__ == "__main__":
    main()