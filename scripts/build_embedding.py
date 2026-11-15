"""
build_embeddings.py
dataset/processed/person_XX/*.png -> embeddings/

Steps:
 1. FaceNet embedding for every processed crop.
 2. Near-duplicate removal inside each person (cosine similarity > DUP_COS).
    Duplicates are listed in dropped_duplicates.csv, files are not deleted.
 3. Reproducible enroll/test split per person (fixed SEED).

Outputs (embeddings/):
  embeddings.npy + labels.csv            enrollment library (what the live system loads)
  test_embeddings.npy + test_labels.csv  test set (never used for enrollment)
  dropped_duplicates.csv, params.json

Run from the project root:   python scripts/build_embeddings.py
"""
from pathlib import Path
from datetime import datetime
import csv
import json
import random
import numpy as np
from PIL import Image
import face_utils as fu

PROCESSED = Path("dataset/processed")
OUT = Path("embeddings")

ENROLL_RATIO = 0.7   # share of each person's photos used for enrollment
SEED = 42
DUP_COS = 0.98       # cosine similarity above this = near-identical photo


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    people = sorted(p for p in PROCESSED.iterdir() if p.is_dir())
    if not people:
        raise SystemExit("Nothing in dataset/processed. Run preprocess.py first.")

    model = fu.get_facenet()
    lib_emb, lib_rows, test_emb, test_rows, dropped = [], [], [], [], []

    for person_dir in people:
        pid = person_dir.name
        files = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in fu.EXTS)
        if not files:
            print(f"{pid}: no processed faces, skipped")
            continue

        crops = [np.array(Image.open(f).convert("RGB")) for f in files]
        emb = fu.embed(model, crops)

        kept = []
        for i in range(len(files)):
            twin = next((j for j in kept if float(emb[i] @ emb[j]) > DUP_COS), None)
            if twin is None:
                kept.append(i)
            else:
                dropped.append([pid, files[i].name, files[twin].name])

        order = kept[:]
        random.Random(SEED).shuffle(order)
        n_enroll = max(1, round(len(order) * ENROLL_RATIO))
        enroll, test = sorted(order[:n_enroll]), sorted(order[n_enroll:])

        for i in enroll:
            lib_emb.append(emb[i])
            lib_rows.append([len(lib_rows), pid, files[i].name, "enroll"])
        for i in test:
            test_emb.append(emb[i])
            test_rows.append([len(test_rows), pid, files[i].name, "test"])

        print(f"{pid}: {len(files)} faces, {len(files) - len(kept)} near-duplicates dropped, "
              f"{len(enroll)} enroll / {len(test)} test")

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "embeddings.npy", np.array(lib_emb, dtype=np.float32))
    write_csv(OUT / "labels.csv", ["index", "person_id", "filename", "split"], lib_rows)
    np.save(OUT / "test_embeddings.npy", np.array(test_emb, dtype=np.float32))
    write_csv(OUT / "test_labels.csv", ["index", "person_id", "filename", "split"], test_rows)
    write_csv(OUT / "dropped_duplicates.csv", ["person_id", "dropped_file", "similar_to"], dropped)

    params = {"date": datetime.now().isoformat(timespec="seconds"),
              "image_size": fu.IMAGE_SIZE, "margin": fu.MARGIN,
              "min_confidence": fu.MIN_CONFIDENCE, "weights": fu.WEIGHTS,
              "enroll_ratio": ENROLL_RATIO, "seed": SEED, "dup_cosine": DUP_COS,
              "library_size": len(lib_rows), "test_size": len(test_rows),
              "people": len(people)}
    (OUT / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    print(f"\nLibrary: {len(lib_rows)} vectors | Test: {len(test_rows)} vectors | "
          f"Duplicates dropped: {len(dropped)}")
    print(f"Saved in {OUT}/")


if __name__ == "__main__":
    main()