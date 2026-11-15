"""
check_quality.py
Checks every photo in dataset/raw/person_XX/ and writes reports/quality_report.csv.
Nothing is deleted or modified: you review the report and decide.

Run from the project root:
    pip install opencv-python numpy
    python scripts/check_quality.py
"""
from pathlib import Path
import csv
import cv2
import numpy as np

# ---------- Settings (tune after looking at your first report) ----------
RAW_DIR = Path("dataset/raw")
REPORT = Path("reports/quality_report.csv")

MIN_SIDE = 200        # px: smallest side below this -> "too_small"
BLUR_MIN = 100.0      # Laplacian variance below this -> "blurry"
DARK_MAX = 60         # mean gray level below this -> "too_dark"
BRIGHT_MIN = 200      # mean gray level above this -> "too_bright"
DUP_MAX_DIST = 1      # dHash bits that differ (out of 64) -> "duplicate"
ANALYSIS_SIZE = 800   # longest side used for blur/brightness, so phone photos
                      # and low-res photos are compared fairly
EXTS = {".jpg", ".jpeg", ".png"}
# -----------------------------------------------------------------------


def downscale(img, long_side):
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def dhash(gray):
    """64-bit perceptual hash: near-identical photos give near-identical hashes."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def analyze(path):
    img = cv2.imread(str(path))  # OpenCV applies the EXIF rotation of phone photos
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(downscale(img, ANALYSIS_SIZE), cv2.COLOR_BGR2GRAY)
    return {
        "width": w,
        "height": h,
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "hash": dhash(gray),
    }


def main():
    people = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if not people:
        raise SystemExit(f"No person folders found in {RAW_DIR}")

    rows = []
    for person_dir in people:
        photos = sorted(f for f in person_dir.iterdir()
                        if f.suffix.lower() in EXTS)
        results = []

        for photo in photos:
            info = analyze(photo)
            flags = []
            if info is None:
                rows.append([person_dir.name, photo.name, "", "", "", "", "unreadable"])
                continue
            if min(info["width"], info["height"]) < MIN_SIDE:
                flags.append("too_small")
            if info["blur"] < BLUR_MIN:
                flags.append("blurry")
            if info["brightness"] < DARK_MAX:
                flags.append("too_dark")
            if info["brightness"] > BRIGHT_MIN:
                flags.append("too_bright")
            results.append((photo.name, info, flags))

        # Duplicates inside the same person: flag the later photo of each pair
        for i, (name_i, info_i, flags_i) in enumerate(results):
            for name_j, info_j, _ in results[:i]:
                dist = int(np.count_nonzero(info_i["hash"] != info_j["hash"]))
                if dist <= DUP_MAX_DIST:
                    flags_i.append(f"duplicate_of_{name_j}")
                    break

        for name, info, flags in results:
            rows.append([person_dir.name, name, f'{info["width"]}x{info["height"]}',
                         round(info["blur"], 1), round(info["brightness"], 1),
                         "", ";".join(flags) if flags else "ok"])
        # (column 6 is left empty on purpose: use it for your manual notes)

        # Per-person summary
        ok = sum(1 for _, _, f in results if not f)
        if results:
            med_blur = np.median([r[1]["blur"] for r in results])
            med_bright = np.median([r[1]["brightness"] for r in results])
            res = {f'{r[1]["width"]}x{r[1]["height"]}' for r in results}
            print(f"{person_dir.name}: {len(photos)} photos, {ok} ok, "
                  f"{len(results) - ok} flagged | median blur {med_blur:.0f}, "
                  f"median brightness {med_bright:.0f} | resolutions: {', '.join(sorted(res))}")
        else:
            print(f"{person_dir.name}: no readable photos")

    REPORT.parent.mkdir(exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "filename", "resolution", "blur_score",
                    "brightness", "notes", "flags"])
        w.writerows(rows)

    flagged = [r for r in rows if r[6] != "ok"]
    print(f"\nTotal: {len(rows)} photos, {len(flagged)} flagged")
    for r in flagged:
        print(f"  {r[0]}/{r[1]}: {r[6]}")
    print(f"\nFull report saved to {REPORT}")


if __name__ == "__main__":
    main()