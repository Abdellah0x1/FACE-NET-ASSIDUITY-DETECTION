"""
check_embeddings.py
Data sanity check on the enrollment library (not a model evaluation).
 - same-person distances should be clearly smaller than different-person distances
 - flags photos that look like a wrong crop or a wrong label

Run from the project root:   python scripts/check_embeddings.py
"""
from pathlib import Path
import csv
import numpy as np

EMB_DIR = Path("embeddings")
REPORT = Path("reports/embedding_check.csv")


def main():
    emb = np.load(EMB_DIR / "embeddings.npy")
    with open(EMB_DIR / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    people = np.array([r["person_id"] for r in rows])
    files = [r["filename"] for r in rows]

    D = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    same = people[:, None] == people[None, :]
    intra = D[same & np.isfinite(D)]
    inter = D[~same]
    if intra.size == 0 or inter.size == 0:
        raise SystemExit("Need at least 2 people with 2+ photos each.")

    print(f"Same person      : mean {intra.mean():.2f} | max {intra.max():.2f}")
    print(f"Different people : mean {inter.mean():.2f} | min {inter.min():.2f}")
    print("Good sign: same-person mean well below different-people mean.\n")

    mean_own = np.array([D[i][same[i] & np.isfinite(D[i])].mean()
                         if (same[i] & np.isfinite(D[i])).any() else np.nan
                         for i in range(len(emb))])
    out = []
    for pid in sorted(set(people)):
        idx = np.where(people == pid)[0]
        vals = mean_own[idx]
        limit = np.nanmean(vals) + 2 * np.nanstd(vals)
        n_flag = 0
        for i in idx:
            nn = int(np.argmin(D[i]))
            flags = []
            if people[nn] != pid:
                flags.append(f"nearest_neighbor_is_{people[nn]}")
            if mean_own[i] > limit:
                flags.append("far_from_own_photos")
            n_flag += bool(flags)
            out.append([pid, files[i], round(float(mean_own[i]), 3),
                        round(float(D[i][~same[i]].min()), 3),
                        ";".join(flags) if flags else "ok"])
        print(f"{pid}: {len(idx)} photos | mean own distance {np.nanmean(vals):.2f} | "
              f"{n_flag} flagged")

    REPORT.parent.mkdir(exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "filename", "mean_dist_own_photos",
                    "nearest_other_person_dist", "flags"])
        w.writerows(out)
    print(f"\nReport saved to {REPORT}. Look at the flagged photos in dataset/processed/.")


if __name__ == "__main__":
    main()