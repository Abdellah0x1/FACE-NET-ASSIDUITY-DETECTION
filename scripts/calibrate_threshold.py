"""
calibrate_threshold.py
Etape 5 : calibration du seuil de decision de la reconnaissance.

Mesures (cahier des charges) : FAR, FRR, precision, courbe ROC, matrice de confusion.

Comment on obtient les deux types d'echantillons :
  - GENUINES (vraie personne inscrite) :
      * les images de TEST contre la bibliotheque complete ;
      * les images d'INSCRIPTION en "leave-one-out" (le centroide est recalcule
        SANS l'image testee, pour ne pas se tester contre soi-meme).
  - INCONNUS (simules) : pour chaque personne, on retire sa carte d'identite
    de la bibliotheque et on teste ses images contre les 4 autres.
    Elle devient "inconnue" pour le systeme.

FAR = part des inconnus ACCEPTES (distance < seuil)          -> a minimiser
FRR = part des vraies personnes NON reconnues (mal identifiees ou rejetees)

Optionnel : reports/live_trials.csv (colonnes : true_id,distance) pour ajouter
vos mesures webcam. true_id = person_XX (vraie personne, distance a SA carte)
ou "unknown" (inconnu, distance a la carte la plus proche).

Lancer depuis la racine du projet :
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --threshold 0.30    (seuil impose)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recognize import build_library

EMB = Path("embeddings")
OUT = Path("reports")
THRESHOLDS = np.round(np.arange(0.05, 0.801, 0.01), 2)
TARGET_FRR = 0.05          # objectif du cahier des charges : FRR < 5 %


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load():
    X = unit(np.load(EMB / "embeddings.npy"))
    y = pd.read_csv(EMB / "labels.csv")["person_id"].values
    Xt = unit(np.load(EMB / "test_embeddings.npy"))
    yt = pd.read_csv(EMB / "test_labels.csv")["person_id"].values
    return X, y, Xt, yt


def collect_samples(X, y, Xt, yt):
    ids, C = build_library(X, y)
    idx = {p: i for i, p in enumerate(ids)}

    genuine = []     # (vrai, plus_proche, dist_plus_proche, source)
    for e, t in zip(Xt, yt):
        d = 1.0 - C @ e
        k = int(np.argmin(d))
        genuine.append((t, ids[k], float(d[k]), "test"))

    for i, (e, t) in enumerate(zip(X, y)):               # leave-one-out
        mask = (y == t)
        mask[i] = False
        if mask.sum() < 1:
            continue
        C2 = C.copy()
        C2[idx[t]] = unit(X[mask].mean(axis=0))
        d = 1.0 - C2 @ e
        k = int(np.argmin(d))
        genuine.append((t, ids[k], float(d[k]), "inscription (LOO)"))

    unknown = []     # (vrai_cache, plus_proche, dist_plus_proche)
    Xall = np.vstack([X, Xt])
    yall = np.concatenate([y, yt])
    for p in ids:
        keep = [i for i, q in enumerate(ids) if q != p]
        Ck, idk = C[keep], [ids[i] for i in keep]
        for e in Xall[yall == p]:
            d = 1.0 - Ck @ e
            k = int(np.argmin(d))
            unknown.append((p, idk[k], float(d[k])))
    return ids, C, genuine, unknown


def add_live_trials(genuine, unknown):
    f = OUT / "live_trials.csv"
    if not f.exists():
        return 0
    df = pd.read_csv(f)
    for t, d in zip(df["true_id"], df["distance"]):
        if str(t).lower() == "unknown":
            unknown.append(("?", "?", float(d)))
        else:
            genuine.append((t, t, float(d), "webcam"))
    return len(df)


def sweep(genuine, unknown):
    g_true = np.array([g[0] for g in genuine])
    g_near = np.array([g[1] for g in genuine])
    g_dist = np.array([g[2] for g in genuine])
    u_dist = np.array([u[2] for u in unknown])
    rows = []
    for thr in THRESHOLDS:
        correct = (g_near == g_true) & (g_dist < thr)
        misid = (g_near != g_true) & (g_dist < thr)
        rows.append({"seuil": thr,
                     "FAR": float(np.mean(u_dist < thr)),
                     "FRR": float(1.0 - np.mean(correct)),
                     "mauvaise_identite": float(np.mean(misid))})
    return pd.DataFrame(rows)


def roc_points(genuine, unknown):
    """Points de la courbe ROC sur des seuils de 0 a 2 (extremites incluses)."""
    g_true = np.array([g[0] for g in genuine])
    g_near = np.array([g[1] for g in genuine])
    g_dist = np.array([g[2] for g in genuine])
    u_dist = np.array([u[2] for u in unknown])
    far, tpr = [], []
    for thr in np.arange(0.0, 2.0001, 0.005):
        far.append(np.mean(u_dist < thr) * 100)
        tpr.append(np.mean((g_near == g_true) & (g_dist < thr)) * 100)
    return np.array(far), np.array(tpr)


def recommend(df):
    """Seuil au milieu de la plage ou FAR est minimal avec FRR <= objectif."""
    ok = df[df["FRR"] <= TARGET_FRR]
    if ok.empty:
        i = int(np.argmin(np.abs(df["FAR"] - df["FRR"])))
        return float(df.loc[i, "seuil"]), "EER (aucun seuil n'atteint FRR <= 5 %)"
    best = ok[ok["FAR"] == ok["FAR"].min()]["seuil"].values
    return float(np.median(best)), "FAR minimal avec FRR <= 5 %"


def confusion(ids, C, genuine_test, unknown, thr):
    """Lignes = identite reelle, colonnes = resultat du systeme.
    Les inconnus mesures a la webcam n'ont pas de 'personne la plus proche' :
    s'ils sont acceptes, ils vont dans la colonne 'accepte (?)'."""
    extra = any(u[1] == "?" and u[2] < thr for u in unknown)
    cols = ids + ["inconnu"] + (["accepte (?)"] if extra else [])
    M = np.zeros((len(ids) + 1, len(cols)), dtype=int)
    for t, near, d in [(g[0], g[1], g[2]) for g in genuine_test]:
        pred = near if d < thr else "inconnu"
        M[ids.index(t), cols.index(pred)] += 1
    for _, near, d in unknown:
        if d >= thr:
            pred = "inconnu"
        else:
            pred = near if near in ids else "accepte (?)"
        M[len(ids), cols.index(pred)] += 1
    return pd.DataFrame(M, index=ids + ["inconnu (simule)"], columns=cols)


def plot_all(df, thr, cm, genuine, unknown):
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(df["seuil"], df["FAR"] * 100, label="FAR (inconnus acceptes)", color="tab:red")
    ax[0].plot(df["seuil"], df["FRR"] * 100, label="FRR (vrais rejetes)", color="tab:blue")
    ax[0].axvline(thr, color="gray", ls="--", label=f"seuil retenu = {thr:.2f}")
    ax[0].axhline(5, color="orange", ls=":", label="objectif 5 %")
    ax[0].set_xlabel("seuil (distance cosinus)")
    ax[0].set_ylabel("%")
    ax[0].set_title("FAR et FRR selon le seuil")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    far, tpr = roc_points(genuine, unknown)
    xs, ys = far / 100, tpr / 100
    auc = float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2))
    ax[1].plot(far, tpr, color="tab:green")
    ax[1].plot([0, 100], [0, 100], color="lightgray", ls="--")
    ax[1].set_xlabel("FAR (%)")
    ax[1].set_ylabel("Taux de reconnaissance (1 - FRR) (%)")
    ax[1].set_title(f"Courbe ROC (AUC = {auc:.3f})")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 5))
    a.imshow(cm.values, cmap="Blues")
    a.set_xticks(range(cm.shape[1]))
    a.set_xticklabels(cm.columns, rotation=45, ha="right")
    a.set_yticks(range(cm.shape[0]))
    a.set_yticklabels(cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            a.text(j, i, v, ha="center", va="center",
                   color="white" if v > cm.values.max() / 2 else "black")
    a.set_xlabel("Resultat du systeme")
    a.set_ylabel("Identite reelle")
    a.set_title(f"Matrice de confusion (seuil {thr:.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None, help="seuil impose")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    X, y, Xt, yt = load()
    ids, C, genuine, unknown = collect_samples(X, y, Xt, yt)
    n_live = add_live_trials(genuine, unknown)

    df = sweep(genuine, unknown)
    df.to_csv(OUT / "threshold_sweep.csv", index=False)
    thr, why = (args.threshold, "seuil impose") if args.threshold is not None else recommend(df)

    g_d = np.array([g[2] for g in genuine if g[1] == g[0]])
    u_d = np.array([u[2] for u in unknown])
    print(f"Echantillons : {len(genuine)} vraies personnes "
          f"({sum(g[3] == 'test' for g in genuine)} test, "
          f"{sum(g[3].startswith('inscription') for g in genuine)} inscription LOO, "
          f"{sum(g[3] == 'webcam' for g in genuine)} webcam) | "
          f"{len(unknown)} inconnus simules")
    if n_live:
        print(f"  + {n_live} mesures webcam lues dans reports/live_trials.csv")
    print(f"\nDistances vraies personnes : moyenne {g_d.mean():.3f} | max {g_d.max():.3f}")
    print(f"Distances inconnus         : moyenne {u_d.mean():.3f} | min  {u_d.min():.3f}")
    gap = u_d.min() - g_d.max()
    print(f"Marge de separation (min inconnus - max vrais) : {gap:+.3f}"
          + ("   <- chevauchement : aucun seuil parfait" if gap < 0 else ""))

    print("\nSeuil   FAR      FRR      Mauvaise identite")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80]:
        r = df[df["seuil"] == t]
        if len(r):
            r = r.iloc[0]
            print(f"{t:.2f}   {r.FAR:6.1%}   {r.FRR:6.1%}   {r.mauvaise_identite:6.1%}")

    r = df.iloc[int(np.argmin(np.abs(df["seuil"] - thr)))]
    print(f"\n>>> SEUIL RETENU : {thr:.2f}   ({why})")
    print(f"    FAR = {r.FAR:.1%} | FRR = {r.FRR:.1%} | precision = {1 - r.FRR:.1%}"
          f" | mauvaise identite = {r.mauvaise_identite:.1%}")
    ok = r.FAR < 0.05 and r.FRR < 0.05
    print(f"    Objectifs FAR < 5 % et FRR < 5 % : {'ATTEINTS' if ok else 'NON ATTEINTS'}")

    genuine_test = [g for g in genuine if g[3] == "test"]
    cm = confusion(ids, C, genuine_test, unknown, thr)
    cm.to_csv(OUT / "confusion_matrix.csv")
    print("\nMatrice de confusion (lignes = identite reelle, colonnes = resultat) :")
    print(cm.to_string())

    auc = plot_all(df, thr, cm, genuine, unknown)
    print(f"\nAUC = {auc:.3f}")
    print("Fichiers : reports/threshold_sweep.csv, confusion_matrix.csv/.png, roc_curve.png")
    print("\nATTENTION : test et inscription viennent de la meme seance de collecte. "
          "Ces chiffres sont OPTIMISTES. Confirmez avec des essais webcam "
          "(reports/live_trials.csv) avant de figer le seuil.")


if __name__ == "__main__":
    main()