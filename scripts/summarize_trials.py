"""
summarize_trials.py
Tableau final des metriques a partir des essais collectes (reports/trials_detail.csv).

Produit : precision, FRR, FAR, temps de reconnaissance, taux de blocage des
usurpations, matrice de confusion reelle, comparaison avec les objectifs du
cahier des charges. Ecrit reports/final_metrics.md (a coller dans le rapport).

Lancer depuis la racine du projet :
    python scripts/summarize_trials.py
    python scripts/summarize_trials.py --threshold 0.30
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS = Path("reports")
DETAIL = REPORTS / "trials_detail.csv"
MIN_VOTE_SHARE = 0.70
ATTACKS = ["photo_phone", "photo_print", "video"]
STATIC = ["photo_phone", "photo_print"]   # critere du cahier des charges : photos statiques
SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def rate(mask):
    return float(mask.mean()) if len(mask) else float("nan")


def load():
    if not DETAIL.exists():
        raise SystemExit(f"{DETAIL} introuvable : lancez d'abord collect_trials.py")
    df = pd.read_csv(DETAIL)
    for c in ["n_votes", "nearest_dist", "own_dist", "vote_share", "recog_ms_mean",
              "recog_ms_max", "trial_s", "stage1_s", "stage2_s", "stage3_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["nearest_id", "true_id", "verdict", "scenario"]:
        df[c] = df[c].fillna("").astype(str)
    return df


def identity_ok(df, thr):
    """Identite stable et assez proche (SANS tenir compte de la vivacite)."""
    return (df["n_votes"] > 0) & (df["vote_share"] >= MIN_VOTE_SHARE) & (df["nearest_dist"] < thr)


def accepted_at(df, thr):
    return (df["verdict"] == "VIVANT") & identity_ok(df, thr)


def first_failed(row):
    for i in (1, 2, 3):
        if row[f"stage{i}_status"] == "FAIL":
            return f"test {i}"
    return "aucun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.30)
    thr = ap.parse_args().threshold

    df = load()
    out = []

    def P(line=""):
        print(line)
        out.append(line)

    g = df[df.scenario == "genuine"]
    u = df[df.scenario == "unknown"]
    atk = df[df.scenario.isin(ATTACKS)]

    P(f"# Resultats des essais en conditions reelles (seuil = {thr:.2f})")
    P()
    P(f"Essais : {len(df)} au total | {len(g)} vraies personnes | {len(u)} inconnus | "
      f"{len(atk)} tentatives d'usurpation")
    P()

    # ---------------- vraies personnes
    acc_g = accepted_at(g, thr)
    correct = acc_g & (g.nearest_id == g.true_id)
    misid = acc_g & (g.nearest_id != g.true_id)
    live_fail = g.verdict != "VIVANT"
    reco_fail = ~acc_g & ~live_fail
    precision = rate(correct)
    frr = 1 - precision if len(g) else float("nan")
    P("## Vraies personnes inscrites")
    P(f"- Reconnues correctement : {int(correct.sum())}/{len(g)}  -> precision = {pct(precision)}")
    P(f"- FRR (vraie personne non reconnue) = {pct(frr)}")
    P(f"    - refusees par la vivacite   : {int(live_fail.sum())}")
    P(f"    - refusees par la reconnaissance (distance/instabilite) : {int(reco_fail.sum())}")
    P(f"    - prises pour quelqu'un d'autre : {int(misid.sum())}")
    if len(g):
        P()
        P("| Personne | Essais | Reconnus | FRR |")
        P("|---|---|---|---|")
        for pid, grp in g.groupby("true_id"):
            c = int((accepted_at(grp, thr) & (grp.nearest_id == grp.true_id)).sum())
            P(f"| {pid} | {len(grp)} | {c} | {pct(1 - c / len(grp))} |")
    P()

    # ---------------- inconnus
    acc_u = accepted_at(u, thr)
    far_u = rate(acc_u)
    P("## Personnes inconnues (non inscrites)")
    P(f"- Acceptees a tort : {int(acc_u.sum())}/{len(u)}  -> FAR = {pct(far_u)}")
    P()
    n_bad = int(acc_u.sum()) + int(misid.sum())
    n_all = len(u) + len(g)
    far = n_bad / n_all if n_all else float("nan")
    P(f"FAR global (A reconnu comme B, ou inconnu accepte) = {pct(far)}")
    P()

    # ---------------- anti-usurpation
    P("## Anti-usurpation")
    spoof_rate = float("nan")
    if len(atk):
        P("| Attaque | Essais | Acceptees | Bloquees | Taux de blocage | Bloquee au |")
        P("|---|---|---|---|---|---|")
        for name in ATTACKS:
            a = atk[atk.scenario == name]
            if a.empty:
                continue
            acc = accepted_at(a, thr)
            where = a[~acc].apply(first_failed, axis=1).value_counts().to_dict()
            P(f"| {name} | {len(a)} | {int(acc.sum())} | {int((~acc).sum())} | "
              f"{pct(1 - rate(acc))} | {where if where else '-'} |")
        stat = atk[atk.scenario.isin(STATIC)]
        if len(stat):
            spoof_rate = 1 - rate(accepted_at(stat, thr))
        P()
        P("Critere du cahier des charges = photos statiques (telephone + imprimee) bloquees a 100 %.")
        vid = atk[atk.scenario == "video"]
        if len(vid):
            P(f"La video est une limite connue de la methode (clignement + defi) : "
              f"{pct(1 - rate(accepted_at(vid, thr)))} bloquees. A discuter dans le rapport.")
    else:
        P("Aucun essai d'usurpation enregistre.")
    P()

    # ---------------- temps
    t = df[df.recog_ms_mean.notna()]
    P("## Temps")
    mean_ms = float(t.recog_ms_mean.mean()) if len(t) else float("nan")
    max_ms = float(t.recog_ms_max.max()) if len(t) else float("nan")
    if len(t):
        P(f"- Temps de reconnaissance par image (MTCNN + FaceNet + comparaison) : "
          f"moyenne {mean_ms:.0f} ms | maximum {max_ms:.0f} ms")
    live_t = g[["stage1_s", "stage2_s", "stage3_s"]].mean() if len(g) else None
    if live_t is not None:
        P(f"- Duree moyenne des tests de vivacite (vraies personnes) : "
          f"T1 {live_t.stage1_s:.1f} s | T2 {live_t.stage2_s:.1f} s | T3 {live_t.stage3_s:.1f} s")
        P(f"- Duree moyenne d'un essai complet : {g.trial_s.mean():.1f} s")
    P()

    # ---------------- objectifs
    def ok(cond):
        return "OK" if cond is True else ("NON" if cond is False else "n/a")

    def lt(v, lim):
        return None if np.isnan(v) else bool(v < lim)

    P("## Objectifs du cahier des charges")
    P("| Critere | Objectif | Mesure | Resultat |")
    P("|---|---|---|---|")
    P(f"| Precision | >= 90 % | {pct(precision)} | {ok(None if np.isnan(precision) else precision >= 0.90)} |")
    P(f"| FAR | < 5 % | {pct(far)} | {ok(lt(far, 0.05))} |")
    P(f"| FRR | < 5 % | {pct(frr)} | {ok(lt(frr, 0.05))} |")
    P(f"| Temps de reconnaissance | < 2 s | {'n/a' if np.isnan(mean_ms) else f'{mean_ms / 1000:.2f} s'} | "
      f"{ok(lt(mean_ms, 2000.0))} |")
    P(f"| Anti-usurpation (photos statiques) | 100 % bloquees | {pct(spoof_rate)} | "
      f"{ok(None if np.isnan(spoof_rate) else spoof_rate >= 1.0)} |")
    P()

    # ---------------- choix du seuil sur donnees reelles
    P("## Effet du seuil sur les essais reels (identite seule, sans vivacite)")
    P("| Seuil | FAR (inconnus acceptes) | FRR (vrais non reconnus) |")
    P("|---|---|---|")
    for s_ in SWEEP:
        far_s = rate(identity_ok(u, s_)) if len(u) else float("nan")
        good = identity_ok(g, s_) & (g.nearest_id == g.true_id)
        frr_s = 1 - rate(good) if len(g) else float("nan")
        P(f"| {s_:.2f} | {pct(far_s)} | {pct(frr_s)} |")
    P()

    # ---------------- matrice de confusion
    real = df[df.scenario.isin(["genuine", "unknown"])]
    if len(real):
        person_ids = sorted(set(g.true_id) | set(real.nearest_id[real.nearest_id != ""]))
        rows = person_ids + ["unknown"]
        cols = person_ids + ["inconnu", "refuse (vivacite)"]
        M = pd.DataFrame(0, index=rows, columns=cols)
        acc_all = accepted_at(real, thr)
        for (_, r), a in zip(real.iterrows(), acc_all):
            if r.verdict != "VIVANT":
                pred = "refuse (vivacite)"
            elif a:
                pred = r.nearest_id if r.nearest_id in person_ids else "inconnu"
            else:
                pred = "inconnu"
            row_name = r.true_id if r.true_id in rows else "unknown"
            M.loc[row_name, pred] += 1
        M.to_csv(REPORTS / "confusion_matrix_live.csv")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(M.values, cmap="Blues")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = M.values[i, j]
                ax.text(j, i, v, ha="center", va="center",
                        color="white" if v > M.values.max() / 2 else "black")
        ax.set_xlabel("Resultat du systeme")
        ax.set_ylabel("Identite reelle")
        ax.set_title(f"Matrice de confusion - essais reels (seuil {thr:.2f})")
        fig.tight_layout()
        fig.savefig(REPORTS / "confusion_matrix_live.png", dpi=150)
        plt.close(fig)
        P("Matrice de confusion : reports/confusion_matrix_live.csv / .png")

    (REPORTS / "final_metrics.md").write_text("\n".join(out), encoding="utf-8")
    print("\nResume ecrit dans reports/final_metrics.md")


if __name__ == "__main__":
    main()