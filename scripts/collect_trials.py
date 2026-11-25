"""
collect_trials.py
Mode COLLECTE : essais en conditions reelles (semaine 5 du cahier des charges).

Chaque essai = une tentative complete (3 tests de vivacite + reconnaissance).
Le programme enregistre pour chaque essai : la distance, le temps de
reconnaissance, le resultat de chaque test de vivacite et la decision finale.

Fichiers ecrits dans reports/ :
  live_trials.csv    true_id,distance  (lu automatiquement par calibrate_threshold.py)
  trials_detail.csv  toutes les infos de chaque essai (lu par summarize_trials.py)

Exemples (depuis la racine du projet) :
  python scripts/collect_trials.py --who person_01                 # vraie personne inscrite
  python scripts/collect_trials.py --who unknown                   # personne NON inscrite
  python scripts/collect_trials.py --attack photo_phone --target person_01
  python scripts/collect_trials.py --attack photo_print --target person_01
  python scripts/collect_trials.py --attack video --target person_01

Options : --trials 10 (nombre d'essais, 10 par defaut)  --threshold 0.30
          --all-stages (enregistre aussi les tests apres un echec : utile pour les attaques)

Touches : ESPACE = lancer l'essai | X = annuler l'essai | Q = quitter
"""
import argparse
import csv
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPORTS = Path("reports")
CALIB_FILE = REPORTS / "live_trials.csv"
DETAIL_FILE = REPORTS / "trials_detail.csv"

RECOG_EVERY = 5            # reconnaissance toutes les 5 images
MIN_VOTE_SHARE = 0.70      # part minimale de votes pour la meme identite
TRIAL_MAX_SECONDS = 90.0   # un essai trop long est annule
DONE_SECONDS = 2.5         # duree d'affichage du resultat
ATTACKS = ["photo_phone", "photo_print", "video"]

DETAIL_FIELDS = [
    "timestamp", "scenario", "true_id", "threshold", "verdict",
    "stage1_status", "stage1_s", "stage2_status", "stage2_s", "stage3_status", "stage3_s",
    "challenge", "liveness_detail",
    "n_votes", "nearest_id", "nearest_dist", "own_dist", "vote_share",
    "accepted", "decision", "recog_ms_mean", "recog_ms_max", "trial_s",
]


# ------------------------------------------------- fonctions pures (testables)
def summarize_votes(votes, true_id, ids):
    """Resume les identifications faites pendant un essai.
    votes : liste de {"nearest_id", "nearest_dist", "own_dist"}."""
    if not votes:
        return {"n_votes": 0, "nearest_id": "", "nearest_dist": "", "own_dist": "",
                "vote_share": 0.0}
    pid, n = Counter(v["nearest_id"] for v in votes).most_common(1)[0]
    near = statistics.median(v["nearest_dist"] for v in votes if v["nearest_id"] == pid)
    own = ""
    if true_id in ids:
        own = round(statistics.median(v["own_dist"] for v in votes), 4)
    return {"n_votes": len(votes), "nearest_id": pid, "nearest_dist": round(near, 4),
            "own_dist": own, "vote_share": round(n / len(votes), 3)}


def decide_trial(verdict, s, threshold):
    """Decision du systeme pour un essai. Retourne (accepte, texte)."""
    if verdict != "VIVANT":
        return False, "REFUSE : vivacite"
    if s["n_votes"] == 0:
        return False, "REFUSE : identite non determinee"
    if s["vote_share"] < MIN_VOTE_SHARE:
        return False, "REFUSE : identite instable"
    if s["nearest_dist"] >= threshold:
        return False, "REFUSE : inconnu"
    return True, f"PRESENT : {s['nearest_id']}"


def outcome(scenario, true_id, accepted, nearest_id):
    """Le resultat est-il celui qu'on ATTENDAIT ? Retourne (ok, libelle)."""
    if scenario == "genuine":
        if accepted and nearest_id == true_id:
            return True, f"OK : reconnu {true_id}"
        if accepted:
            return False, f"ERREUR : pris pour {nearest_id}"
        return False, "ECHEC : vraie personne refusee"
    if scenario == "unknown":
        if not accepted:
            return True, "OK : inconnu refuse"
        return False, f"FAUSSE ACCEPTATION ({nearest_id})"
    if not accepted:
        return True, "BLOQUE : usurpation refusee"
    return False, "ATTENTION : usurpation ACCEPTEE"


def build_row(scenario, true_id, threshold, report, s, accepted, decision,
              recog_ms, trial_s, challenge):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "scenario": scenario, "true_id": true_id, "threshold": threshold,
           "verdict": report["verdict"], "challenge": challenge,
           "accepted": int(accepted), "decision": decision,
           "recog_ms_mean": round(statistics.mean(recog_ms), 1) if recog_ms else "",
           "recog_ms_max": round(max(recog_ms), 1) if recog_ms else "",
           "trial_s": round(trial_s, 2)}
    row.update(s)
    details = []
    for i, (_, status, detail, dur) in enumerate(report["stages"], 1):
        row[f"stage{i}_status"] = status
        row[f"stage{i}_s"] = round(dur, 2)
        if detail:
            details.append(f"T{i}: {detail}")
    row["liveness_detail"] = " | ".join(details)
    return row


def append_row(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerow([row.get(k, "") for k in fields])


# ------------------------------------------------------------------ programme
def main():
    ap = argparse.ArgumentParser(description="Collecte d'essais en conditions reelles")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--who", help="person_01 ... (vraie personne inscrite) ou unknown")
    g.add_argument("--attack", choices=ATTACKS, help="type d'usurpation testee")
    ap.add_argument("--target", help="personne usurpee (obligatoire avec --attack)")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--all-stages", action="store_true",
                    help="continue les 3 tests meme apres un echec")
    args = ap.parse_args()
    if args.attack and not args.target:
        ap.error("--attack demande --target person_XX")

    # imports lourds ici (les fonctions ci-dessus restent testables sans torch)
    import cv2
    import numpy as np
    from PIL import Image
    import face_utils as fu
    import liveness as LV
    import ui
    from recognize import load_library

    if args.all_stages:
        LV.CONTINUE_AFTER_FAIL = True

    ids, centroids = load_library()
    if args.attack:
        scenario, true_id = args.attack, args.target
    elif args.who.lower() == "unknown":
        scenario, true_id = "unknown", "unknown"
    else:
        scenario, true_id = "genuine", args.who
    if true_id != "unknown" and true_id not in ids:
        raise SystemExit(f"Identifiant inconnu : {true_id}. Disponibles : {ids}")
    label = f"{scenario}" + (f" ({true_id})" if scenario != "unknown" else "")

    mtcnn = fu.get_mtcnn()
    model = fu.get_facenet()
    live = LV.LivenessChecker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam introuvable")

    print(f"\nCOLLECTE : {label} | {args.trials} essais | seuil {args.threshold}")
    print("ESPACE = lancer l'essai | X = annuler | Q = quitter\n")

    state = "WAIT"                 # WAIT -> RUN -> DONE -> WAIT ...
    done_count, frame_i = 0, 0
    votes, recog_ms = [], []
    t_trial = t_done = 0.0
    last_attempt = live.attempt_id
    result_view = ("", True)
    last_msg = ""
    h = w = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = cv2.flip(frame, 1)
        h, w = display.shape[:2]

        if state == "RUN":
            face_present, last_msg = live.update(rgb)
            if live.attempt_id != last_attempt:          # visage perdu : on repart
                last_attempt = live.attempt_id
                votes.clear()
                recog_ms.clear()

            if face_present and live.verdict is None and live.frontal \
                    and (frame_i % RECOG_EVERY == 0 or not votes):
                t0 = time.perf_counter()
                crop, conf = fu.detect_and_crop(mtcnn, Image.fromarray(rgb))
                if crop is not None and conf >= fu.MIN_CONFIDENCE:
                    emb = fu.embed(model, [crop])[0]
                    emb = emb / np.linalg.norm(emb)
                    d = 1.0 - centroids @ emb
                    k = int(np.argmin(d))
                    recog_ms.append((time.perf_counter() - t0) * 1000.0)
                    votes.append({
                        "nearest_id": ids[k], "nearest_dist": float(d[k]),
                        "own_dist": float(d[ids.index(true_id)]) if true_id in ids else None})

            report = live.take_report()
            if report is not None:
                s = summarize_votes(votes, true_id, ids)
                accepted, decision = decide_trial(report["verdict"], s, args.threshold)
                good, text = outcome(scenario, true_id, accepted, s["nearest_id"])
                challenge = live.history[-1] if live.history else ""
                row = build_row(scenario, true_id, args.threshold, report, s, accepted,
                                decision, recog_ms, time.time() - t_trial, challenge)
                append_row(DETAIL_FILE, DETAIL_FIELDS, row)
                if scenario in ("genuine", "unknown") and s["n_votes"] >= 2:
                    dist = s["own_dist"] if scenario == "genuine" else s["nearest_dist"]
                    append_row(CALIB_FILE, ["true_id", "distance"],
                               {"true_id": true_id, "distance": dist})
                done_count += 1
                ms = f"{row['recog_ms_mean']} ms" if recog_ms else "n/a"
                print(f"[{done_count}/{args.trials}] {label} | vivacite {report['verdict']} | "
                      f"dist {s['nearest_dist'] if s['n_votes'] else 'n/a'} "
                      f"(vote {s['vote_share']:.0%}) | reco {ms} | {decision} -> {text}")
                result_view = (text, good)
                state, t_done = "DONE", time.time()
            elif time.time() - t_trial > TRIAL_MAX_SECONDS:
                print("Essai trop long : annule (non enregistre).")
                state = "WAIT"

            if state == "RUN":
                ui.draw_ui(display, live, last_msg, None, True)
                ui.text(display, f"Essai {done_count + 1}/{args.trials} - {label}",
                        8, h - 66, 0.55, ui.WHITE, 1)

        if state == "DONE":
            ui.draw_ui(display, live, last_msg, result_view, False)
            if time.time() - t_done > DONE_SECONDS:
                state = "WAIT"
                if done_count >= args.trials:
                    break

        if state == "WAIT":
            ui.panel(display, 0, 0, w, 70, ui.DARK, 0.75)
            ui.text(display, f"COLLECTE : {label}", w // 2, 28, 0.7, ui.YELLOW, 2, center=True)
            ui.text(display, f"Essai {done_count + 1}/{args.trials} - placez-vous, puis ESPACE",
                    w // 2, 56, 0.6, ui.WHITE, 2, center=True)
            ui.panel(display, 0, h - 50, w, h, ui.DARK, 0.75)
            ui.text(display, "ESPACE = lancer   X = annuler   Q = quitter",
                    w // 2, h - 18, 0.6, ui.WHITE, 1, center=True)

        cv2.imshow("Collecte d'essais", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and state == "WAIT":
            live.reset()
            votes.clear()
            recog_ms.clear()
            last_attempt = live.attempt_id
            t_trial = time.time()
            state = "RUN"
        if key == ord("x") and state == "RUN":
            print("Essai annule (non enregistre).")
            state = "WAIT"

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{done_count} essai(s) enregistres dans {DETAIL_FILE}")
    print("Ensuite : python scripts/summarize_trials.py  puis  python scripts/calibrate_threshold.py")


if __name__ == "__main__":
    main()