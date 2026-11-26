"""
live_webcam.py
Reconnaissance en direct + vivacite en 3 tests (images, clignement, defi aleatoire
GAUCHE / DROITE / HAUT / BAS). Chaque test a son propre chrono.
Le resultat de chaque test et la decision finale sont affiches dans le terminal.
Touche 'q' pour quitter.

Lancer depuis la racine du projet :  python scripts/live_webcam.py
"""
from collections import Counter, deque

import cv2
import pandas as pd
from PIL import Image

import face_utils as fu
from recognize import load_library, identify
from liveness import LivenessChecker
from ui import draw_ui

THRESHOLD = 0.34        # seuil de distance cosinus (sera recalibre avec FAR/FRR)
RECOG_EVERY = 5         # reconnaissance toutes les 5 images (MTCNN + FaceNet sont lourds)
MIN_VOTE_SHARE = 0.70   # part minimale de votes pour la meme identite
DEBUG = True            # affiche EAR / yaw / pitch a l'ecran

ids, centroids = load_library()
names = pd.read_csv("embeddings/labels.csv").drop_duplicates("person_id") \
          .set_index("person_id")["name"].to_dict()
mtcnn = fu.get_mtcnn()
model = fu.get_facenet()
live = LivenessChecker()


def decide(report, votes):
    """Combine vivacite + identite (vote sur toute la tentative).
    Retourne (texte_decision, accepte, identite)."""
    if report["verdict"] != "VIVANT":
        return "REFUSE : usurpation suspectee", False, "-"
    if not votes:
        return "REFUSE : identite non determinee", False, "-"
    pids = [p for p, _ in votes]
    pid, n = Counter(pids).most_common(1)[0]
    share = n / len(pids)
    dist = sum(d for p, d in votes if p == pid) / n
    if share < MIN_VOTE_SHARE:
        return "REFUSE : identite instable", False, f"ambigu (vote {share:.0%})"
    if pid == "inconnu":
        return "REFUSE : personne inconnue", False, f"inconnu (dist {dist:.2f})"
    who = names.get(pid, pid)
    return f"PRESENT : {who}", True, f"{who} (dist {dist:.2f}, vote {share:.0%})"


def print_report(report, decision, identity):
    print("\n" + "=" * 72)
    print(f" VERIFICATION D'ASSIDUITE - {report['time']}")
    print("=" * 72)
    total = 0.0
    for i, (name, status, detail, dur) in enumerate(report["stages"], 1):
        total += dur
        t = f"{dur:4.1f} s" if status in ("PASS", "FAIL") else "   -  "
        print(f" Test {i} - {name:<21}: {status:<9} [{t}] {detail}")
    print("-" * 72)
    print(f" Temps total des tests : {total:.1f} s")
    print(f" Vivacite globale      : {report['verdict']}")
    print(f" Identite              : {identity}")
    print(f" DECISION FINALE       : {decision}")
    print("=" * 72 + "\n")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam introuvable")

frame_i = 0
votes = deque(maxlen=40)        # (identite, distance) collectes pendant la tentative
decision = None                 # (texte, accepte) affiche apres le verdict
last_attempt = live.attempt_id

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_i += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # image NON retournee

    face_present, msg = live.update(rgb)

    # nouvelle tentative (visage perdu, fin de validite...) : on repart de zero
    if live.attempt_id != last_attempt:
        last_attempt = live.attempt_id
        votes.clear()
        decision = None

    # reconnaissance uniquement de face, pendant la tentative
    if face_present and live.verdict is None and live.frontal \
            and (frame_i % RECOG_EVERY == 0 or not votes):
        crop, conf = fu.detect_and_crop(mtcnn, Image.fromarray(rgb))
        if crop is not None and conf >= fu.MIN_CONFIDENCE:
            emb = fu.embed(model, [crop])[0]
            votes.append(identify(emb, ids, centroids, THRESHOLD))

    # fin de tentative : rapport dans le terminal + decision finale
    report = live.take_report()
    if report is not None:
        text, accepted, identity = decide(report, votes)
        decision = (text, accepted)
        print_report(report, text, identity)

    # ---- affichage ----
    display = cv2.flip(frame, 1)                      # miroir, comme un selfie
    draw_ui(display, live, msg, decision, DEBUG)
    cv2.imshow("Controle d'assiduite", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()