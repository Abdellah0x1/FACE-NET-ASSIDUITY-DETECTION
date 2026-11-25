"""
engine.py
Interface simple entre le MODELE et l'INTERFACE (Streamlit / Tkinter).

L'ingenieur interface n'a besoin que de ceci :

    from engine import AttendanceEngine
    engine = AttendanceEngine()                  # charge les modeles (1 seule fois)

    while True:                                  # boucle webcam
        ok, frame = cap.read()                   # frame BGR (OpenCV)
        r = engine.process_frame(frame)          # <-- tout le travail est ici
        image = engine.annotate(frame)           # image annotee (BGR) a afficher
        if r["new_report"]:                      # vrai UNE SEULE FOIS a la fin d'une verification
            if r["accepted"]:
                enregistrer_presence(r["person_id"], r["name"])   # export CSV
            else:
                proposer_confirmation_manuelle()                  # mecanisme de secours
            # engine.reset()  # pour passer a l'etudiant suivant tout de suite

Dictionnaire retourne par process_frame() :
    face_present  bool    un visage est visible
    message       str     consigne a afficher ("CLIGNEZ des yeux", ...)
    steps         list    3 etapes [{"name","status"}]  status = PASS/FAIL/NON TESTE/A FAIRE
    time_left     float   secondes restantes pour le test en cours (None si aucun)
    finished      bool    la verification est terminee
    new_report    bool    True uniquement a l'image ou la verification se termine
    liveness      str     None / "VIVANT" / "REJETE"
    accepted      bool    None tant que ce n'est pas fini ; True = PRESENT
    person_id     str     identifiant (person_XX) ou None
    name          str     nom affiche ou None
    distance      float   distance cosinus moyenne a la personne reconnue
    decision_text str     "PRESENT : ..." / "REFUSE : ..."
    needs_manual  bool    refuse -> l'enseignant peut confirmer a la main
    report        dict    detail des 3 tests (uniquement quand new_report est True)

Lancer ce fichier directement ouvre une petite demo :  python scripts/engine.py
"""
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import face_utils as fu
from recognize import load_library, identify
from liveness import LivenessChecker
import ui

THRESHOLD = 0.34         # seuil de distance cosinus (calibre : voir reports/)
RECOG_EVERY = 5          # reconnaissance toutes les 5 images
MIN_VOTE_SHARE = 0.70    # part minimale de votes pour la meme identite


class AttendanceEngine:
    def __init__(self, threshold=THRESHOLD, recog_every=RECOG_EVERY, warmup=True):
        self.threshold = threshold
        self.recog_every = recog_every
        self.ids, self.centroids = load_library()
        labels = pd.read_csv("embeddings/labels.csv").drop_duplicates("person_id")
        self.names = labels.set_index("person_id")["name"].to_dict()
        self.mtcnn = fu.get_mtcnn()
        self.model = fu.get_facenet()
        self.live = LivenessChecker()
        self.votes = deque(maxlen=40)
        self.decision = None
        self._msg = ""
        self._frame_i = 0
        self._last_attempt = self.live.attempt_id
        if warmup:
            self._warmup()

    # ------------------------------------------------------------ interne
    def _warmup(self):
        """Premier passage a vide : evite un premier resultat lent."""
        fu.detect_and_crop(self.mtcnn, Image.new("RGB", (640, 480)))
        fu.embed(self.model, [np.zeros((160, 160, 3), np.uint8)])

    def _decide(self, report):
        liveness = report["verdict"]
        out = {"accepted": False, "person_id": None, "name": None, "distance": None,
               "liveness": liveness}
        if liveness != "VIVANT":
            out["text"] = "REFUSE : usurpation suspectee"
            return out
        if not self.votes:
            out["text"] = "REFUSE : identite non determinee"
            return out
        pids = [p for p, _ in self.votes]
        pid, n = Counter(pids).most_common(1)[0]
        share = n / len(pids)
        dist = float(np.mean([d for p, d in self.votes if p == pid]))
        if share < MIN_VOTE_SHARE:
            out["text"] = "REFUSE : identite instable"
        elif pid == "inconnu":
            out["text"] = "REFUSE : personne inconnue"
        else:
            out.update(accepted=True, person_id=pid, name=self.names.get(pid, pid),
                       distance=dist, text=f"PRESENT : {self.names.get(pid, pid)}")
        return out

    # ------------------------------------------------------------ public
    def reset(self):
        """Prepare la verification suivante (nouvel etudiant)."""
        self.live.reset()
        self.votes.clear()
        self.decision = None
        self._last_attempt = self.live.attempt_id

    def process_frame(self, frame_bgr):
        """Traite UNE image de la webcam (BGR). A appeler a chaque image."""
        self._frame_i += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        live = self.live

        face, self._msg = live.update(rgb)
        if live.attempt_id != self._last_attempt:           # visage perdu / nouvelle tentative
            self._last_attempt = live.attempt_id
            self.votes.clear()
            self.decision = None

        if face and live.verdict is None and live.frontal \
                and (self._frame_i % self.recog_every == 0 or not self.votes):
            crop, conf = fu.detect_and_crop(self.mtcnn, Image.fromarray(rgb))
            if crop is not None and conf >= fu.MIN_CONFIDENCE:
                emb = fu.embed(self.model, [crop])[0]
                pid, dist = identify(emb, self.ids, self.centroids, self.threshold)
                self.votes.append((pid, dist))

        report, new_report = live.take_report(), False
        if report is not None:
            self.decision = self._decide(report)
            new_report = True

        d = self.decision
        finished = live.verdict is not None
        left = live.time_info()
        return {
            "face_present": face,
            "message": d["text"] if d else self._msg,
            "steps": [{"name": n, "status": r.status}
                      for n, r in zip(["Comparaison d'images", "Clignement", "Defi aleatoire"],
                                      live.results)],
            "time_left": None if left is None else left[0],
            "finished": finished,
            "new_report": new_report,
            "liveness": live.verdict,
            "accepted": d["accepted"] if d else None,
            "person_id": d["person_id"] if d else None,
            "name": d["name"] if d else None,
            "distance": d["distance"] if d else None,
            "decision_text": d["text"] if d else None,
            "needs_manual": bool(d) and not d["accepted"],
            "report": report,
        }

    def annotate(self, frame_bgr, debug=False):
        """Image en miroir avec l'interface coloree (etapes, chrono, fleche, decision)."""
        display = cv2.flip(frame_bgr, 1)
        d = self.decision
        ui.draw_ui(display, self.live, d["text"] if d else self._msg,
                   (d["text"], d["accepted"]) if d else None, debug)
        return display


# ------------------------------------------------------------ demo
if __name__ == "__main__":
    engine = AttendanceEngine()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam introuvable")
    present = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = engine.process_frame(frame)
        if r["new_report"]:
            print(r["decision_text"], "| manuel :", r["needs_manual"])
            if r["accepted"]:
                present.append(r["person_id"])
        cv2.imshow("Demo engine", engine.annotate(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Presents :", present)