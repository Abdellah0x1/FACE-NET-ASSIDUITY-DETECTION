"""
liveness.py
Detection de vivacite (anti-usurpation) en 3 tests SEQUENTIELS.
Chaque test a son propre temps limite et sa duree est mesuree.

  Test 1 - Comparaison d'images : changements entre images successives,
           apres alignement du visage (pixels + deformation de la forme).
  Test 2 - Clignement (EAR)     : yeux ouverts -> fermes -> rouverts.
           Le test ne demarre que si les yeux sont d'abord OUVERTS.
  Test 3 - Defi aleatoire       : GAUCHE, DROITE, HAUT ou BAS (tirage au hasard).
           Mauvaise direction (l'opposee) = echec.

Verdict global = VIVANT seulement si les 3 tests sont reussis.
Chaque tentative est ecrite dans reports/liveness_log.csv.

Dependance : pip install "mediapipe==0.10.14"   (ancienne API mp.solutions)
Messages sans accents (cv2.putText ne les gere pas).
"""
import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

if not hasattr(mp, "solutions"):
    raise SystemExit('MediaPipe trop recent : pip install "mediapipe==0.10.14"')

# ======================= PARAMETRES (a calibrer) =======================
CONTINUE_AFTER_FAIL = False   # True = fait les 3 tests meme apres un echec (pour les mesures)
STAGE_PAUSE = 1.5             # pause (s) entre deux tests : le resultat reste affiche

# --- Test 1 : comparaison d'images (temps propre : IMG_TIMEOUT) ---
IMG_TIMEOUT = 10.0
IMG_FRAMES = 20          # nombre d'images analysees
IMG_MIN_SECONDS = 1.5    # duree minimale d'observation
PIXEL_MIN = 1.0          # variation moyenne des pixels (visage aligne)
SHAPE_MIN = 0.006        # deformation moyenne de la forme du visage

# --- Test 2 : clignement (temps propre : BLINK_TIMEOUT) ---
BLINK_TIMEOUT = 12.0
EAR_MIN_OPEN = 0.18      # un oeil "ouvert" a au moins cet EAR
OPEN_FRAMES = 8          # images consecutives yeux ouverts avant de commencer
BLINK_RATIO = 0.75       # ferme si EAR < 75 % de l'EAR ouvert personnel
REOPEN_RATIO = 0.88      # rouvert si EAR > 88 % de l'EAR ouvert personnel
MAX_BLINK_SECONDS = 0.8  # au-dela, ce n'est plus un clignement

# --- Test 3 : defi aleatoire (temps propres : CENTER_TIMEOUT puis CHALLENGE_TIMEOUT) ---
CENTER_TIMEOUT = 10.0    # temps pour se placer de face
CHALLENGE_TIMEOUT = 8.0  # temps pour executer la consigne
CENTER_FRAMES = 8        # images de face (stables) avant la consigne
TURN_FRAMES = 3          # images consecutives dans la bonne direction
FRONTAL_MARGIN = 0.10    # |yaw - 0.5| < marge : visage de face
TURN_YAW_DELTA = 0.15    # ecart de yaw pour valider GAUCHE / DROITE
PITCH_DELTA = 0.05       # ecart de pitch pour valider HAUT / BAS
WRONG_FACTOR = 1.3       # direction opposee depassee de 30 % = mauvaise reponse
YAW_INVERT = False       # True si gauche/droite est inverse
PITCH_INVERT = False     # True si haut/bas est inverse

VALID_SECONDS = 10.0     # duree de validite apres un verdict VIVANT
FAIL_COOLDOWN = 3.0      # pause apres un rejet avant de recommencer
# =======================================================================

STAGES = ["Comparaison d'images", "Clignement (EAR)", "Defi aleatoire"]
DIRECTIONS = ["left", "right", "up", "down"]
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
CONSIGNE = {
    "left": ("GAUCHE", "Test 3 : tournez la tete a GAUCHE  <<<"),
    "right": ("DROITE", "Test 3 : tournez la tete a DROITE  >>>"),
    "up": ("HAUT", "Test 3 : levez la tete  ^ HAUT ^"),
    "down": ("BAS", "Test 3 : baissez la tete  v BAS v"),
}

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
NOSE, CHEEK_L, CHEEK_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152
# points stables servant a aligner le visage (coins des yeux, nez, coins de la bouche)
ALIGN_IDX = [33, 263, 1, 61, 291]
ALIGN_DST = np.float32([[16, 24], [48, 24], [32, 36], [22, 50], [42, 50]])
# points utilises pour mesurer la deformation du visage
SHAPE_IDX = (RIGHT_EYE + LEFT_EYE
             + [1, 168, 61, 291, 13, 14, 152, 10, 70, 63, 105, 66, 107,
                336, 296, 334, 293, 300, 234, 454])


@dataclass
class StageResult:
    status: str = "A FAIRE"   # PASS / FAIL / NON TESTE / A FAIRE
    detail: str = ""
    duration: float = 0.0     # secondes passees sur ce test


def procrustes_distance(a, b):
    """Distance entre deux formes apres alignement (translation, echelle, rotation)."""
    a0 = a - a.mean(0)
    b0 = b - b.mean(0)
    a0 = a0 / (np.linalg.norm(a0) + 1e-9)
    b0 = b0 / (np.linalg.norm(b0) + 1e-9)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    aligned = a0 @ (u @ vt)
    return float(np.linalg.norm(aligned - b0))


class LivenessChecker:
    def __init__(self, log_path="reports/liveness_log.csv"):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.log_path = Path(log_path)
        self.rng = random.SystemRandom()
        self.history = []          # directions deja demandees
        self.attempt_id = 0
        self.report = None
        self.ear, self.yaw, self.pitch, self.frontal = 0.0, 0.5, 0.0, False
        self.face_box = None       # (x0, y0, x1, y1) du visage, image non retournee
        self.reset()

    # ------------------------------------------------------------ etat
    def reset(self):
        self.attempt_id += 1
        self.results = [StageResult() for _ in STAGES]
        self.verdict = None        # None / "VIVANT" / "REJETE"
        self.t_verdict = 0.0
        self.pending, self.pause_until, self.pause_msg = None, 0.0, ""
        self._start_stage(0)

    def _start_stage(self, idx):
        self.stage = idx
        self.t_stage = time.time()
        # test 1
        self.buf_crops, self.buf_shapes, self.t_collect = [], [], 0.0
        # test 2
        self.open_run, self.baseline = [], None
        self.closed_since, self.min_ear = None, 1.0
        # test 3
        self.center_run, self.turn_run = 0, 0
        self.pitch_buf, self.pitch0 = [], 0.0
        self.direction, self.t_turn = None, 0.0

    def is_alive(self):
        return self.verdict == "VIVANT"

    def take_report(self):
        rep, self.report = self.report, None
        return rep

    def time_left(self):
        """Secondes restantes pour le test en cours (None si aucun test actif)."""
        if self.verdict is not None or self.pending is not None:
            return None
        now = time.time()
        if self.stage == 0:
            limit, t0 = IMG_TIMEOUT, self.t_stage
        elif self.stage == 1:
            limit, t0 = BLINK_TIMEOUT, self.t_stage
        elif self.direction is None:
            limit, t0 = CENTER_TIMEOUT, self.t_stage
        else:
            limit, t0 = CHALLENGE_TIMEOUT, self.t_turn
        return max(0.0, limit - (now - t0))

    def time_info(self):
        """(secondes restantes, temps limite) du test en cours, ou None."""
        if self.verdict is not None or self.pending is not None:
            return None
        now = time.time()
        if self.stage == 0:
            limit, t0 = IMG_TIMEOUT, self.t_stage
        elif self.stage == 1:
            limit, t0 = BLINK_TIMEOUT, self.t_stage
        elif self.direction is None:
            limit, t0 = CENTER_TIMEOUT, self.t_stage
        else:
            limit, t0 = CHALLENGE_TIMEOUT, self.t_turn
        return max(0.0, limit - (now - t0)), limit

    def pause_progress(self):
        """Avancement (0 a 1) de la pause entre deux tests."""
        if self.pending is None:
            return 0.0
        return min(1.0, max(0.0, 1.0 - (self.pause_until - time.time()) / STAGE_PAUSE))

    def progress_text(self):
        sym = {"PASS": "OK", "FAIL": "ECHEC", "NON TESTE": "-", "A FAIRE": "..."}
        active = self.pending if self.pending is not None else self.stage
        parts = []
        for i, r in enumerate(self.results):
            s = ">>" if (self.verdict is None and i == active and r.status == "A FAIRE") \
                else sym[r.status]
            parts.append(f"T{i + 1}:{s}")
        return "  ".join(parts)

    # -------------------------------------------------------- mesures
    @staticmethod
    def _ear(pts, idx):
        p = pts[idx]
        vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return vertical / (2.0 * horizontal + 1e-9)

    @staticmethod
    def _yaw(pts):
        """~0.5 = de face. Inverse pour correspondre a l'affichage en miroir."""
        nose, left, right = pts[NOSE], pts[CHEEK_L], pts[CHEEK_R]
        raw = (nose[0] - left[0]) / (right[0] - left[0] + 1e-9)
        return raw if YAW_INVERT else 1.0 - raw

    @staticmethod
    def _pitch(pts):
        """Position verticale du nez entre le front et le menton.
        Valeur plus petite = tete levee, plus grande = tete baissee.
        On la compare a la valeur de reference prise de face (propre a chaque personne)."""
        top, nose, chin = pts[FOREHEAD][1], pts[NOSE][1], pts[CHIN][1]
        ratio = (nose - top) / (chin - top + 1e-9)
        return -ratio if PITCH_INVERT else ratio

    @staticmethod
    def _aligned_gray(rgb, pts):
        """Visage gris 64x64 aligne sur un modele fixe (supprime le mouvement rigide)."""
        src = pts[ALIGN_IDX].astype(np.float32)
        m, _ = cv2.estimateAffinePartial2D(src, ALIGN_DST, method=cv2.LMEDS)
        if m is None:
            return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        warped = cv2.warpAffine(gray, m, (64, 64))
        warped = cv2.GaussianBlur(warped, (3, 3), 0).astype(np.float32)
        return warped - warped.mean()

    # ------------------------------------------------------- resultats
    def _finish(self, ok, detail, now):
        dur = now - self.t_stage
        self.results[self.stage] = StageResult("PASS" if ok else "FAIL", detail, dur)
        if (ok or CONTINUE_AFTER_FAIL) and self.stage < len(STAGES) - 1:
            self.pending = self.stage + 1
            self.pause_until = now + STAGE_PAUSE
            self.pause_msg = (f"Test {self.stage + 1} {'reussi' if ok else 'echoue'}"
                              f" - preparez le test {self.stage + 2}")
            return self.pause_msg
        if not (ok or CONTINUE_AFTER_FAIL):
            for k in range(self.stage + 1, len(STAGES)):
                self.results[k] = StageResult("NON TESTE", "")
        self._conclude(now)
        return self.verdict_message()

    def _conclude(self, now):
        ok = all(r.status == "PASS" for r in self.results)
        self.verdict = "VIVANT" if ok else "REJETE"
        self.t_verdict = now
        self.report = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "verdict": self.verdict,
            "stages": [(STAGES[i], r.status, r.detail, r.duration)
                       for i, r in enumerate(self.results)],
        }
        self._log()

    def verdict_message(self):
        if self.verdict == "VIVANT":
            return "VIVANT - 3 tests reussis"
        for i, r in enumerate(self.results):
            if r.status == "FAIL":
                return f"REJETE - echec test {i + 1}"
        return "REJETE"

    def _log(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.log_path.exists()
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "verdict"] + [
                        c for n in STAGES for c in (f"{n} statut", f"{n} detail", f"{n} duree_s")])
                row = [self.report["time"], self.verdict]
                for _, status, detail, dur in self.report["stages"]:
                    row += [status, detail, f"{dur:.1f}"]
                w.writerow(row)
        except OSError as e:
            print(f"[liveness] journal non ecrit : {e}")

    # ----------------------------------------------------------- test 1
    def _stage_images(self, rgb, pts, now):
        if now - self.t_stage > IMG_TIMEOUT:
            return self._finish(False, f"donnees insuffisantes ({len(self.buf_crops)} images)", now)
        if not self.frontal:
            return "Test 1 : regardez la camera de face"
        g = self._aligned_gray(rgb, pts)
        if g is None:
            return "Test 1 : analyse des images..."
        if not self.buf_crops:
            self.t_collect = now
        self.buf_crops.append(g)
        self.buf_shapes.append(pts[SHAPE_IDX])
        if len(self.buf_crops) < IMG_FRAMES or now - self.t_collect < IMG_MIN_SECONDS:
            return "Test 1 : analyse des images - restez naturel"

        pix = float(np.mean([np.abs(self.buf_crops[i] - self.buf_crops[i + 1]).mean()
                             for i in range(len(self.buf_crops) - 1)]))
        ref = self.buf_shapes[0]
        shp = float(np.mean([procrustes_distance(s, ref) for s in self.buf_shapes[1:]]))
        ok = pix >= PIXEL_MIN and shp >= SHAPE_MIN
        detail = f"pixel={pix:.2f} (min {PIXEL_MIN}) ; forme={shp:.4f} (min {SHAPE_MIN})"
        return self._finish(ok, detail, now)

    # ----------------------------------------------------------- test 2
    def _stage_blink(self, now):
        if now - self.t_stage > BLINK_TIMEOUT:
            return self._finish(False, "aucun clignement valide detecte", now)

        # Phase A : reference personnelle = yeux OUVERTS pendant OPEN_FRAMES images
        if self.baseline is None:
            if self.ear >= EAR_MIN_OPEN:
                self.open_run.append(self.ear)
            else:
                self.open_run.clear()
            if len(self.open_run) >= OPEN_FRAMES:
                self.baseline = float(np.median(self.open_run))
                self.closed_since, self.min_ear = None, 1.0
            return "Test 2 : gardez les yeux OUVERTS"

        # Phase B : attendre fermeture puis reouverture
        closed_thr = self.baseline * BLINK_RATIO
        reopen_thr = self.baseline * REOPEN_RATIO
        if self.closed_since is None:
            if self.ear < closed_thr:
                self.closed_since, self.min_ear = now, self.ear
            return "Test 2 : CLIGNEZ des yeux"

        self.min_ear = min(self.min_ear, self.ear)
        if self.ear >= reopen_thr:
            dur = now - self.closed_since
            if dur <= MAX_BLINK_SECONDS:
                detail = (f"EAR ouvert={self.baseline:.2f} -> ferme={self.min_ear:.2f} ; "
                          f"clignement={dur * 1000:.0f} ms")
                return self._finish(True, detail, now)
            self.baseline, self.closed_since = None, None     # trop long : on repart
            self.open_run.clear()
            return "Yeux fermes trop longtemps - clignez brievement"
        return "Test 2 : CLIGNEZ des yeux"

    # ----------------------------------------------------------- test 3
    def _pick_direction(self):
        """Tirage au hasard parmi 4 directions, sans repeter la precedente."""
        options = [d for d in DIRECTIONS if not self.history or d != self.history[-1]]
        choice = self.rng.choice(options)
        self.history = (self.history + [choice])[-8:]
        return choice

    def _stage_challenge(self, now):
        # Phase A : se placer de face, immobile (sert de reference pour haut/bas)
        if self.direction is None:
            if self.frontal:
                self.center_run += 1
                self.pitch_buf.append(self.pitch)
            else:
                self.center_run = 0
                self.pitch_buf.clear()
            if self.center_run >= CENTER_FRAMES:
                self.pitch0 = float(np.mean(self.pitch_buf))
                self.direction = self._pick_direction()
                self.t_turn, self.turn_run = now, 0
            elif now - self.t_stage > CENTER_TIMEOUT:
                return self._finish(False, "visage jamais de face", now)
            else:
                return "Test 3 : regardez de face, immobile"

        # Phase B : executer la consigne
        d = self.direction
        yaw_dev = (self.yaw - 0.5) / TURN_YAW_DELTA
        pit_dev = (self.pitch - self.pitch0) / PITCH_DELTA
        score = {"left": -yaw_dev, "right": yaw_dev, "up": -pit_dev, "down": pit_dev}
        name = CONSIGNE[d][0]
        info = f"yaw={self.yaw:.2f} pitch={self.pitch - self.pitch0:+.3f}"

        if score[OPPOSITE[d]] >= WRONG_FACTOR:
            return self._finish(False, f"consigne {name} mais direction opposee ({info})", now)
        self.turn_run = self.turn_run + 1 if score[d] >= 1.0 else 0
        if self.turn_run >= TURN_FRAMES:
            return self._finish(True, f"consigne {name} respectee ({info})", now)
        if now - self.t_turn > CHALLENGE_TIMEOUT:
            return self._finish(False, f"consigne {name} non realisee en {CHALLENGE_TIMEOUT:.0f} s", now)
        return CONSIGNE[d][1]

    # ------------------------------------------------------ boucle
    def update(self, rgb):
        """Analyse une image RGB NON retournee. Retourne (visage_present, message)."""
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            self.reset()
            self.ear, self.yaw, self.pitch, self.frontal = 0.0, 0.5, 0.0, False
            self.face_box = None
            return False, "Aucun visage"

        h, w = rgb.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])
        self.ear = (self._ear(pts, RIGHT_EYE) + self._ear(pts, LEFT_EYE)) / 2.0
        self.yaw = self._yaw(pts)
        self.pitch = self._pitch(pts)
        self.frontal = abs(self.yaw - 0.5) < FRONTAL_MARGIN
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        self.face_box = (int(x0), int(y0), int(x1), int(y1))
        now = time.time()

        if self.verdict is not None:
            limit = VALID_SECONDS if self.verdict == "VIVANT" else FAIL_COOLDOWN
            if now - self.t_verdict > limit:
                self.reset()
                return True, "Nouvelle verification"
            return True, self.verdict_message()

        if self.pending is not None:                 # pause entre deux tests
            if now < self.pause_until:
                return True, self.pause_msg
            idx, self.pending = self.pending, None
            self._start_stage(idx)                   # le chrono du test commence ICI

        if self.stage == 0:
            msg = self._stage_images(rgb, pts, now)
        elif self.stage == 1:
            msg = self._stage_blink(now)
        else:
            msg = self._stage_challenge(now)
        return True, msg