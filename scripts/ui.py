"""
ui.py
Affichage de la camera : etapes colorees, chrono, cadre du visage, fleche de
direction, ecran de transition entre les tests et decision finale.
Textes sans accents (cv2.putText ne les gere pas).
"""
import time

import cv2
import numpy as np

from liveness import CONSIGNE

# Couleurs (BGR)
GREEN = (60, 200, 60)
RED = (50, 50, 230)
YELLOW = (0, 200, 255)
GRAY = (90, 90, 90)
DARK = (35, 35, 35)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

NAMES = ["IMAGES", "CLIGNEMENT", "DEFI"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------ outils
def text(img, s, x, y, scale=0.6, color=WHITE, thick=1, center=False):
    """Texte avec contour noir (lisible sur n'importe quel fond)."""
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    if center:
        x = x - tw // 2
    cv2.putText(img, s, (x, y), FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def fit_scale(s, max_width, scale, thick):
    """Reduit l'echelle jusqu'a ce que le texte tienne dans max_width."""
    while scale > 0.4:
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
        if tw <= max_width:
            break
        scale -= 0.05
    return scale


def panel(img, x0, y0, x1, y1, color=BLACK, alpha=0.55):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0, roi)


def stage_state(live, i):
    r = live.results[i]
    if r.status == "PASS":
        return "pass"
    if r.status == "FAIL":
        return "fail"
    if r.status == "NON TESTE":
        return "skip"
    active = live.pending if live.pending is not None else live.stage
    return "active" if (live.verdict is None and i == active) else "todo"


# ------------------------------------------------------------ blocs
def draw_steps(img, live):
    """Barre du haut : 3 cases colorees (a venir / en cours / reussi / echec)."""
    h, w = img.shape[:2]
    panel(img, 0, 0, w, 66, DARK, 0.75)
    blink = int(time.time() * 4) % 2 == 0
    bw = w // 3
    fill = {"pass": GREEN, "fail": RED, "active": YELLOW, "todo": GRAY, "skip": DARK}
    label = {"pass": "REUSSI", "fail": "ECHEC", "active": "EN COURS",
             "todo": "A VENIR", "skip": "NON TESTE"}
    for i in range(3):
        st = stage_state(live, i)
        x0, x1 = i * bw + 5, (i + 1) * bw - 5
        cv2.rectangle(img, (x0, 6), (x1, 60), fill[st], -1)
        fg = BLACK if st == "active" else WHITE
        r = live.results[i]
        sub = label[st]
        if st in ("pass", "fail") and r.duration:
            sub += f"  {r.duration:.1f}s"
        cx = (x0 + x1) // 2
        text(img, f"{i + 1}. {NAMES[i]}", cx, 28, 0.6, fg, 2, center=True)
        text(img, sub, cx, 50, 0.5, fg, 1, center=True)
        # transition : la case qui vient de finir et la suivante clignotent
        if live.pending is not None and live.verdict is None:
            if i == live.pending and blink:
                cv2.rectangle(img, (x0, 6), (x1, 60), WHITE, 4)
            if i == live.stage:
                cv2.rectangle(img, (x0, 6), (x1, 60), WHITE, 2)


def draw_timer(img, live):
    """Barre de temps du test en cours (verte -> orange -> rouge)."""
    info = live.time_info()
    if info is None:
        return
    left, limit = info
    h, w = img.shape[:2]
    frac = left / limit if limit else 0
    color = GREEN if frac > 0.5 else (YELLOW if frac > 0.25 else RED)
    cv2.rectangle(img, (0, 68), (w, 84), DARK, -1)
    cv2.rectangle(img, (0, 68), (int(w * frac), 84), color, -1)
    text(img, f"{left:.0f} s", w - 8 - 40, 82, 0.5, WHITE, 1)


def draw_face_frame(img, live, color):
    """Cadre a coins autour du visage (coordonnees converties pour l'image en miroir)."""
    if live.face_box is None:
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = live.face_box
    x0, x1 = w - x1, w - x0                      # miroir
    pad = int(0.08 * (x1 - x0))
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    L = max(18, (x1 - x0) // 4)
    for (px, py, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * L, py), color, 4, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, 4, cv2.LINE_AA)


def draw_direction(img, live):
    """Grande fleche pulsante pour le defi (GAUCHE / DROITE / HAUT / BAS)."""
    if not (live.stage == 2 and live.direction and live.verdict is None and live.pending is None):
        return
    h, w = img.shape[:2]
    d = live.direction
    cx, cy = w // 2, (h + 84) // 2
    a = int(0.22 * w)
    pts = {"left": ((cx - a // 2 - 60, cy), (cx - a - 120, cy)),
           "right": ((cx + a // 2 + 60, cy), (cx + a + 120, cy)),
           "up": ((cx, cy - 20), (cx, cy - 20 - a)),
           "down": ((cx, cy + 20), (cx, cy + 20 + a))}
    start, end = pts[d]
    th = 10 + int(4 * (1 + np.sin(time.time() * 8)))
    cv2.arrowedLine(img, start, end, BLACK, th + 6, tipLength=0.45)
    cv2.arrowedLine(img, start, end, YELLOW, th, tipLength=0.45)
    word = CONSIGNE[d][0]
    ty = {"left": cy + 60, "right": cy + 60, "up": cy + 60, "down": cy - 40}[d]
    text(img, word, cx, ty, 1.2, YELLOW, 3, center=True)


def draw_transition(img, live):
    """Ecran central entre deux tests : 'TEST X REUSSI' + chargement."""
    if live.pending is None or live.verdict is not None:
        return
    h, w = img.shape[:2]
    done = live.stage
    ok = live.results[done].status == "PASS"
    color = GREEN if ok else RED
    panel(img, w // 2 - 230, h // 2 - 70, w // 2 + 230, h // 2 + 70, DARK, 0.8)
    cv2.rectangle(img, (w // 2 - 230, h // 2 - 70), (w // 2 + 230, h // 2 + 70), color, 3)
    text(img, f"TEST {done + 1} {'REUSSI' if ok else 'ECHOUE'}", w // 2, h // 2 - 18,
         1.1, color, 3, center=True)
    text(img, f"Preparez le test {live.pending + 1} : {NAMES[live.pending]}",
         w // 2, h // 2 + 20, 0.65, WHITE, 2, center=True)
    bx0, bx1, by = w // 2 - 200, w // 2 + 200, h // 2 + 45
    cv2.rectangle(img, (bx0, by), (bx1, by + 12), GRAY, -1)
    cv2.rectangle(img, (bx0, by), (bx0 + int(400 * live.pause_progress()), by + 12), YELLOW, -1)


def draw_decision(img, live, decision):
    """Ecran final : grand bandeau vert (PRESENT) ou rouge (REFUSE) + cadre."""
    if decision is None:
        return
    h, w = img.shape[:2]
    txt, accepted = decision
    color = GREEN if accepted else RED
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 12)
    panel(img, 20, h // 2 - 65, w - 20, h // 2 + 65, DARK, 0.8)
    cv2.rectangle(img, (20, h // 2 - 65), (w - 20, h // 2 + 65), color, 3)
    sc = fit_scale(txt, w - 80, 1.1, 3)
    text(img, txt, w // 2, h // 2 - 5, sc, color, 3, center=True)
    alive = "VIVANT" if live.verdict == "VIVANT" else "REJETE"
    text(img, f"Vivacite : {alive}  (3 tests)", w // 2, h // 2 + 38, 0.65, WHITE, 2, center=True)


def draw_message(img, live, msg, color, debug):
    h, w = img.shape[:2]
    panel(img, 0, h - 58, w, h, DARK, 0.75)
    sc = fit_scale(msg, w - 24, 0.8, 2)
    text(img, msg, w // 2, h - 22, sc, color, 2, center=True)
    if debug:
        text(img, f"EAR {live.ear:.2f}  yaw {live.yaw:.2f}  pitch {live.pitch:.2f}",
             8, 106, 0.5, WHITE, 1)


# ------------------------------------------------------------ principal
def draw_ui(img, live, msg, decision=None, debug=True):
    """Dessine toute l'interface sur l'image (deja en miroir)."""
    if decision is not None:
        state_color = GREEN if decision[1] else RED
    elif live.pending is not None:
        state_color = GREEN if live.results[live.stage].status == "PASS" else RED
    else:
        state_color = YELLOW

    if live.face_box is not None:
        draw_face_frame(img, live, state_color)
    draw_direction(img, live)
    draw_steps(img, live)
    draw_timer(img, live)
    draw_message(img, live, msg, state_color if decision else WHITE, debug)
    draw_transition(img, live)
    draw_decision(img, live, decision)