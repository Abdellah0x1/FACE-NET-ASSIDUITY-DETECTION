"""
live_webcam.py
Reconnaissance en direct. Touche 'q' pour quitter.
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import face_utils as fu
from recognize import load_library, identify

THRESHOLD = 0.35

# Chargement (une seule fois)
ids, centroids = load_library()
names = pd.read_csv("embeddings/labels.csv").drop_duplicates("person_id") \
          .set_index("person_id")["name"].to_dict()
mtcnn = fu.get_mtcnn()
model = fu.get_facenet()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam introuvable")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # OpenCV = BGR, MTCNN attend du RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop, conf = fu.detect_and_crop(mtcnn, Image.fromarray(rgb))

    if crop is not None and conf >= fu.MIN_CONFIDENCE:
        emb = fu.embed(model, [crop])[0]
        pid, dist = identify(emb, ids, centroids, THRESHOLD)
        label = f"{names.get(pid, pid)} ({dist:.2f})"
        color = (0, 200, 0) if pid != "inconnu" else (0, 0, 255)
    else:
        label, color = "Aucun visage", (0, 165, 255)

    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Reconnaissance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()