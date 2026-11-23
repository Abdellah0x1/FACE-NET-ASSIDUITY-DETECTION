"""
recognize.py
Module de reconnaissance (ingénieur modèle).
Bibliothèque de référence + identification par distance cosinus.
"""
from pathlib import Path
import numpy as np
import pandas as pd

EMB_DIR = Path("embeddings")
DEFAULT_THRESHOLD = 0.5   # à ajuster à l'étape 8 (semaine 5)


def build_library(embeddings, person_ids):
    """Un embedding moyen normalisé par personne.
    Retourne (liste des ids, matrice (n_personnes, 512))."""
    person_ids = np.asarray(person_ids)
    ids = sorted(set(person_ids))
    centroids = []
    for pid in ids:
        mean = embeddings[person_ids == pid].mean(axis=0)
        centroids.append(mean / np.linalg.norm(mean))   # renormalisation
    return ids, np.array(centroids, dtype=np.float32)


def load_library():
    """Charge embeddings.npy + labels.csv et construit la bibliothèque."""
    X = np.load(EMB_DIR / "embeddings.npy")
    y = pd.read_csv(EMB_DIR / "labels.csv")
    return build_library(X, y["person_id"].values)


def identify(embedding, ids, centroids, threshold=DEFAULT_THRESHOLD):
    """Compare un embedding (512,) aux cartes d'identité.
    Retourne (person_id ou 'inconnu', distance au plus proche)."""
    embedding = embedding / np.linalg.norm(embedding)
    distances = 1.0 - centroids @ embedding        # une distance par personne
    best = int(np.argmin(distances))
    if distances[best] < threshold:
        return ids[best], float(distances[best])
    return "inconnu", float(distances[best])