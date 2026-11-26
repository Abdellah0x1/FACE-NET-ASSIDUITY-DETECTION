"""
eval_offline.py
Teste la reconnaissance sur les embeddings de TEST (jamais vus à l'inscription).
"""
import numpy as np
import pandas as pd
from recognize import load_library, identify

ids, centroids = load_library()
Xt = np.load("embeddings/test_embeddings.npy")
yt = pd.read_csv("embeddings/test_labels.csv")

THRESHOLD = 0.5
rows = []
for emb, true_id, fname in zip(Xt, yt["person_id"], yt["filename"]):
    pred, dist = identify(emb, ids, centroids, THRESHOLD)
    rows.append((fname, true_id, pred, round(dist, 3), pred == true_id))

res = pd.DataFrame(rows, columns=["fichier", "vrai", "predit", "distance", "correct"])
print(res.to_string(index=False))

print(f"\nPrécision globale : {res['correct'].mean():.1%} "
      f"({res['correct'].sum()}/{len(res)})")
print("\nPrécision par personne :")
print(res.groupby("vrai")["correct"].mean().map("{:.0%}".format))
print(f"\nDistance moyenne des bonnes réponses : {res[res.correct]['distance'].mean():.3f}")
if (~res.correct).any():
    print(f"Distance moyenne des erreurs         : {res[~res.correct]['distance'].mean():.3f}")