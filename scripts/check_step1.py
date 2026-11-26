import numpy as np
import pandas as pd

# Charger la bibliothèque d'inscription et le test
X = np.load("embeddings/embeddings.npy")
y = pd.read_csv("embeddings/labels.csv")
Xt = np.load("embeddings/test_embeddings.npy")
yt = pd.read_csv("embeddings/test_labels.csv")

print("Inscription :", X.shape, "| labels :", y.shape)
print("Test        :", Xt.shape, "| labels :", yt.shape)

print("\nImages par personne (inscription) :")
print(y["person_id"].value_counts().sort_index())

print("\nImages par personne (test) :")
print(yt["person_id"].value_counts().sort_index())

print("\nNormes des 5 premiers vecteurs (doivent valoir ~1.0) :")
print(np.linalg.norm(X, axis=1)[:5])