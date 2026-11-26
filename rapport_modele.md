# Partie « Modèle » du rapport (Ingénieur modèle)

> Texte à adapter et à intégrer au rapport de groupe (≤ 6 pages). Les chiffres viennent de `reports/final_metrics.md`, `reports/threshold_sweep.csv` et des figures `roc_curve.png` et `confusion_matrix*.png`. Les crochets [ ] sont à compléter.

## 1. Rôle et périmètre

L'ingénieur modèle intègre MTCNN et FaceNet, calcule la similarité, détecte la vivacité et fixe le seuil de décision. Le module est livré à l'ingénieur interface sous la forme d'une fonction unique (`engine.py`) : image de la webcam en entrée, identité, vivacité et décision en sortie.

## 2. Architecture du pipeline

```
Webcam -> MTCNN (détection, recadrage 160x160) -> FaceNet (embedding 512-D)
       -> distance cosinus à la carte d'identité de chaque inscrit
       -> identité = visage le plus proche, si distance < seuil
En parallèle : vivacité en 3 tests successifs (images, clignement, défi aléatoire)
Décision : PRÉSENT = vivacité réussie ET identité reconnue (vote sur toute la tentative)
```

- **Détection et recadrage** : MTCNN, confiance minimale 0,95, marge 20, taille 160x160 (mêmes paramètres que l'inscription, partagés dans `face_utils.py`).
- **Embeddings** : FaceNet (InceptionResnetV1, poids VGGFace2), vecteurs de 512 dimensions normalisés.
- **Bibliothèque** : un embedding moyen (centroïde) par personne, calculé sur les images d'inscription (69 vecteurs, 5 personnes, 70 % d'inscription et 30 % de test).
- **Similarité** : distance cosinus `1 - a·b`. Seuil retenu : **0,34**.
- **Identité stable** : la reconnaissance est répétée toutes les 5 images et la décision repose sur un vote (au moins 70 % des votes pour la même personne). Un changement de personne en cours de vérification est refusé.

## 3. Détection de vivacité (anti-usurpation)

Trois tests successifs, chacun avec son temps limite. La vérification est réussie seulement si les trois le sont.

| Test | Principe | Ce qu'il bloque |
|---|---|---|
| 1. Comparaison d'images | Variation des pixels et déformation de la forme du visage, après alignement (le mouvement rigide d'une photo tenue à la main est supprimé) | Image immobile |
| 2. Clignement (EAR) | Yeux ouverts, puis fermés, puis rouverts. La référence « ouvert » est mesurée sur la personne, donc des yeux déjà fermés ne comptent pas | Photo, même tremblante |
| 3. Défi aléatoire | Tourner la tête à gauche, à droite, en haut ou en bas (tirage au hasard, jamais deux fois de suite la même consigne). Le sens opposé fait échouer | Vidéo préenregistrée |

Une première version, basée uniquement sur la différence de pixels entre trois images, acceptait des photos tenues à la main : elle a été remplacée par cette approche combinée.

## 4. Choix du seuil

Évaluation hors ligne (97 échantillons de vraies personnes, dont 28 images de test et 69 images d'inscription en leave-one-out ; 97 inconnus simulés en retirant la carte d'une personne) :

- distances des vraies personnes : moyenne 0,079, maximum 0,274 ;
- distances des inconnus simulés : moyenne 0,621, minimum 0,494 ;
- marge de séparation : +0,22 ; FAR 0 % et FRR 0 % pour un seuil de 0,30 à 0,40 ; aucune confusion entre personnes inscrites ;
- le seuil de **0,34** se situe dans la zone sûre, avec de la marge pour les images plus éloignées de leur carte.

[Insérer `roc_curve.png` (FAR et FRR selon le seuil, courbe ROC) et `confusion_matrix.png`.]

Ces chiffres sont **optimistes** : inscription et test viennent de la même séance de collecte. Ils ont été confirmés par des essais en conditions réelles (section 5).

## 5. Résultats en conditions réelles (webcam, seuil 0,34)

| Critère | Objectif | Mesure | Résultat |
|---|---|---|---|
| Précision (vraie personne reconnue) | ≥ 90 % | 100 % (10/10) | Atteint |
| FAR (inconnu accepté) | < 5 % | 0 % (0/10) | Atteint |
| FRR (vraie personne refusée) | < 5 % | 0 % (0/10) | Atteint |
| Temps de reconnaissance par image | < 2 s | 0,31 s en moyenne, 1,56 s au maximum | Atteint |
| Photos statiques bloquées | 100 % | 100 % (15/15, bloquées au test de clignement) | Atteint |
| Vidéo rejouée | (limite connue) | 67 % (2/3, bloquées par le défi aléatoire) | Limite |

Durée moyenne d'une vérification complète : 10,4 s (tests de vivacité : images 2,4 s, clignement 2,9 s, défi 1,9 s).

[Insérer `confusion_matrix_live.png`.]

## 6. Limites

- **Échantillon restreint** : 5 personnes inscrites, et les essais réels portent sur une seule personne inscrite (10 essais), 10 inconnus, 15 photos et 3 vidéos. « 0 erreur sur 10 » ne signifie pas 0 % d'erreur : ces résultats doivent être confirmés sur un échantillon plus grand.
- **Vidéo** : une vidéo de la personne qui cligne et tourne la tête peut réussir les tests (1 vidéo sur 3 a réussi). Un deepfake en temps réel ou un masque 3D ne sont pas couverts.
- **Conditions de collecte** : inscription et test hors ligne dans la même séance ; la personne ayant le moins d'images d'inscription (9 contre 20 à 25 recommandées) est la plus fragile.
- **Éclairage, lunettes, angle** peuvent éloigner un visage de sa carte et augmenter le taux de faux rejets.
- **Temps maximal** (1,56 s) : probablement dû à la première reconnaissance après le lancement ; un passage à vide au démarrage (`warmup`) le réduit.
- **Un visage à la fois**, ensemble fermé : une personne non inscrite est refusée, jamais identifiée.

## 7. Aspects éthiques

- Le système utilise des identifiants (`person_XX`) ; la correspondance avec les noms est tenue dans un fichier séparé. [Vérifier qu'aucune donnée nominative n'est dans un dépôt public.]
- Consentement écrit de chaque participant et suppression des données après la démonstration (voir la section de l'ingénieur tests et éthique).
- **Risques** : un faux rejet pénalise un étudiant présent ; une fausse acceptation permet de signer pour un autre. Le système propose donc un **mécanisme de secours** : tout refus signale `needs_manual`, l'enseignant peut confirmer la présence à la main.
- Biais possibles selon l'éclairage et les caractéristiques des visages, à surveiller sur un échantillon plus diversifié.

## 8. Livrables du modèle

`face_utils.py` (paramètres partagés), `recognize.py` (bibliothèque et identification), `liveness.py` (3 tests), `ui.py` (affichage), `live_webcam.py` (démonstration), `engine.py` (module pour l'interface), `calibrate_threshold.py` et `summarize_trials.py` (évaluation), `collect_trials.py` (essais réels).