# Resultats des essais en conditions reelles (seuil = 0.34)

Essais : 38 au total | 10 vraies personnes | 10 inconnus | 18 tentatives d'usurpation

## Vraies personnes inscrites
- Reconnues correctement : 10/10  -> precision = 100.0%
- FRR (vraie personne non reconnue) = 0.0%
    - refusees par la vivacite   : 0
    - refusees par la reconnaissance (distance/instabilite) : 0
    - prises pour quelqu'un d'autre : 0

| Personne | Essais | Reconnus | FRR |
|---|---|---|---|
| person_01 | 10 | 10 | 0.0% |

## Personnes inconnues (non inscrites)
- Acceptees a tort : 0/10  -> FAR = 0.0%

FAR global (A reconnu comme B, ou inconnu accepte) = 0.0%

## Anti-usurpation
| Attaque | Essais | Acceptees | Bloquees | Taux de blocage | Bloquee au |
|---|---|---|---|---|---|
| photo_phone | 15 | 0 | 15 | 100.0% | {'test 2': 15} |
| video | 3 | 1 | 2 | 66.7% | {'test 3': 2} |

Critere du cahier des charges = photos statiques (telephone + imprimee) bloquees a 100 %.
La video est une limite connue de la methode (clignement + defi) : 66.7% bloquees. A discuter dans le rapport.

## Temps
- Temps de reconnaissance par image (MTCNN + FaceNet + comparaison) : moyenne 306 ms | maximum 1559 ms
- Duree moyenne des tests de vivacite (vraies personnes) : T1 2.4 s | T2 2.9 s | T3 1.9 s
- Duree moyenne d'un essai complet : 10.4 s

## Objectifs du cahier des charges
| Critere | Objectif | Mesure | Resultat |
|---|---|---|---|
| Precision | >= 90 % | 100.0% | OK |
| FAR | < 5 % | 0.0% | OK |
| FRR | < 5 % | 0.0% | OK |
| Temps de reconnaissance | < 2 s | 0.31 s | OK |
| Anti-usurpation (photos statiques) | 100 % bloquees | 100.0% | OK |

## Effet du seuil sur les essais reels (identite seule, sans vivacite)
| Seuil | FAR (inconnus acceptes) | FRR (vrais non reconnus) |
|---|---|---|
| 0.20 | 0.0% | 0.0% |
| 0.25 | 0.0% | 0.0% |
| 0.30 | 0.0% | 0.0% |
| 0.35 | 0.0% | 0.0% |
| 0.40 | 0.0% | 0.0% |
| 0.45 | 0.0% | 0.0% |
| 0.50 | 70.0% | 0.0% |

Matrice de confusion : reports/confusion_matrix_live.csv / .png