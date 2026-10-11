# Résultats — best_relu_10xcascade

**Contrôle sur les sources d’entraînement Cascade, sans indépendance.** Les 1 217 chiffres validés de ces feuilles ont alimenté le dataset. Les scores sur toutes les zones détectées incluent aussi les dix zones exclues. Le test final indépendant de la nouvelle expérience ×10 est MNIST test. Le modèle ×2 conservé utilise le protocole historique avec copies partagées entre entraînement et validation.

12 feuilles · Tests du 2026-10-07 au 2026-10-07 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 1215/1227 | 99,02 % |
| Top-2 | 1219/1227 | 99,35 % |
| Top-3 | 1219/1227 | 99,35 % |
| Top-5 | 1221/1227 | 99,51 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 6 zones au Top-1 (+0.49 points). 6 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 98 | 100,00 % | 100,00 % | 100,00 % |
| 1 | 100 | 100 | 100,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 228 | 98,28 % | 98,28 % | 98,28 % |
| 3 | 102 | 102 | 100,00 % | 100,00 % | 100,00 % |
| 4 | 99 | 98 | 98,99 % | 100,00 % | 100,00 % |
| 5 | 105 | 102 | 97,14 % | 98,10 % | 100,00 % |
| 6 | 189 | 188 | 99,47 % | 99,47 % | 99,47 % |
| 7 | 100 | 100 | 100,00 % | 100,00 % | 100,00 % |
| 8 | 101 | 99 | 98,02 % | 100,00 % | 100,00 % |
| 9 | 101 | 100 | 99,01 % | 99,01 % | 99,01 % |

Meilleur taux Top-1 : chiffre(s) 0, 1, 3, 7 (100,00 %). Plus faible : chiffre(s) 5 (97,14 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 2 → 9 | 1 |
| 2 → 1 | 1 |
| 2 → 7 | 1 |
| 2 → 4 | 1 |
| 4 → 9 | 1 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-10-07_13-23-57/predictions.csv) | 2026-10-07_13-23-57 | 100 | 98 | -2 | 100,00 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-10-07_13-24-02/predictions.csv) | 2026-10-07_13-24-02 | 100 | 100 | +0 | 100,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-10-07_13-24-04/predictions.csv) | 2026-10-07_13-24-04 | 100 | 130 | +30 | 100,00 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-10-07_13-24-05/predictions.csv) | 2026-10-07_13-24-05 | 100 | 102 | +2 | 96,08 % | 96,08 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-10-07_13-24-06/predictions.csv) | 2026-10-07_13-24-06 | 100 | 102 | +2 | 100,00 % | 100,00 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-10-07_13-24-10/predictions.csv) | 2026-10-07_13-24-10 | 100 | 99 | -1 | 98,99 % | 100,00 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-10-07_13-24-13/predictions.csv) | 2026-10-07_13-24-13 | 103 | 105 | +2 | 97,14 % | 100,00 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-10-07_13-24-14/predictions.csv) | 2026-10-07_13-24-14 | 100 | 91 | -9 | 98,90 % | 98,90 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-10-07_13-24-16/predictions.csv) | 2026-10-07_13-24-16 | 98 | 98 | +0 | 100,00 % | 100,00 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-10-07_13-24-17/predictions.csv) | 2026-10-07_13-24-17 | 100 | 100 | +0 | 100,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-10-07_13-24-20/predictions.csv) | 2026-10-07_13-24-20 | 100 | 101 | +1 | 98,02 % | 100,00 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-10-07_13-24-25/predictions.csv) | 2026-10-07_13-24-25 | 100 | 101 | +1 | 99,01 % | 99,01 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
