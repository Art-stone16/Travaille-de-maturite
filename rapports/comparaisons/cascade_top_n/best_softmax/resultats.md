# Résultats — best_softmax

12 feuilles · Tests du 2026-10-04 au 2026-10-04 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 880/1227 | 71,72 % |
| Top-2 | 1066/1227 | 86,88 % |
| Top-3 | 1143/1227 | 93,15 % |
| Top-5 | 1194/1227 | 97,31 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 314 zones au Top-1 (+25.59 points). 33 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 97 | 98,98 % | 100,00 % | 100,00 % |
| 1 | 100 | 65 | 65,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 226 | 97,41 % | 98,28 % | 98,71 % |
| 3 | 102 | 81 | 79,41 % | 94,12 % | 97,06 % |
| 4 | 99 | 47 | 47,47 % | 78,79 % | 88,89 % |
| 5 | 105 | 98 | 93,33 % | 94,29 % | 96,19 % |
| 6 | 189 | 82 | 43,39 % | 91,01 % | 96,83 % |
| 7 | 100 | 61 | 61,00 % | 93,00 % | 100,00 % |
| 8 | 101 | 84 | 83,17 % | 93,07 % | 98,02 % |
| 9 | 101 | 39 | 38,61 % | 84,16 % | 96,04 % |

Meilleur taux Top-1 : chiffre(s) 0 (98,98 %). Plus faible : chiffre(s) 9 (38,61 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 6 → 5 | 85 |
| 9 → 3 | 33 |
| 4 → 8 | 31 |
| 1 → 7 | 21 |
| 9 → 0 | 19 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-10-04_12-15-02/predictions.csv) | 2026-10-04_12-15-02 | 100 | 98 | -2 | 98,98 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-10-04_12-15-07/predictions.csv) | 2026-10-04_12-15-07 | 100 | 100 | +0 | 65,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-10-04_12-15-09/predictions.csv) | 2026-10-04_12-15-09 | 100 | 130 | +30 | 100,00 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-10-04_12-15-09/predictions.csv) | 2026-10-04_12-15-09 | 100 | 102 | +2 | 94,12 % | 97,06 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-10-04_12-15-11/predictions.csv) | 2026-10-04_12-15-11 | 100 | 102 | +2 | 79,41 % | 97,06 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-10-04_12-15-14/predictions.csv) | 2026-10-04_12-15-14 | 100 | 99 | -1 | 47,47 % | 88,89 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-10-04_12-15-18/predictions.csv) | 2026-10-04_12-15-18 | 103 | 105 | +2 | 93,33 % | 96,19 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-10-04_12-15-19/predictions.csv) | 2026-10-04_12-15-19 | 100 | 91 | -9 | 47,25 % | 95,60 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-10-04_12-15-21/predictions.csv) | 2026-10-04_12-15-21 | 98 | 98 | +0 | 39,80 % | 97,96 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-10-04_12-15-22/predictions.csv) | 2026-10-04_12-15-22 | 100 | 100 | +0 | 61,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-10-04_12-15-26/predictions.csv) | 2026-10-04_12-15-26 | 100 | 101 | +1 | 83,17 % | 98,02 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-10-04_12-15-30/predictions.csv) | 2026-10-04_12-15-30 | 100 | 101 | +1 | 38,61 % | 96,04 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
