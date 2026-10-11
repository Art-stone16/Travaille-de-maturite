# Résultats — best_relu

12 feuilles · Tests du 2026-09-25 au 2026-09-25 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 923/1227 | 75,22 % |
| Top-2 | 1092/1227 | 89,00 % |
| Top-3 | 1150/1227 | 93,72 % |
| Top-5 | 1198/1227 | 97,64 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 275 zones au Top-1 (+22.41 points). 29 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 98 | 100,00 % | 100,00 % | 100,00 % |
| 1 | 100 | 54 | 54,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 221 | 95,26 % | 98,28 % | 98,71 % |
| 3 | 102 | 82 | 80,39 % | 89,22 % | 96,08 % |
| 4 | 99 | 59 | 59,60 % | 87,88 % | 93,94 % |
| 5 | 105 | 93 | 88,57 % | 93,33 % | 95,24 % |
| 6 | 189 | 118 | 62,43 % | 91,53 % | 95,77 % |
| 7 | 100 | 69 | 69,00 % | 93,00 % | 100,00 % |
| 8 | 101 | 83 | 82,18 % | 92,08 % | 98,02 % |
| 9 | 101 | 46 | 45,54 % | 88,12 % | 99,01 % |

Meilleur taux Top-1 : chiffre(s) 0 (100,00 %). Plus faible : chiffre(s) 9 (45,54 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 6 → 5 | 40 |
| 1 → 7 | 39 |
| 9 → 0 | 25 |
| 9 → 3 | 20 |
| 7 → 2 | 19 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-09-25_16-03-16/predictions.csv) | 2026-09-25_16-03-16 | 100 | 98 | -2 | 100,00 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-09-25_16-03-20/predictions.csv) | 2026-09-25_16-03-20 | 100 | 100 | +0 | 54,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-09-25_16-03-22/predictions.csv) | 2026-09-25_16-03-22 | 100 | 130 | +30 | 99,23 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-09-25_16-03-23/predictions.csv) | 2026-09-25_16-03-23 | 100 | 102 | +2 | 90,20 % | 97,06 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-09-25_16-03-24/predictions.csv) | 2026-09-25_16-03-24 | 100 | 102 | +2 | 80,39 % | 96,08 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-09-25_16-03-27/predictions.csv) | 2026-09-25_16-03-27 | 100 | 99 | -1 | 59,60 % | 93,94 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-09-25_16-03-30/predictions.csv) | 2026-09-25_16-03-30 | 103 | 105 | +2 | 88,57 % | 95,24 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-09-25_16-03-32/predictions.csv) | 2026-09-25_16-03-32 | 100 | 91 | -9 | 63,74 % | 92,31 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-09-25_16-03-36/predictions.csv) | 2026-09-25_16-03-36 | 98 | 98 | +0 | 61,22 % | 98,98 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-09-25_16-03-37/predictions.csv) | 2026-09-25_16-03-37 | 100 | 100 | +0 | 69,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-09-25_16-03-41/predictions.csv) | 2026-09-25_16-03-41 | 100 | 101 | +1 | 82,18 % | 98,02 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-09-25_16-03-45/predictions.csv) | 2026-09-25_16-03-45 | 100 | 101 | +1 | 45,54 % | 99,01 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
