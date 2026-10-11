# Résultats — Best_relu_cascade_V2

12 feuilles · Tests du 2026-09-05 au 2026-09-05 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 1138/1227 | 92,75 % |
| Top-2 | 1177/1227 | 95,93 % |
| Top-3 | 1195/1227 | 97,39 % |
| Top-5 | 1213/1227 | 98,86 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 75 zones au Top-1 (+6.11 points). 14 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 98 | 100,00 % | 100,00 % | 100,00 % |
| 1 | 100 | 99 | 99,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 227 | 97,84 % | 98,28 % | 98,28 % |
| 3 | 102 | 84 | 82,35 % | 95,10 % | 99,02 % |
| 4 | 99 | 87 | 87,88 % | 96,97 % | 100,00 % |
| 5 | 105 | 97 | 92,38 % | 94,29 % | 97,14 % |
| 6 | 189 | 170 | 89,95 % | 96,30 % | 97,88 % |
| 7 | 100 | 96 | 96,00 % | 100,00 % | 100,00 % |
| 8 | 101 | 92 | 91,09 % | 97,03 % | 99,01 % |
| 9 | 101 | 88 | 87,13 % | 96,04 % | 99,01 % |

Meilleur taux Top-1 : chiffre(s) 0 (100,00 %). Plus faible : chiffre(s) 3 (82,35 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 3 → 2 | 14 |
| 6 → 0 | 9 |
| 8 → 0 | 7 |
| 5 → 2 | 5 |
| 4 → 6 | 4 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-09-05_15-53-31/predictions.csv) | 2026-09-05_15-53-31 | 100 | 98 | -2 | 100,00 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-09-05_15-53-35/predictions.csv) | 2026-09-05_15-53-35 | 100 | 100 | +0 | 99,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-09-05_15-53-38/predictions.csv) | 2026-09-05_15-53-38 | 100 | 130 | +30 | 100,00 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-09-05_15-53-38/predictions.csv) | 2026-09-05_15-53-38 | 100 | 102 | +2 | 95,10 % | 96,08 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-09-05_15-53-39/predictions.csv) | 2026-09-05_15-53-39 | 100 | 102 | +2 | 82,35 % | 99,02 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-09-05_15-53-43/predictions.csv) | 2026-09-05_15-53-43 | 100 | 99 | -1 | 87,88 % | 100,00 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-09-05_15-53-46/predictions.csv) | 2026-09-05_15-53-46 | 103 | 105 | +2 | 92,38 % | 97,14 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-09-05_15-53-47/predictions.csv) | 2026-09-05_15-53-47 | 100 | 91 | -9 | 87,91 % | 95,60 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-09-05_15-53-49/predictions.csv) | 2026-09-05_15-53-49 | 98 | 98 | +0 | 91,84 % | 100,00 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-09-05_15-53-50/predictions.csv) | 2026-09-05_15-53-50 | 100 | 100 | +0 | 96,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-09-05_15-53-53/predictions.csv) | 2026-09-05_15-53-53 | 100 | 101 | +1 | 91,09 % | 99,01 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-09-05_15-53-57/predictions.csv) | 2026-09-05_15-53-57 | 100 | 101 | +1 | 87,13 % | 99,01 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
