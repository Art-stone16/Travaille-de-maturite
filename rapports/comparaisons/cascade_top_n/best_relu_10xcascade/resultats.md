# Résultats — best_relu_10xcascade

12 feuilles · Tests du 2026-09-25 au 2026-09-25 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 1209/1227 | 98,53 % |
| Top-2 | 1219/1227 | 99,35 % |
| Top-3 | 1221/1227 | 99,51 % |
| Top-5 | 1222/1227 | 99,59 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 13 zones au Top-1 (+1.06 points). 5 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 98 | 100,00 % | 100,00 % | 100,00 % |
| 1 | 100 | 100 | 100,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 228 | 98,28 % | 98,28 % | 98,28 % |
| 3 | 102 | 101 | 99,02 % | 100,00 % | 100,00 % |
| 4 | 99 | 96 | 96,97 % | 100,00 % | 100,00 % |
| 5 | 105 | 104 | 99,05 % | 100,00 % | 100,00 % |
| 6 | 189 | 182 | 96,30 % | 99,47 % | 99,47 % |
| 7 | 100 | 100 | 100,00 % | 100,00 % | 100,00 % |
| 8 | 101 | 100 | 99,01 % | 100,00 % | 100,00 % |
| 9 | 101 | 100 | 99,01 % | 99,01 % | 100,00 % |

Meilleur taux Top-1 : chiffre(s) 0, 1, 7 (100,00 %). Plus faible : chiffre(s) 6 (96,30 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 4 → 1 | 3 |
| 6 → 7 | 2 |
| 6 → 0 | 2 |
| 2 → 5 | 1 |
| 2 → 1 | 1 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-09-25_15-55-52/predictions.csv) | 2026-09-25_15-55-52 | 100 | 98 | -2 | 100,00 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-09-25_15-55-56/predictions.csv) | 2026-09-25_15-55-56 | 100 | 100 | +0 | 100,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-09-25_15-55-58/predictions.csv) | 2026-09-25_15-55-58 | 100 | 130 | +30 | 100,00 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-09-25_15-55-59/predictions.csv) | 2026-09-25_15-55-59 | 100 | 102 | +2 | 96,08 % | 96,08 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-09-25_15-56-00/predictions.csv) | 2026-09-25_15-56-00 | 100 | 102 | +2 | 99,02 % | 100,00 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-09-25_15-56-04/predictions.csv) | 2026-09-25_15-56-04 | 100 | 99 | -1 | 96,97 % | 100,00 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-09-25_15-56-07/predictions.csv) | 2026-09-25_15-56-07 | 103 | 105 | +2 | 99,05 % | 100,00 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-09-25_15-56-08/predictions.csv) | 2026-09-25_15-56-08 | 100 | 91 | -9 | 93,41 % | 98,90 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-09-25_15-56-11/predictions.csv) | 2026-09-25_15-56-11 | 98 | 98 | +0 | 98,98 % | 100,00 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-09-25_15-56-11/predictions.csv) | 2026-09-25_15-56-11 | 100 | 100 | +0 | 100,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-09-25_15-56-15/predictions.csv) | 2026-09-25_15-56-15 | 100 | 101 | +1 | 99,01 % | 100,00 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-09-25_15-56-19/predictions.csv) | 2026-09-25_15-56-19 | 100 | 101 | +1 | 99,01 % | 100,00 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
