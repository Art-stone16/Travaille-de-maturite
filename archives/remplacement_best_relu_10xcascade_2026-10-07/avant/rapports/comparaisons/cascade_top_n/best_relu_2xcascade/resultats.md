# Résultats — best_relu_2xcascade

12 feuilles · Tests du 2026-09-25 au 2026-09-25 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 1182/1227 | 96,33 % |
| Top-2 | 1214/1227 | 98,94 % |
| Top-3 | 1216/1227 | 99,10 % |
| Top-5 | 1221/1227 | 99,51 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 39 zones au Top-1 (+3.18 points). 6 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 98 | 100,00 % | 100,00 % | 100,00 % |
| 1 | 100 | 100 | 100,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 228 | 98,28 % | 98,28 % | 98,71 % |
| 3 | 102 | 96 | 94,12 % | 100,00 % | 100,00 % |
| 4 | 99 | 93 | 93,94 % | 98,99 % | 100,00 % |
| 5 | 105 | 102 | 97,14 % | 100,00 % | 100,00 % |
| 6 | 189 | 177 | 93,65 % | 98,41 % | 98,94 % |
| 7 | 100 | 99 | 99,00 % | 100,00 % | 100,00 % |
| 8 | 101 | 95 | 94,06 % | 98,02 % | 99,01 % |
| 9 | 101 | 94 | 93,07 % | 99,01 % | 100,00 % |

Meilleur taux Top-1 : chiffre(s) 0, 1 (100,00 %). Plus faible : chiffre(s) 9 (93,07 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 3 → 2 | 6 |
| 6 → 1 | 4 |
| 8 → 0 | 4 |
| 4 → 1 | 3 |
| 5 → 2 | 3 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-09-25_15-48-52/predictions.csv) | 2026-09-25_15-48-52 | 100 | 98 | -2 | 100,00 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-09-25_15-48-56/predictions.csv) | 2026-09-25_15-48-56 | 100 | 100 | +0 | 100,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-09-25_15-48-58/predictions.csv) | 2026-09-25_15-48-58 | 100 | 130 | +30 | 100,00 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-09-25_15-48-59/predictions.csv) | 2026-09-25_15-48-59 | 100 | 102 | +2 | 96,08 % | 97,06 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-09-25_15-49-00/predictions.csv) | 2026-09-25_15-49-00 | 100 | 102 | +2 | 94,12 % | 100,00 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-09-25_15-49-04/predictions.csv) | 2026-09-25_15-49-04 | 100 | 99 | -1 | 93,94 % | 100,00 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-09-25_15-49-07/predictions.csv) | 2026-09-25_15-49-07 | 103 | 105 | +2 | 97,14 % | 100,00 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-09-25_15-49-08/predictions.csv) | 2026-09-25_15-49-08 | 100 | 91 | -9 | 87,91 % | 97,80 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-09-25_15-49-10/predictions.csv) | 2026-09-25_15-49-10 | 98 | 98 | +0 | 98,98 % | 100,00 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-09-25_15-49-11/predictions.csv) | 2026-09-25_15-49-11 | 100 | 100 | +0 | 99,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-09-25_15-49-14/predictions.csv) | 2026-09-25_15-49-14 | 100 | 101 | +1 | 94,06 % | 99,01 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-09-25_15-49-18/predictions.csv) | 2026-09-25_15-49-18 | 100 | 101 | +1 | 93,07 % | 100,00 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
