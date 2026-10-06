# Résultats — Best_COLOR_MAP_cascade

12 feuilles · Tests du 2026-10-01 au 2026-10-01 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 1058/1227 | 86,23 % |
| Top-2 | 1155/1227 | 94,13 % |
| Top-3 | 1174/1227 | 95,68 % |
| Top-5 | 1203/1227 | 98,04 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 145 zones au Top-1 (+11.82 points). 24 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 97 | 98,98 % | 100,00 % | 100,00 % |
| 1 | 100 | 92 | 92,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 217 | 93,53 % | 98,28 % | 98,28 % |
| 3 | 102 | 86 | 84,31 % | 97,06 % | 100,00 % |
| 4 | 99 | 78 | 78,79 % | 88,89 % | 92,93 % |
| 5 | 105 | 92 | 87,62 % | 92,38 % | 99,05 % |
| 6 | 189 | 156 | 82,54 % | 94,71 % | 96,30 % |
| 7 | 100 | 90 | 90,00 % | 99,00 % | 100,00 % |
| 8 | 101 | 85 | 84,16 % | 93,07 % | 96,04 % |
| 9 | 101 | 65 | 64,36 % | 91,09 % | 99,01 % |

Meilleur taux Top-1 : chiffre(s) 0 (98,98 %). Plus faible : chiffre(s) 9 (64,36 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 9 → 0 | 19 |
| 6 → 5 | 12 |
| 3 → 2 | 11 |
| 6 → 0 | 10 |
| 9 → 3 | 8 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-10-01_14-58-46/predictions.csv) | 2026-10-01_14-58-46 | 100 | 98 | -2 | 98,98 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-10-01_14-58-50/predictions.csv) | 2026-10-01_14-58-50 | 100 | 100 | +0 | 92,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-10-01_14-58-53/predictions.csv) | 2026-10-01_14-58-53 | 100 | 130 | +30 | 99,23 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-10-01_14-58-53/predictions.csv) | 2026-10-01_14-58-53 | 100 | 102 | +2 | 86,27 % | 96,08 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-10-01_14-58-54/predictions.csv) | 2026-10-01_14-58-54 | 100 | 102 | +2 | 84,31 % | 100,00 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-10-01_14-58-58/predictions.csv) | 2026-10-01_14-58-58 | 100 | 99 | -1 | 78,79 % | 92,93 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-10-01_14-59-01/predictions.csv) | 2026-10-01_14-59-01 | 103 | 105 | +2 | 87,62 % | 99,05 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-10-01_14-59-03/predictions.csv) | 2026-10-01_14-59-03 | 100 | 91 | -9 | 75,82 % | 94,51 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-10-01_14-59-05/predictions.csv) | 2026-10-01_14-59-05 | 98 | 98 | +0 | 88,78 % | 97,96 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-10-01_14-59-06/predictions.csv) | 2026-10-01_14-59-06 | 100 | 100 | +0 | 90,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-10-01_14-59-09/predictions.csv) | 2026-10-01_14-59-09 | 100 | 101 | +1 | 84,16 % | 96,04 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-10-01_14-59-13/predictions.csv) | 2026-10-01_14-59-13 | 100 | 101 | +1 | 64,36 % | 99,01 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
