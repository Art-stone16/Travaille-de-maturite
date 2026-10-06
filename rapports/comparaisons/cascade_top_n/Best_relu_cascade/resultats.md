# Résultats — Best_relu_cascade

12 feuilles · Tests du 2026-09-25 au 2026-09-25 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 1118/1227 | 91,12 % |
| Top-2 | 1167/1227 | 95,11 % |
| Top-3 | 1192/1227 | 97,15 % |
| Top-5 | 1207/1227 | 98,37 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 89 zones au Top-1 (+7.25 points). 20 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 98 | 100,00 % | 100,00 % | 100,00 % |
| 1 | 100 | 97 | 97,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 227 | 97,84 % | 98,71 % | 98,71 % |
| 3 | 102 | 83 | 81,37 % | 96,08 % | 98,04 % |
| 4 | 99 | 86 | 86,87 % | 94,95 % | 96,97 % |
| 5 | 105 | 93 | 88,57 % | 93,33 % | 96,19 % |
| 6 | 189 | 169 | 89,42 % | 96,30 % | 97,88 % |
| 7 | 100 | 89 | 89,00 % | 99,00 % | 100,00 % |
| 8 | 101 | 90 | 89,11 % | 94,06 % | 98,02 % |
| 9 | 101 | 86 | 85,15 % | 98,02 % | 98,02 % |

Meilleur taux Top-1 : chiffre(s) 0 (100,00 %). Plus faible : chiffre(s) 3 (81,37 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 3 → 2 | 13 |
| 6 → 0 | 9 |
| 7 → 2 | 7 |
| 8 → 0 | 7 |
| 5 → 2 | 6 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-09-25_16-06-04/predictions.csv) | 2026-09-25_16-06-04 | 100 | 98 | -2 | 100,00 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-09-25_16-06-08/predictions.csv) | 2026-09-25_16-06-08 | 100 | 100 | +0 | 97,00 % | 100,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-09-25_16-06-11/predictions.csv) | 2026-09-25_16-06-11 | 100 | 130 | +30 | 100,00 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-09-25_16-06-11/predictions.csv) | 2026-09-25_16-06-11 | 100 | 102 | +2 | 95,10 % | 97,06 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-09-25_16-06-12/predictions.csv) | 2026-09-25_16-06-12 | 100 | 102 | +2 | 81,37 % | 98,04 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-09-25_16-06-16/predictions.csv) | 2026-09-25_16-06-16 | 100 | 99 | -1 | 86,87 % | 96,97 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-09-25_16-06-19/predictions.csv) | 2026-09-25_16-06-19 | 103 | 105 | +2 | 88,57 % | 96,19 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-09-25_16-06-20/predictions.csv) | 2026-09-25_16-06-20 | 100 | 91 | -9 | 85,71 % | 95,60 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-09-25_16-06-22/predictions.csv) | 2026-09-25_16-06-22 | 98 | 98 | +0 | 92,86 % | 100,00 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-09-25_16-06-23/predictions.csv) | 2026-09-25_16-06-23 | 100 | 100 | +0 | 89,00 % | 100,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-09-25_16-06-26/predictions.csv) | 2026-09-25_16-06-26 | 100 | 101 | +1 | 89,11 % | 98,02 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-09-25_16-06-30/predictions.csv) | 2026-09-25_16-06-30 | 100 | 101 | +1 | 85,15 % | 98,02 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
