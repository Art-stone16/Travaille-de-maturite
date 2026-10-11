# Résultats — Best_COLOR_MAP

12 feuilles · Tests du 2026-07-24 au 2026-07-25 · 1201 chiffres attendus · 1227 zones détectées · Écart net : +26.

## Résultats globaux

| Indicateur | Corrects / zones | Taux |
| --- | --- | --- |
| Top-1 | 803/1227 | 65,44 % |
| Top-2 | 1014/1227 | 82,64 % |
| Top-3 | 1098/1227 | 89,49 % |
| Top-5 | 1187/1227 | 96,74 % |
| Top-10 | 1227/1227 | 100,00 % |

Le Top-5 ajoute 384 zones au Top-1 (+31.30 points). 40 zones restent hors du Top-5.

## Résultats par chiffre

| Chiffre | Zones | Corrects Top-1 | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| 0 | 98 | 97 | 98,98 % | 100,00 % | 100,00 % |
| 1 | 100 | 41 | 41,00 % | 83,00 % | 99,00 % |
| 2 | 232 | 222 | 95,69 % | 98,28 % | 98,28 % |
| 3 | 102 | 78 | 76,47 % | 95,10 % | 100,00 % |
| 4 | 99 | 52 | 52,53 % | 79,80 % | 88,89 % |
| 5 | 105 | 93 | 88,57 % | 95,24 % | 97,14 % |
| 6 | 189 | 78 | 41,27 % | 88,36 % | 94,18 % |
| 7 | 100 | 25 | 25,00 % | 67,00 % | 97,00 % |
| 8 | 101 | 84 | 83,17 % | 93,07 % | 98,02 % |
| 9 | 101 | 33 | 32,67 % | 84,16 % | 95,05 % |

Meilleur taux Top-1 : chiffre(s) 0 (98,98 %). Plus faible : chiffre(s) 7 (25,00 %).

## Principales confusions Top-1

| Attendu → prédit | Nombre de zones |
| --- | --- |
| 6 → 5 | 76 |
| 1 → 7 | 46 |
| 7 → 2 | 43 |
| 9 → 3 | 35 |
| 9 → 0 | 22 |

## Feuilles et sources

| Feuille / prédictions | Exécution | Attendus | Détectés | Écart | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| [CTN_0](../../../../resultats/cascade_top_n/CTN_0/2026-07-25_09-56-17/predictions.csv) | 2026-07-25_09-56-17 | 100 | 98 | -2 | 98,98 % | 100,00 % |
| [CTN_1](../../../../resultats/cascade_top_n/CTN_1/2026-07-24_14-42-21/predictions.csv) | 2026-07-24_14-42-21 | 100 | 100 | +0 | 41,00 % | 99,00 % |
| [CTN_2](../../../../resultats/cascade_top_n/CTN_2/2026-07-24_14-45-21/predictions.csv) | 2026-07-24_14-45-21 | 100 | 130 | +30 | 99,23 % | 100,00 % |
| [CTN_2.2](../../../../resultats/cascade_top_n/CTN_2.2/2026-07-24_14-45-30/predictions.csv) | 2026-07-24_14-45-30 | 100 | 102 | +2 | 91,18 % | 96,08 % |
| [CTN_3](../../../../resultats/cascade_top_n/CTN_3/2026-07-24_14-31-31/predictions.csv) | 2026-07-24_14-31-31 | 100 | 102 | +2 | 76,47 % | 100,00 % |
| [CTN_4](../../../../resultats/cascade_top_n/CTN_4/2026-07-25_09-09-24/predictions.csv) | 2026-07-25_09-09-24 | 100 | 99 | -1 | 52,53 % | 88,89 % |
| [CTN_5](../../../../resultats/cascade_top_n/CTN_5/2026-07-25_09-16-56/predictions.csv) | 2026-07-25_09-16-56 | 103 | 105 | +2 | 88,57 % | 97,14 % |
| [CTN_6](../../../../resultats/cascade_top_n/CTN_6/2026-07-25_09-23-41/predictions.csv) | 2026-07-25_09-23-41 | 100 | 91 | -9 | 48,35 % | 93,41 % |
| [CTN_6.2](../../../../resultats/cascade_top_n/CTN_6.2/2026-07-25_09-30-33/predictions.csv) | 2026-07-25_09-30-33 | 98 | 98 | +0 | 34,69 % | 94,90 % |
| [CTN_7](../../../../resultats/cascade_top_n/CTN_7/2026-07-25_09-39-36/predictions.csv) | 2026-07-25_09-39-36 | 100 | 100 | +0 | 25,00 % | 97,00 % |
| [CTN_8](../../../../resultats/cascade_top_n/CTN_8/2026-07-25_09-56-38/predictions.csv) | 2026-07-25_09-56-38 | 100 | 101 | +1 | 83,17 % | 98,02 % |
| [CTN_9](../../../../resultats/cascade_top_n/CTN_9/2026-07-25_09-56-55/predictions.csv) | 2026-07-25_09-56-55 | 100 | 101 | +1 | 32,67 % | 95,05 % |

## Lecture des résultats

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

[Comparaison des modèles](../comparaison_modeles.md)
