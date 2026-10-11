# Comparaison des modèles — Cascade Top-N

## Résultats globaux

| Modèle | Feuilles | Attendus | Zones | Top-1 | Top-2 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [best_relu_10xcascade](best_relu_10xcascade/resultats.md) | 12 | 1201 | 1227 | 98,53 % | 99,35 % | 99,51 % | 99,59 % |
| [best_relu_2xcascade](best_relu_2xcascade/resultats.md) | 12 | 1201 | 1227 | 96,33 % | 98,94 % | 99,10 % | 99,51 % |
| [Best_relu_cascade_V2](Best_relu_cascade_V2/resultats.md) | 12 | 1201 | 1227 | 92,75 % | 95,93 % | 97,39 % | 98,86 % |
| [Best_relu_cascade](Best_relu_cascade/resultats.md) | 12 | 1201 | 1227 | 91,12 % | 95,11 % | 97,15 % | 98,37 % |
| [Best_COLOR_MAP_cascade](Best_COLOR_MAP_cascade/resultats.md) | 12 | 1201 | 1227 | 86,23 % | 94,13 % | 95,68 % | 98,04 % |
| [best_relu](best_relu/resultats.md) | 12 | 1201 | 1227 | 75,22 % | 89,00 % | 93,72 % | 97,64 % |
| [best_softmax](best_softmax/resultats.md) | 12 | 1201 | 1227 | 71,72 % | 86,88 % | 93,15 % | 97,31 % |
| [Best_COLOR_MAP](Best_COLOR_MAP/resultats.md) | 12 | 1201 | 1227 | 65,44 % | 82,64 % | 89,49 % | 96,74 % |

Modèles classés par précision globale Top-1 décroissante.

## Précision Top-1 par chiffre

| Modèle | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_relu_10xcascade | 100,00 % | 100,00 % | 98,28 % | 99,02 % | 96,97 % | 99,05 % | 96,30 % | 100,00 % | 99,01 % | 99,01 % |
| best_relu_2xcascade | 100,00 % | 100,00 % | 98,28 % | 94,12 % | 93,94 % | 97,14 % | 93,65 % | 99,00 % | 94,06 % | 93,07 % |
| Best_relu_cascade_V2 | 100,00 % | 99,00 % | 97,84 % | 82,35 % | 87,88 % | 92,38 % | 89,95 % | 96,00 % | 91,09 % | 87,13 % |
| Best_relu_cascade | 100,00 % | 97,00 % | 97,84 % | 81,37 % | 86,87 % | 88,57 % | 89,42 % | 89,00 % | 89,11 % | 85,15 % |
| Best_COLOR_MAP_cascade | 98,98 % | 92,00 % | 93,53 % | 84,31 % | 78,79 % | 87,62 % | 82,54 % | 90,00 % | 84,16 % | 64,36 % |
| best_relu | 100,00 % | 54,00 % | 95,26 % | 80,39 % | 59,60 % | 88,57 % | 62,43 % | 69,00 % | 82,18 % | 45,54 % |
| best_softmax | 98,98 % | 65,00 % | 97,41 % | 79,41 % | 47,47 % | 93,33 % | 43,39 % | 61,00 % | 83,17 % | 38,61 % |
| Best_COLOR_MAP | 98,98 % | 41,00 % | 95,69 % | 76,47 % | 52,53 % | 88,57 % | 41,27 % | 25,00 % | 83,17 % | 32,67 % |

## Méthode

Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu est le premier choix ; Top-N indique sa présence parmi les N premières propositions. Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. Ces scores ne mesurent donc pas une précision de détection annotée manuellement. Seule la dernière exécution de chaque couple modèle/feuille est retenue. Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. La précision globale est pondérée par le nombre de zones.

Les campagnes ont été réalisées à des dates différentes ; la comparaison décrit les résultats sauvegardés. Un écart entre zones détectées et chiffres attendus est un bilan de comptage, pas un nombre exact de faux positifs ou de chiffres manqués.

Régénération depuis la racine du projet : `python scripts/visualisation/generer_rapports_cascade_md.py`.
