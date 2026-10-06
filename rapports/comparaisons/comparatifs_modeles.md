# Comparatif des modèles actifs

Configuration des huit modèles utilisés pour les tests terrain, relevée directement dans leurs fichiers `best_model.keras` le 6 octobre 2026. Ils sont tous regroupés sous `modeles/actifs/<nom_modele>/`. Les anciens modèles sont conservés séparément dans `archives/modeles/`.

| Nom | Nombre de paramètres | Nombre de filtres première convolution | FA première convolution | Nombre de filtres deuxième convolution | FA deuxième convolution | Dropout | Fonction d’activation de sortie | Données de cascade |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| [Best_COLOR_MAP](../../modeles/actifs/Best_COLOR_MAP/best_model.keras) | 1 962 | 4 | softmax | 8 | softmax | 20 % | softmax | Non |
| [Best_COLOR_MAP_cascade](../../modeles/actifs/Best_COLOR_MAP_cascade/best_model.keras) | 1 962 | 4 | softmax | 8 | softmax | 20 % | softmax | Oui |
| [best_relu](../../modeles/actifs/best_relu/best_model.keras) | 13 258 | 16 | relu | 32 | relu | 20 % | softmax | Non |
| [best_relu_10xcascade](../../modeles/actifs/best_relu_10xcascade/best_model.keras) | 55 114 | 64 | relu | 64 | relu | 60 % | softmax | Oui |
| [best_relu_2xcascade](../../modeles/actifs/best_relu_2xcascade/best_model.keras) | 55 114 | 64 | relu | 64 | relu | 60 % | softmax | Oui |
| [Best_relu_cascade](../../modeles/actifs/Best_relu_cascade/best_model.keras) | 13 258 | 16 | relu | 32 | relu | 20 % | softmax | Oui |
| [Best_relu_cascade_V2](../../modeles/actifs/Best_relu_cascade_V2/best_model.keras) | 55 114 | 64 | relu | 64 | relu | 60 % | softmax | Oui |
| [best_softmax](../../modeles/actifs/best_softmax/best_model.keras) | 183 946 | 128 | softmax | 128 | softmax | 60 % | softmax | Non |

FA = fonction d’activation. La colonne « Fonction d’activation de sortie » correspond à l’activation de la couche Dense de sortie (10 classes).

La colonne « Données de cascade » indique si le modèle a été entraîné avec ces données. Un test sur une feuille de cascade ne compte pas comme un entraînement.

Le nombre de paramètres est le total renvoyé par `model.count_params()` : paramètres entraînables et non entraînables du modèle, y compris ceux de BatchNormalization. Le dropout indique la proportion désactivée pendant l’entraînement.

Des modèles peuvent avoir la même architecture et le même nombre de paramètres tout en ayant des poids et des résultats différents.
