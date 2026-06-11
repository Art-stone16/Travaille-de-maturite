# Structure du projet

## Scripts

- `scripts/train_modele_principal.py` : entraîne le modèle principal et sauvegarde `best_model.keras` + `final_model.keras`.
- `scripts/recherche_architectures.py` : teste beaucoup de combinaisons d'architectures et sauvegarde un CSV comparatif.
- `scripts/generer_color_map.py` : teste dropout/filtres et génère une color map d'accuracy.
- `scripts/matrice_confusion.py` : charge un modèle sauvegardé et génère les matrices de confusion.
- `scripts/test_stabilite.py` : répète plusieurs entraînements pour visualiser la stabilité du modèle.
- `scripts/test_condition_reelle.py` : teste le modèle sur une image réelle.
- `scripts/env_config.py` : configure les chemins du projet et les caches Python/Matplotlib.

## Données

- `donnees/image_reelle.jpg` : image réelle utilisée par `test_condition_reelle.py`.

## Modèles

- `modeles/modeles_valides/` : modèles principaux gardés pour analyse ou test.
- `modeles/recherche_architectures/` : modèles générés par la grande recherche en boucle.

## Sorties

- `sorties/graphiques/` : courbes d'entraînement.
- `sorties/matrices_confusion/` : matrices de confusion.
- `sorties/resultats/` : tableaux CSV de comparaison.
- `sorties/color_maps/` : color maps.
- `sorties/tests_condition_reelle/` : tests sur image réelle et graphiques de stabilité.

## Commandes utiles

Depuis la racine du projet :

```bash
.venv/bin/python scripts/test_condition_reelle.py
.venv/bin/python scripts/matrice_confusion.py
.venv/bin/python scripts/train_modele_principal.py
```
