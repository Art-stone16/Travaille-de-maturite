# Guide des expériences

Ce guide donne l'ordre recommandé et les commandes à lancer depuis la racine du
projet. Les scripts affichent le chemin exact du dossier créé à la fin.

## 1. Tester une photographie terrain et contrôler le 28×28

Place d'abord les nouvelles photos dans
`donnees/tests_terrain_a_analyser/<nom_du_protocole>/`.

Exemple pour une feuille contenant uniquement des 7 :

```bash
.venv/bin/python scripts/test_condition_reelle.py \
  --image donnees/tests_terrain_a_analyser/papier_blanc_stylo_noir/classe_7.jpg \
  --chiffre-reel 7 \
  --nom-experience papier_blanc_stylo_noir
```

Le script crée une planche QC et une matrice CSV 0/1 pour chaque chiffre. Il
ajoute aussi les résultats au fichier
`sorties/tests_condition_reelle/<experience>/journal_global_tests_terrain.csv`.

Utilise `--chiffre-reel inconnu` pour une image dont la vérité terrain n'est pas
commune à tous les chiffres. `--chiffre-reel auto` n'est fiable que si le nom du
fichier contient un unique chiffre isolé, par exemple `classe_7.jpg`.

Protocole conseillé : même papier blanc, même stylo noir, même distance, lumière
homogène, au moins trois écritures par classe, et conservation de tous les
résultats — y compris les erreurs.

## 2. Calculer les chiffres moyens MNIST

Analyse complète du jeu d'entraînement :

```bash
.venv/bin/python scripts/analyser_chiffres_moyens.py \
  --source entrainement \
  --nom-experience mnist_entrainement_complet
```

Avec deux écritures personnelles isolées :

```bash
.venv/bin/python scripts/analyser_chiffres_moyens.py \
  --source entrainement \
  --nom-experience comparaison_ecriture_personnelle \
  --personnel 2=donnees/ecritures_personnelles/mon_2.jpg \
  --personnel 7=donnees/ecritures_personnelles/mon_7.jpg
```

Une image fournie avec `--personnel` doit contenir un seul chiffre. Les mesures
MAE, RMSE et corrélation sont descriptives : elles ne remplacent pas l'accuracy
du classificateur.

## 3. Préparer un dataset généré par Claude

Le modèle de demande se trouve dans
`donnees/synthetiques_a_importer/PROMPT_CLAUDE.md`. Enregistre la réponse JSON,
sans la modifier silencieusement, dans ce même dossier.

Importer et contrôler le lot :

```bash
.venv/bin/python scripts/preparer_dataset_synthetique.py importer \
  --entree donnees/synthetiques_a_importer/lot_claude_01.json \
  --nom-dataset claude_lot_01 \
  --origine claude \
  --description "Premier lot Claude, plusieurs styles manuscrits" \
  --binaire
```

Revalider plus tard :

```bash
.venv/bin/python scripts/preparer_dataset_synthetique.py valider \
  --dataset donnees/datasets_synthetiques/claude_lot_01
```

Le pipeline accepte aussi les images, NPY, NPZ, CSV et TXT. Les fichiers bruts
restent dans `synthetiques_a_importer/`; les matrices validées vont dans
`datasets_synthetiques/`. Les données IA ne sont donc jamais mélangées
silencieusement avec MNIST.

Pour disposer d'un témoin non-IA :

```bash
.venv/bin/python scripts/preparer_dataset_synthetique.py generer-baseline \
  --nom-dataset baseline_procedural_01 \
  --par-classe 20
```

Cette baseline utilise les polices OpenCV et est explicitement identifiée comme
procédurale, pas comme manuscrite ni générée par IA.

## 4. Tester les cartographies sans entraînement long

Afficher d'abord le plan exact, sans créer de fichiers :

```bash
.venv/bin/python scripts/generer_color_map.py \
  --nom-experience grille_dense_v1 \
  --dry-run
```

Valider le pipeline avec une petite grille :

```bash
.venv/bin/python scripts/generer_color_map.py \
  --nom-experience validation_rapide \
  --preset-rapide
```

La grille dense par défaut représente 135 entraînements. Elle exige donc une
confirmation explicite :

```bash
.venv/bin/python scripts/generer_color_map.py \
  --nom-experience grille_dense_v1 \
  --confirmer-grande-grille
```

Chaque résultat est écrit immédiatement dans le CSV brut. Une interruption ne
fait pas perdre les entraînements déjà terminés : relance la même commande pour
reprendre. Pour refaire seulement les graphiques :

```bash
.venv/bin/python scripts/generer_color_map.py \
  --nom-experience grille_dense_v1 \
  --plot-only
```

Pour obtenir une surface plus détaillée autour du meilleur dropout observé,
le preset suivant teste 7 valeurs de filtres sur chaque axe, uniquement avec
`dropout=0.4`. Avec trois répétitions, il planifie 147 entraînements :

```bash
.venv/bin/python scripts/generer_color_map.py \
  --nom-experience surface_dense_dropout_04 \
  --preset-surface-dropout-04 \
  --confirmer-grande-grille
```

Les points mesurés restent visibles sur la surface ; le lissage entre eux est
une interpolation linéaire sans extrapolation.

Ne compare pas des différences d'accuracy minuscules sur une seule exécution :
les répétitions et l'écart-type servent précisément à estimer cette variabilité.

## 5. Ordre pratique recommandé

1. Photographier et ranger les nouvelles feuilles terrain.
2. Exécuter le test terrain et inspecter les planches 28×28.
3. Corriger le prétraitement si les chiffres sont coupés ou mal centrés.
4. Produire les chiffres moyens et les superpositions.
5. Lancer la petite grille d'hyperparamètres, puis seulement la grille dense.
6. Importer un petit lot Claude, le valider visuellement, puis augmenter sa taille.
7. Comparer séparément MNIST seul, augmentation classique et données IA sur le
   même jeu de test terrain.
