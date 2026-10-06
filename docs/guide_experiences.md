# Guide des expériences

Ce guide donne l'ordre recommandé et les commandes à lancer depuis la racine du
projet. Les scripts affichent le chemin exact du dossier créé à la fin.

Après la réorganisation du code, une reprise d'entraînement d'une ancienne
expérience peut être refusée : le protocole vérifie notamment l'empreinte du
script, qui a changé avec les imports et les chemins. Pour un nouvel
entraînement, choisir un nouveau `--nom-experience`. Le mode `--plot-only`
reste utilisable pour les résultats existants ; leurs mesures et identifiants
historiques sont conservés.

## 1. Tester une photographie terrain et contrôler le 28×28

Place d'abord les nouvelles photos dans
`donnees/brutes/terrain/<nom_du_protocole>/`.

Exemple pour une feuille contenant uniquement des 7 :

```bash
.venv/bin/python scripts/evaluation/test_condition_reelle.py \
  --image donnees/brutes/terrain/papier_blanc_stylo_noir/classe_7.jpg \
  --chiffre-reel 7 \
  --nom-experience papier_blanc_stylo_noir
```

Le script crée une planche QC et une matrice CSV 0/1 pour chaque chiffre. Il
ajoute aussi les résultats au fichier
`resultats/photos_terrain/journal_global_tests_terrain.csv`.
Les essais sont rangés sous `<nom_image>/<date_heure>/` ; le nom de
l’expérience reste enregistré dans les paramètres et le journal.

Utilise `--chiffre-reel inconnu` pour une image dont la vérité terrain n'est pas
commune à tous les chiffres. `--chiffre-reel auto` n'est fiable que si le nom du
fichier contient un unique chiffre isolé, par exemple `classe_7.jpg`.

Protocole conseillé : même papier blanc, même stylo noir, même distance, lumière
homogène, au moins trois écritures par classe, et conservation de tous les
résultats — y compris les erreurs.

## 2. Calculer les chiffres moyens MNIST

Analyse complète du jeu d'entraînement :

```bash
.venv/bin/python scripts/visualisation/analyser_chiffres_moyens.py \
  --source entrainement \
  --nom-experience mnist_entrainement_complet
```

Avec deux écritures personnelles isolées :

```bash
.venv/bin/python scripts/visualisation/analyser_chiffres_moyens.py \
  --source entrainement \
  --nom-experience comparaison_ecriture_personnelle \
  --personnel 2=donnees/brutes/ecritures_personnelles/mon_2.JPG \
  --personnel 7=donnees/brutes/ecritures_personnelles/mon_7.JPG
```

Une image fournie avec `--personnel` doit contenir un seul chiffre. Les mesures
MAE, RMSE et corrélation sont descriptives : elles ne remplacent pas l'accuracy
du classificateur.

## 3. Préparer un dataset généré par Claude

Le modèle de demande se trouve dans
`donnees/brutes/synthetiques/PROMPT_CLAUDE.md`. Enregistre la réponse JSON,
sans la modifier silencieusement, dans ce même dossier.

Importer et contrôler le lot :

```bash
.venv/bin/python scripts/preparation_donnees/preparer_dataset_synthetique.py importer \
  --entree donnees/brutes/synthetiques/lot_claude_01.json \
  --nom-dataset claude_lot_01 \
  --origine claude \
  --description "Premier lot Claude, plusieurs styles manuscrits" \
  --binaire
```

Revalider plus tard :

```bash
.venv/bin/python scripts/preparation_donnees/preparer_dataset_synthetique.py valider \
  --dataset donnees/preparees/synthetiques/claude_lot_01
```

Le pipeline accepte aussi les images, NPY, NPZ, CSV et TXT. Les fichiers bruts
restent dans `donnees/brutes/synthetiques/` ; les matrices validées vont dans
`donnees/preparees/synthetiques/`. Les données IA ne sont donc jamais mélangées
silencieusement avec MNIST.

Pour disposer d'un témoin non-IA :

```bash
.venv/bin/python scripts/preparation_donnees/preparer_dataset_synthetique.py generer-baseline \
  --nom-dataset baseline_procedural_01 \
  --par-classe 20
```

Cette baseline utilise les polices OpenCV et est explicitement identifiée comme
procédurale, pas comme manuscrite ni générée par IA.

## 4. Tester les cartographies sans entraînement long

Afficher d'abord le plan exact, sans créer de fichiers :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience grille_dense_v2 \
  --dry-run
```

Valider le pipeline avec une petite grille :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience validation_rapide_v2 \
  --preset-rapide
```

La grille dense par défaut représente 135 entraînements. Elle exige donc une
confirmation explicite :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience grille_dense_v2 \
  --confirmer-grande-grille
```

Chaque résultat est écrit immédiatement dans le CSV brut. Une interruption ne
fait pas perdre les entraînements déjà terminés : relance la même commande pour
reprendre. Pour refaire seulement les graphiques :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience limite_128_criblage \
  --plot-only
```

Pour obtenir une surface plus détaillée autour du meilleur dropout observé,
le preset suivant teste 7 valeurs de filtres sur chaque axe, uniquement avec
`dropout=0.4`. Avec trois répétitions, il planifie 147 entraînements :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience surface_dense_dropout_04_v2 \
  --preset-surface-dropout-04 \
  --confirmer-grande-grille
```

Les points mesurés restent visibles sur la surface ; le lissage entre eux est
une interpolation linéaire sans extrapolation.

Une image détaillée est créée pour chaque activation réellement testée dans
`graphiques/surfaces_3d_detaillees/`. C'est l'unique format de surface 3D actif :
chaque panneau correspond à un dropout, avec des titres et axes suffisamment
espacés pour une lecture à l'écran.

Pour cribler la limite jusqu'à 128 filtres, commence toujours par afficher le
plan. Ce preset teste aussi les deux activations convolutives déjà présentes
dans le projet (`relu` et `softmax`). La même activation est utilisée dans les
deux Conv2D, les noyaux restent fixés à 5×5 puis 5×5, et la couche de sortie
reste toujours en `softmax` :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience plan_limite_128 \
  --preset-limite-128 \
  --dry-run
```

Le plan contient 300 entraînements : 6 valeurs de `filter_1`, 5 de `filter_2`,
5 dropouts, 2 activations et une graine. Après vérification, la commande réelle
est la même sans `--dry-run` et avec `--confirmer-grande-grille`.

Chaque nouvelle ligne v2 porte un `protocol_id` calculé à partir du dataset, de
son empreinte lorsqu'un fichier local est disponible, de la grille, des
activations, des graines et du protocole d'entraînement. Une reprise compatible
est contrôlée avant tout nouvel essai. Une erreur ou un manque de mémoire est
catalogué puis la grille continue ; utilise
`--arreter-sur-erreur` uniquement si tu veux retrouver le comportement strict.

Les scatter plots, histogrammes et tables de corrélation peuvent aussi être
recréés seuls depuis les CSV, sans charger Keras :

```bash
.venv/bin/python scripts/experiences/analyser_hyperparametres.py \
  --nom-experience limite_128_criblage
```

Les corrélations sont calculées au grain « une configuration agrégée », jamais
en considérant artificiellement chaque répétition comme une configuration
indépendante. Elles sont descriptives et ne prouvent pas un lien causal.

Ne compare pas des différences d'accuracy minuscules sur une seule exécution :
les répétitions et l'écart-type servent précisément à estimer cette variabilité.

## 5. Mesurer la stabilité avec des graines contrôlées

`test_stabilite.py` conserve par défaut l'architecture historique : 4 puis 8
filtres, noyaux 5×5 puis 4×4, activation `softmax` dans les deux convolutions et
dropout à 0,3. L'activation de sortie reste toujours `softmax`.

Pour comparer `relu` et `softmax` sur une architecture représentative avec les
mêmes cinq graines, affiche d'abord le plan :

```bash
.venv/bin/python scripts/experiences/test_stabilite.py \
  --nom-experience comparaison_relu_softmax_16_32_v2 \
  --filters-1 16 \
  --filters-2 32 \
  --kernel-1 5 \
  --kernel-2 5 \
  --dropouts 0.4 \
  --activation-conv relu,softmax \
  --seeds 42,43,44,45,46 \
  --dry-run
```

La commande réelle est identique sans `--dry-run`. Elle représente dix
entraînements : deux activations × cinq graines. Toutes les configurations
reçoivent exactement les mêmes graines, et la graine du découpage des données
est enregistrée séparément.

Pour étudier plus finement la zone autour de `dropout=0.4`, cette commande
planifie 9 valeurs × 5 graines, soit 45 entraînements :

```bash
.venv/bin/python scripts/experiences/test_stabilite.py \
  --nom-experience stabilite_dropout_fin_20_48_softmax \
  --filters-1 20 \
  --filters-2 48 \
  --kernel-1 5 \
  --kernel-2 5 \
  --dropouts 0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475,0.50 \
  --activation-conv softmax \
  --seeds 42,43,44,45,46 \
  --dry-run
```

Après cette présélection, augmente le nombre de graines uniquement pour les
meilleures configurations. À partir de 50 runs, le script demande
`--confirmer-grande-etude`.

L'étude historique complète demande 100 entraînements. Elle doit être autorisée
explicitement :

```bash
.venv/bin/python scripts/experiences/test_stabilite.py \
  --nom-experience stabilite_architecture_historique \
  --confirmer-grande-etude
```

Une interruption est reprenable. Les historiques conservent `NaN` après un
arrêt anticipé : aucune époque artificielle n'est ajoutée. Les résultats
agrégés fournissent moyenne, écart-type, erreur standard, intervalle de Student
à 95 %, médiane et quartiles. Pour recréer seulement les tableaux et figures :

```bash
.venv/bin/python scripts/experiences/test_stabilite.py \
  --nom-experience comparaison_relu_softmax_16_32 \
  --plot-only
```

## 6. Calculer les performances de chaque chiffre

Cette commande évalue un modèle déjà entraîné ; elle ne lance aucun
entraînement :

```bash
.venv/bin/python scripts/evaluation/matrice_confusion.py \
  --modele modeles/actifs/Best_COLOR_MAP/best_model.keras \
  --nom-evaluation best_color_map_mnist
```

Le dossier créé dans
`resultats/evaluation_mnist/performances_par_chiffre/` contient les
matrices de confusion, les TP/FN/FP/TN, la sensibilité, la spécificité, la
précision, le F1, les taux FNR/FPR et les intervalles de Wilson à 95 % pour
chaque chiffre. Ajoute `--sauver-predictions` si tu veux aussi conserver les dix
scores de chaque image.

Le script vérifie que le modèle produit exactement dix scores. Le modèle
historique `Best_TEST`, rangé dans
`archives/modeles/modeles_historiques/Best_TEST/`, en produit 20 et est
donc refusé ; `Best_COLOR_MAP` possède bien les dix sorties attendues.

## 7. Prérequis des entraînements avec cascade

Les quatre scripts dans `scripts/entrainement/` utilisent les fichiers
`x_train_cascade.npy` et `y_train_cascade.npy` du dossier
`donnees/preparees/cascade/cascade_top_n_v1/dataset_numpy/`.
`train_best_color_map_cascade.py` lit également `ids_train_cascade.npy`.
Ces datasets préparés sont ignorés par Git : ils existent dans l'environnement
local, mais un clone ne les fournit pas.

Si le dataset n'a pas été fourni séparément, extraire les candidats depuis les
photographies brutes :

```bash
.venv/bin/python scripts/preparation_donnees/preparer_dataset_cascade.py extraire
```

Inspecter les planches dans
`donnees/preparees/cascade/cascade_top_n_v1/planches_controle/`, puis renseigner
les décisions du `manifest.csv` : `inclure` pour les exemples validés,
`exclure` avec une justification pour les autres. Vérifier les étiquettes et
le `split` de chaque exemple. La finalisation refuse les lignes encore
`a_verifier` et les exclusions sans justification.

```bash
.venv/bin/python scripts/preparation_donnees/preparer_dataset_cascade.py finaliser
```

Une nouvelle extraction demande un nouveau contrôle visuel. Pour reproduire
exactement un entraînement existant, conserver aussi le dataset validé et son
manifeste : les décisions de sélection font partie de sa provenance.

Le script `scripts/entrainement/train_best_color_map_cascade.py` exige en plus
MNIST à l'emplacement `.cache/keras/datasets/mnist.npz`, sans téléchargement
automatique. Pour préparer ce cache depuis la racine du dépôt :

```bash
KERAS_HOME="$PWD/.cache/keras" .venv/bin/python -c "import keras; keras.datasets.mnist.load_data()"
```

Les autres scripts d'entraînement utilisent `keras.datasets.mnist.load_data()`.
Les prérequis de données doivent être remplis avant de lancer un entraînement
long. Les tests de modèles déjà entraînés ne nécessitent pas de reconstruire le
dataset de cascade.

## 8. Ordre pratique recommandé

1. Photographier et ranger les nouvelles feuilles terrain.
2. Exécuter le test terrain et inspecter les planches 28×28.
3. Corriger le prétraitement si les chiffres sont coupés ou mal centrés.
4. Produire les chiffres moyens et les superpositions.
5. Lancer la petite grille d'hyperparamètres, puis le criblage jusqu'à 128.
6. Analyser les scatter plots et histogrammes avant de raffiner la grille.
7. Confirmer les finalistes avec plusieurs graines dans l'étude de stabilité.
8. Évaluer le modèle retenu chiffre par chiffre sur MNIST.
9. Importer un petit lot Claude, le valider visuellement, puis augmenter sa taille.
10. Comparer séparément MNIST seul, augmentation classique et données IA sur le
   même jeu de test terrain.
