# Structure du projet

Le projet distingue trois types de contenu :

- `donnees/` contient les entrées brutes et les datasets préparés ;
- `modeles/` contient les modèles Keras ;
- `sorties/` contient uniquement les résultats d'expériences et les figures.

Les nouvelles expériences ne remplacent jamais un dossier existant. Elles
enregistrent aussi leur provenance, leurs paramètres et un manifeste ou un CSV.

## Arborescence principale

```text
.
├── donnees/
│   ├── tests_terrain_a_analyser/     nouvelles photographies terrain
│   ├── ecritures_personnelles/       chiffres isolés pour les superpositions
│   ├── synthetiques_a_importer/      fichiers bruts reçus de Claude ou ailleurs
│   └── datasets_synthetiques/        datasets 28×28 validés par le pipeline
├── modeles/
│   ├── modeles_valides/
│   └── recherche_architectures/
├── scripts/
└── sorties/
    ├── tests_condition_reelle/       QC 28×28 et résultats terrain
    ├── analyse_chiffres_moyens/      moyennes, écarts-types et superpositions
    ├── experiences_hyperparametres/  CSV, heatmaps 2D et cartographies 3D
    ├── cascade_top_n/
    ├── graphiques/
    └── matrices_confusion/
```

Les photographies historiques placées directement dans `donnees/` sont
conservées pour ne casser aucun ancien script. Les nouvelles photographies
doivent être rangées dans les sous-dossiers indiqués ci-dessus.

## Scripts principaux

- `scripts/detection_chiffres.py` : détection et prétraitement communs.
- `scripts/test_condition_reelle.py` : test terrain, planches QC, matrices 0/1,
  CSV détaillé et journal cumulatif.
- `scripts/analyser_chiffres_moyens.py` : moyenne et écart-type MNIST par classe,
  puis comparaison avec une écriture personnelle.
- `scripts/preparer_dataset_synthetique.py` : import, génération de baseline,
  augmentation, validation et visualisation de datasets 28×28.
- `scripts/generer_color_map.py` : recherche configurable avec répétitions,
  résultats reprenables, heatmaps 2D, nuage 3D et surfaces 3D par dropout.
- `scripts/train_modele_principal.py` : entraînement du modèle principal.
- `scripts/recherche_architectures.py` : recherche historique d'architectures.
- `scripts/matrice_confusion.py` : matrices de confusion MNIST.
- `scripts/test_stabilite.py` : répétitions d'entraînement et stabilité.
- `scripts/test_cascade_top_n.py` : classement Top-N sur une feuille répétée.

## Repérage dans chaque nouvelle expérience

### Test terrain

```text
sorties/tests_condition_reelle/<experience>/<image>/<date_heure>/
├── 00_source/parametres.json
├── 01_detection/
├── 02_pretraitement/chiffre_NNN/
│   ├── 01_recadrage_original.png
│   ├── 02_entree_modele_28x28_gris.png
│   ├── 03_matrice_28x28_binaire.png
│   ├── 04_matrice_28x28_0_1.csv
│   └── planche_controle_qualite.png
├── 03_resultats/
│   ├── resultat_annote.jpg
│   ├── resultats_terrain.csv
│   └── resume.txt
└── README.txt
```

### Chiffres moyens

```text
sorties/analyse_chiffres_moyens/<experience>/
├── manifest.json
├── statistiques/
├── figures/
├── personnels/
└── rapports/
```

### Hyperparamètres

```text
sorties/experiences_hyperparametres/<experience>/
├── configuration.json
├── catalogue_sorties.csv
├── donnees/
│   ├── resultats_bruts.csv
│   └── resultats_agreges.csv
└── graphiques/
    ├── heatmaps_2d_accuracy_moyenne.png
    ├── nuage_3d_accuracy_moyenne.png
    └── surfaces_3d_accuracy_par_dropout.png
```

### Dataset synthétique préparé

```text
donnees/datasets_synthetiques/<dataset>/
├── manifest.json
├── manifest.csv
├── matrices/<classe>/
├── images/<classe>/
├── apercus/
└── rapports/
```

## Commandes

Les commandes prêtes à copier et le protocole recommandé sont dans
[`GUIDE_EXPERIENCES.md`](GUIDE_EXPERIENCES.md).

Toujours lancer les scripts depuis la racine avec `.venv/bin/python`.
