# Reconnaissance de chiffres manuscrits

Travail de maturité consacré à la reconnaissance des chiffres de 0 à 9 avec un
réseau de neurones convolutif. Le projet permet d'entraîner et comparer des
modèles, d'étudier leurs hyperparamètres, puis de les évaluer sur MNIST et sur
des photographies réelles.

## Installation

Depuis la racine du dépôt :

```bash
git clone https://github.com/Art-stone16/Travaille-de-maturite.git
cd Travaille-de-maturite
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

L'environnement de référence utilise Python 3.11.9. Les dépendances de
`requirements.txt` ne sont pas verrouillées à une version exacte. L'installation
éditable rend le package `reconnaissance_chiffres` importable. Les commandes
`python scripts/<categorie>/<script>.py` fonctionnent aussi directement depuis
ce dépôt grâce à leur initialisation commune.

## Organisation du dépôt

```text
.
├── src/reconnaissance_chiffres/    code commun : chemins, données, prétraitement, modèles, rapports
├── scripts/
│   ├── entrainement/              entraînement des modèles
│   ├── evaluation/                MNIST, photographies et cascade Top-N
│   ├── experiences/               hyperparamètres, architectures et stabilité
│   ├── preparation_donnees/       extraction et validation des datasets
│   ├── visualisation/             planches, motifs et rapports Markdown
│   └── webcam/                    reconnaissance en direct
├── donnees/
│   ├── brutes/                    sources conservées sans transformation
│   └── preparees/                 datasets générés et validés localement
├── modeles/actifs/                les huit modèles utilisés actuellement
├── resultats/                     résultats détaillés et diagnostics
├── rapports/
│   ├── comparaisons/              scripts, modèles, cascade et terrain en Markdown
│   ├── figures/                   figures sélectionnées pour le TM
│   └── tableaux/                  tableaux sélectionnés pour le TM
├── docs/                          guides et inventaire des graphiques
├── configurations/               emplacement des configurations partagées
├── tests/                         contrôles automatiques du code
├── archives/                      modèles, résultats et documents historiques
└── tmp/                           fichiers temporaires locaux
```

Tous les modèles actifs sont regroupés sous `modeles/actifs/<nom_modele>/`.
Les anciens modèles restent dans `archives/modeles/`. Le package commun contient
`config.py`, `detection.py`, `pretraitement.py`, `datasets.py`, `modeles.py` et
`rapports.py` ; les scripts sont les points d'entrée des commandes. `scripts/_bootstrap.py` initialise leur accès au package.

Les photographies et les lots sources se rangent dans :

- `donnees/brutes/terrain/` : photographies libres, regroupées par protocole ;
- `donnees/brutes/cascade_top_n/` : feuilles CTN contenant un chiffre répété ;
- `donnees/brutes/ecritures_personnelles/` : images contenant un seul chiffre ;
- `donnees/brutes/synthetiques/` : lots bruts générés artificiellement.

## Commandes principales

Les chemins des modèles peuvent être fournis avec `--modele` aux commandes
d'évaluation. Les valeurs par défaut restent propres à chaque script :
`best_relu_10xcascade` pour le test terrain, `Best_relu_cascade_V2` pour Cascade
Top-N et `Best_COLOR_MAP` pour la matrice de confusion.

| Objectif | Script |
|---|---|
| Tester une photographie et inspecter le 28×28 | `scripts/evaluation/test_condition_reelle.py` |
| Tester une feuille contenant un chiffre répété | `scripts/evaluation/test_cascade_top_n.py` |
| Évaluer les performances par chiffre sur MNIST | `scripts/evaluation/matrice_confusion.py` |
| Explorer les hyperparamètres | `scripts/experiences/generer_color_map.py` |
| Recréer corrélations et histogrammes | `scripts/experiences/analyser_hyperparametres.py` |
| Mesurer la stabilité statistique | `scripts/experiences/test_stabilite.py` |
| Entraîner un modèle avec les données de cascade | `scripts/entrainement/train_modele_principal.py` |
| Préparer un dataset de cascade | `scripts/preparation_donnees/preparer_dataset_cascade.py` |
| Préparer un dataset synthétique | `scripts/preparation_donnees/preparer_dataset_synthetique.py` |
| Calculer les chiffres moyens MNIST | `scripts/visualisation/analyser_chiffres_moyens.py` |
| Régénérer les synthèses Cascade Top-N | `scripts/visualisation/generer_rapports_cascade_md.py` |
| Reconnaître les chiffres avec la webcam | `scripts/webcam/reconnaissance_webcam.py` |

Afficher le plan de l'étude jusqu'à 128 filtres sans lancer d'entraînement :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience plan_limite_128 \
  --preset-limite-128 \
  --dry-run
```

Tester une photographie existante avec un modèle explicitement choisi :

```bash
.venv/bin/python scripts/evaluation/test_condition_reelle.py \
  --image donnees/brutes/terrain/historiques/test_terrain.jpg \
  --modele modeles/actifs/Best_COLOR_MAP/best_model.keras \
  --nom-experience essai_photo
```

Comparer deux écritures personnelles avec les moyennes MNIST :

```bash
.venv/bin/python scripts/visualisation/analyser_chiffres_moyens.py \
  --source entrainement \
  --nom-experience comparaison_personnelle \
  --personnel 2=donnees/brutes/ecritures_personnelles/mon_2.JPG \
  --personnel 7=donnees/brutes/ecritures_personnelles/mon_7.JPG
```

L'option `--personnel` peut être répétée. Pour recréer les graphiques d'une
expérience déjà présente, utiliser `--plot-only` :

```bash
.venv/bin/python scripts/experiences/generer_color_map.py \
  --nom-experience limite_128_criblage \
  --plot-only
```

## Résultats et rapports

```text
resultats/
├── recherche/
│   ├── hyperparametres/
│   ├── stabilite/
│   ├── architectures/
│   └── entrainements/
├── evaluation_mnist/performances_par_chiffre/
├── visualisations/
│   ├── chiffres_moyens/
│   ├── exemples_mnist/
│   └── motifs_classes/
├── photos_terrain/
│   ├── journal_global_tests_terrain.csv
│   └── <nom_image>/<date_heure>/
├── cascade_top_n/<nom_feuille>/<date_heure>/
└── webcam/
```

Tous les essais terrain d'une même photographie sont regroupés sous son nom.
Le nom d'expérience et celui du modèle sont conservés dans les paramètres et
le journal global. Les diagnostics et les résultats bruts restent dans
`resultats/` ; les synthèses Cascade Top-N se trouvent dans
[rapports/comparaisons/cascade_top_n/comparaison_modeles.md](rapports/comparaisons/cascade_top_n/comparaison_modeles.md).
Les figures et tableaux choisis pour le document final peuvent être copiés dans
`rapports/figures/` et `rapports/tableaux/`.

La cartographie des hyperparamètres cherche des configurations prometteuses ;
l'étude de stabilité les compare ensuite avec plusieurs graines. Les surfaces
3D détaillées sont dans
`resultats/recherche/hyperparametres/<experience>/graphiques/surfaces_3d_detaillees/`.

## Tests du code

Le dossier `tests/` contient 40 contrôles rapides fondés sur de petites données
synthétiques, des simulations et des dossiers temporaires. Ils vérifient le code
et les calculs, sans mesurer l'accuracy réelle des modèles.

| Fichier | Rôle | Nombre de tests |
|---|---|---:|
| `tests/test_hyperparametres.py` | Protocoles, reprises, agrégations et graphiques | 15 |
| `tests/test_matrice_confusion.py` | TP/FN/FP/TN, métriques et intervalles de Wilson | 10 |
| `tests/test_stabilite_statistique.py` | Graines, intervalles de Student, splits et reprises | 15 |

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## Reproductibilité et publication Git

Git fournit les fichiers présents dans le dernier commit. `.gitignore` autorise
les modèles `.keras` de `modeles/` et `archives/modeles/` : les modèles actifs,
les nouveaux scripts et les résultats souhaités doivent être ajoutés au dépôt
avant sa publication. La présence d'un fichier dans le workspace ne suffit pas
à le rendre disponible après un clone.

Les environnements, caches, fichiers temporaires et datasets préparés de
cascade ou synthétiques restent locaux. Les quatre scripts d'entraînement avec
cascade nécessitent les tableaux sous
`donnees/preparees/cascade/cascade_top_n_v1/dataset_numpy/`. Il faut disposer de
ce dataset ou le reconstruire et le valider avant l'entraînement. Le script
`train_best_color_map_cascade.py` exige aussi le fichier MNIST dans
`.cache/keras/datasets/mnist.npz`. Les étapes sont décrites dans le
[guide des expériences](docs/guide_experiences.md#7-prérequis-des-entraînements-avec-cascade).

Les expériences enregistrent notamment `configuration.json`,
`resultats_bruts.csv`, `resultats_agreges.csv` et `catalogue_sorties.csv`.
Les graines, paramètres et sauvegardes progressives permettent de comparer les
configurations et de reprendre les essais compatibles.

## Documentation

- [Guide des expériences](docs/guide_experiences.md)
- [Vue d'ensemble des graphiques](docs/vue_ensemble_graphiques.md)
- [Comparatif des scripts et modules](rapports/comparaisons/comparatifs_scripts.md)
- [Comparatif des modèles actifs](rapports/comparaisons/comparatifs_modeles.md)
- [Format des écritures personnelles](donnees/brutes/ecritures_personnelles/README.md)
- [Préparation des datasets synthétiques](scripts/preparation_donnees/preparer_dataset_synthetique.py)
- [Préparer les photographies terrain](donnees/brutes/terrain/README.md)
- [Contenu des archives](archives/README.md)
