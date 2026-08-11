# Reconnaissance de chiffres manuscrits

Projet de travail de maturité consacré à la reconnaissance des chiffres de 0 à
9 avec un réseau de neurones convolutif. Le dépôt permet d'entraîner et comparer
des modèles, d'étudier leurs hyperparamètres, puis de les tester sur MNIST et
sur de vraies photographies.

## Fonctionnalités principales

- détection de plusieurs chiffres dans une photographie ;
- contrôle visuel du prétraitement réellement envoyé au modèle en 28×28 ;
- comparaison de l'écriture personnelle avec le chiffre moyen de MNIST ;
- exploration des filtres, du dropout et des activations ReLU/Softmax ;
- étude de stabilité avec plusieurs graines et intervalles de confiance ;
- matrice de confusion, sensibilité et spécificité pour chaque chiffre ;
- import, validation et augmentation de datasets synthétiques ;
- grilles d'hyperparamètres et études de stabilité enregistrées dans des
  expériences séparées et reprenables.

Le modèle utilisé dans l'environnement local se trouve dans
`modeles/modeles_valides/Best_COLOR_MAP/best_model.keras`. Les fichiers
`.keras` ne sont pas versionnés sur GitHub en raison de leur taille. Après un
nouveau clone, il faut donc placer un modèle compatible avec les dix classes à
ce chemin ou fournir explicitement son chemin avec l'option `--modele` des
scripts concernés.

## Installation

```bash
git clone https://github.com/Art-stone16/Travaille-de-maturite.git
cd Travaille-de-maturite

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les commandes de ce README doivent être lancées depuis la racine du dépôt.
L'environnement a été vérifié avec Python 3.11.9. Les dépendances de
`requirements.txt` ne sont pas encore verrouillées à une version exacte.

## Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `donnees/` | Photographies, écritures personnelles et données synthétiques à importer |
| `modeles/` | Destination locale des modèles validés ou en cours |
| `scripts/` | Entraînement, analyse, évaluation et préparation des données |
| `sorties/` | Destination locale des résultats générés, classés par objectif |
| `Sécurité/` | Contrôles automatiques rapides du code et des calculs |
| `Archives/` | Modèles et résultats historiques volontairement conservés |

Un clone GitHub fournit le code, sa documentation et les photographies de test
du projet, y compris les exemples d'écriture personnelle. Les modèles `.keras`
et la majorité des résultats générés restent locaux ; ils doivent être produits
ou ajoutés séparément.

Les photographies ne doivent pas être déposées directement à la racine de
`donnees/` :

- `donnees/tests_terrain_a_analyser/` pour les tests sur une photo libre ;
- `donnees/tests_cascade_top_n/` pour les feuilles CTN ;
- `donnees/ecritures_personnelles/` pour une image contenant un seul chiffre ;
- `donnees/synthetiques_a_importer/` pour un lot brut généré artificiellement.

## Scripts principaux

| Objectif | Script |
|---|---|
| Tester une photographie et inspecter le 28×28 | `scripts/test_condition_reelle.py` |
| Tester une feuille contenant un chiffre répété | `scripts/test_cascade_top_n.py` |
| Calculer les chiffres moyens MNIST | `scripts/analyser_chiffres_moyens.py` |
| Explorer les hyperparamètres | `scripts/generer_color_map.py` |
| Recréer corrélations et histogrammes | `scripts/analyser_hyperparametres.py` |
| Mesurer la stabilité statistique | `scripts/test_stabilite.py` |
| Évaluer les performances par chiffre | `scripts/matrice_confusion.py` |
| Préparer un dataset synthétique | `scripts/preparer_dataset_synthetique.py` |
| Entraîner manuellement un modèle | `scripts/train_modele_principal.py` |

`scripts/env_config.py` centralise les chemins du projet. Ce n'est pas un
programme à lancer directement.

## Premières commandes

Afficher le plan de l'étude jusqu'à 128 filtres, sans créer de résultat, charger
TensorFlow ni lancer d'entraînement :

```bash
.venv/bin/python scripts/generer_color_map.py \
  --nom-experience limite_128_criblage \
  --preset-limite-128 \
  --dry-run
```

Les exemples suivants nécessitent les fichiers locaux indiqués.

Tester une photographie terrain :

```bash
.venv/bin/python scripts/test_condition_reelle.py \
  --image donnees/tests_terrain_a_analyser/historiques/test_terrain.jpg \
  --nom-experience essai_photo
```

Comparer deux écritures personnelles avec les moyennes MNIST :

```bash
.venv/bin/python scripts/analyser_chiffres_moyens.py \
  --source entrainement \
  --nom-experience comparaison_personnelle \
  --personnel 2=donnees/ecritures_personnelles/mon_2.JPG \
  --personnel 7=donnees/ecritures_personnelles/mon_7.JPG
```

L'option `--personnel` peut être répétée pour ajouter les autres chiffres.

Lorsqu'une expérience existe déjà localement, ses graphiques peuvent être
recréés sans entraînement :

```bash
.venv/bin/python scripts/generer_color_map.py \
  --nom-experience limite_128_criblage \
  --plot-only
```

Les protocoles complets et les commandes pour les expériences longues sont
réunis dans [GUIDE_EXPERIENCES.md](GUIDE_EXPERIENCES.md).

## Résultats

Les sorties actives sont séparées en trois catégories :

```text
sorties/
├── 01_RECHERCHE_MODELE/
│   ├── 01_HYPERPARAMETRES/
│   ├── 02_STABILITE/
│   ├── 03_RECHERCHE_ARCHITECTURES/
│   └── 04_ENTRAINEMENTS_MANUELS/
├── 02_EVALUATION_MODELE/
│   ├── 01_PERFORMANCES_PAR_CHIFFRE/
│   └── 02_CHIFFRES_MOYENS/
└── 03_TESTS_PHOTOS/
    ├── 01_TESTS_TERRAIN/
    └── 02_CASCADE_TOP_N/
```

La cartographie des hyperparamètres cherche les meilleures configurations. Une
étude de stabilité répète ensuite quelques configurations avec plusieurs
graines pour vérifier que leur résultat est reproductible : ce sont deux étapes
différentes.

Pour les surfaces 3D, le résultat principal est le format détaillé situé dans
`sorties/01_RECHERCHE_MODELE/01_HYPERPARAMETRES/<experience>/graphiques/surfaces_3d_detaillees/`.
Les images ReLU et Softmax représentent deux activations réellement
différentes, et non deux copies du même résultat.

## Sécurité du code

Le dossier `Sécurité/` contient 40 tests automatiques rapides. Ils utilisent de petites
données synthétiques, des simulations et des dossiers temporaires. Ils ne
lancent aucun entraînement long, aucune grille réelle et ne modifient pas les
résultats présents dans `sorties/`.

| Fichier | Rôle | Nombre de tests |
|---|---|---:|
| `Sécurité/test_hyperparametres.py` | Vérifie les protocoles, identifiants, reprises, agrégations et graphiques des expériences d'hyperparamètres | 15 |
| `Sécurité/test_matrice_confusion.py` | Vérifie TP/FN/FP/TN, sensibilité, précision, F1, normalisation et intervalles de Wilson | 10 |
| `Sécurité/test_stabilite_statistique.py` | Vérifie les graines, intervalles de Student, comparaisons appariées, splits et reprises de l'étude de stabilité | 15 |

Pour lancer les trois fichiers :

```bash
.venv/bin/python -m unittest discover -s 'Sécurité' -p 'test_*.py' -v
```

Un résultat `ok` signifie que la règle vérifiée fonctionne toujours. Un résultat
`FAILED` indique généralement qu'une modification du code a cassé un calcul,
un format de fichier ou une règle de reprise. Ces tests contrôlent le logiciel ;
ils ne mesurent pas l'accuracy réelle du modèle.

## Reproductibilité des expériences

Les nouveaux workflows produisent notamment :

- `configuration.json` : paramètres, provenance et identité du protocole ;
- `resultats_bruts.csv` : une ligne par configuration et par graine ;
- `resultats_agreges.csv` : moyenne, écart-type et intervalle de confiance ;
- `catalogue_sorties.csv` : rôle et présence des fichiers produits.

Une graine fixe les tirages pseudo-aléatoires d'un entraînement. Réutiliser les
mêmes graines permet de comparer deux configurations dans les mêmes conditions.
Les CSV sont sauvegardés progressivement afin qu'une expérience interrompue
puisse être reprise sans recommencer les essais déjà validés.

## Documentation spécialisée

- [Guide des expériences](GUIDE_EXPERIENCES.md)
- [Format des écritures personnelles](donnees/ecritures_personnelles/README.md)
- [Importer un dataset synthétique](donnees/synthetiques_a_importer/README.md)
- [Préparer les photographies terrain](donnees/tests_terrain_a_analyser/README.md)
- [Contenu des archives](Archives/README.md)

Les éléments placés dans `Archives/` sont les anciens modèles et résultats que
tu as choisi de conserver. Les scripts actifs n'écrivent jamais dans ce dossier.
