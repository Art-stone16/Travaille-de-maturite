# Vue d'ensemble des graphiques

Inventaire des fichiers présents dans `resultats/` au **6 octobre 2026** :
**5 177 images PNG/JPG**. Ce total compte les fichiers, y compris les essais
successifs, les diagnostics et une image portant le suffixe « copie » ; il ne
correspond pas à 5 177 graphiques ou mesures indépendantes. Les photographies
sources de `donnees/`, les archives et les rapports Markdown sont hors de ce
périmètre.

| Emplacement dans `resultats/` | Images | Contenu |
|---|---:|---|
| `recherche/hyperparametres/` | 27 | Cartes, surfaces, nuages de points, corrélations et histogrammes |
| `recherche/stabilite/` | 4 | Comparaison ReLU/softmax avec plusieurs graines |
| `recherche/entrainements/` | 8 | Courbes des huit modèles actifs |
| `visualisations/chiffres_moyens/` | 41 | 31 visualisations analytiques et 10 entrées personnelles 28×28 |
| `visualisations/motifs_classes/` | 9 | Sept planches de modèles, une comparaison et une légende |
| `visualisations/exemples_mnist/` | 1 | Planche d'exemples MNIST de 0 à 9 |
| `photos_terrain/` | 4 484 | Prétraitement, diagnostics et résultats de 40 exécutions |
| `cascade_top_n/` | 594 | Diagnostics et résultats de 97 exécutions, plus 12 planches comparatives |
| `webcam/` | 9 | Captures annotées |
| **Total** | **5 177** | |

## À ouvrir en premier

1. [Heatmaps des hyperparamètres](../resultats/recherche/hyperparametres/limite_128_criblage/graphiques/heatmaps_2d_accuracy_moyenne.png) : vue générale des configurations testées.
2. [Compromis coût–performance](../resultats/recherche/hyperparametres/limite_128_criblage/graphiques/correlations/scatter_cout_performance.png) : précision en regard du coût de calcul.
3. [Accuracy moyenne et IC 95 %](../resultats/recherche/stabilite/comparaison_relu_softmax_16_32/graphiques/accuracy_moyenne_ic95.png) : stabilité des activations.
4. [Comparaisons appariées par graine](../resultats/recherche/stabilite/comparaison_relu_softmax_16_32/graphiques/trajectoires_appariees_par_seed.png) : évolution entre activations à graine identique.
5. [Courbes Best_COLOR_MAP](../resultats/recherche/entrainements/courbes/Best_COLOR_MAP_training_curves.png) : suivi de l'apprentissage.
6. [Moyennes et écarts-types MNIST](../resultats/visualisations/chiffres_moyens/comparaison_tous_mes_chiffres_2026-08-07/figures/moyennes_et_ecarts_types_mnist.png) : référence visuelle des dix classes.
7. [Synthèse Cascade Top-N](../rapports/comparaisons/cascade_top_n/comparaison_modeles.md) : résultats sauvegardés des huit modèles sur les feuilles CTN.

## Recherche des hyperparamètres : 27 images

Trois expériences sont présentes : `limite_128_criblage` (9 images),
`limite_128_criblage_mnist_cascade_top_n_v1` (9 images) et
`surface_dense_dropout_04` (9 images). La dernière inclut
`activation_relu copie.png`, conservée comme fichier supplémentaire ; son nom ne
constitue pas une preuve d'une expérience supplémentaire.

| Forme | Fichiers | Utilité |
|---|---:|---|
| Heatmaps 2D | 3 | Comparer `filter_1` × `filter_2` pour chaque dropout et activation |
| Surfaces 3D détaillées | 6 | Montrer les tendances dans l'espace des filtres ; ce compte inclut l'image « copie » |
| Nuages de points 3D | 3 | Situer les configurations dans l'espace filtres–dropout |
| Nuages accuracy–hyperparamètres | 3 | Examiner les associations descriptives |
| Nuages coût–performance | 3 | Examiner le compromis précision/coût |
| Matrices de corrélation de Spearman | 3 | Décrire les associations monotones |
| Histogrammes d'accuracy | 3 | Montrer la distribution des scores |
| Histogrammes stabilité–durée | 3 | Montrer la variabilité et les durées |
| **Total** | **27** | |

Les heatmaps permettent les comparaisons les plus directes. Les surfaces 3D
sont complémentaires : la perspective rend les petites différences difficiles
à apprécier. Les corrélations sont descriptives et ne démontrent pas un effet
causal. Comparer les datasets et protocoles avant de rapprocher deux expériences.

## Stabilité et apprentissage : 12 images

Les quatre graphiques de stabilité montrent le niveau moyen avec intervalle de
confiance, la distribution des accuracies, les comparaisons appariées par graine
et les courbes moyennes au fil des époques. Ils répondent à des questions
différentes. Le chevauchement de deux intervalles ne suffit pas à conclure à
l'absence de différence.

Les huit fichiers de `resultats/recherche/entrainements/courbes/` présentent
l'accuracy et la loss d'entraînement/validation de chacun des modèles actifs.
Ils permettent d'examiner la convergence et l'écart entre entraînement et
validation ; ils ne remplacent pas les tests sur un jeu indépendant.

## Chiffres moyens et motifs : 51 images

La comparaison des écritures personnelles contient 31 visualisations : une
planche globale, dix moyennes, dix écarts-types à échelle commune et dix
comparaisons personnelles. Dix fichiers `image_28x28.png` supplémentaires sont
les entrées prétraitées utilisées pour ces comparaisons.

Le dossier `motifs_classes/` contient sept planches, la comparaison des sept
modèles et une légende de couleurs. Les motifs proviennent des gradients sur
des images MNIST ; ce ne sont pas les filtres bruts des convolutions. La
planche `exemples_mnist/planche_mnist_0_a_9.png` montre dix exemples réels du jeu
d'entraînement.

Une ressemblance avec le chiffre moyen décrit une forme. Elle ne mesure pas
l'accuracy du classificateur et ne suffit pas à expliquer sa décision.

## Diagnostics des photographies : 5 078 images

| Catégorie | Images | Unité comptée |
|---|---:|---|
| Prétraitement terrain | 4 244 | Quatre images de contrôle pour chacune des 1 061 zones détectées |
| Diagnostics de détection terrain | 200 | Cinq images pour chacune des 40 exécutions |
| Résultats terrain annotés | 40 | Une image par exécution |
| Diagnostics de détection CTN | 485 | Cinq images pour chacune des 97 exécutions |
| Résultats CTN annotés | 97 | Une image par exécution |
| Comparaisons CTN des huit modèles | 12 | Une planche par feuille, variantes comprises |
| **Total** | **5 078** | |

Ces fichiers servent à localiser une erreur de détection ou de prétraitement.
Les neuf captures de webcam s'ajoutent à cet ensemble dans un dossier séparé.
Les nombres ci-dessus incluent toutes les exécutions conservées, alors que la
synthèse Cascade Top-N retient la dernière exécution de chaque couple
modèle/feuille.

Les scores Top-N portent sur les **zones détectées**. Ils ne donnent pas
directement le taux de reconnaissance de tous les vrais chiffres de la feuille.
Un écart de comptage ne correspond pas, à lui seul, au nombre de faux positifs
ou de chiffres manqués. La méthode est détaillée dans le
[comparatif Cascade Top-N](../rapports/comparaisons/cascade_top_n/comparaison_modeles.md).

## Figures pour le travail écrit

Sélectionner les vues qui soutiennent directement le raisonnement du TM :
heatmap, compromis coût–performance, stabilité avec IC 95 %, comparaison
appariée, courbes d'apprentissage et tableau Cascade Top-N. Ajouter une planche
MNIST ou un diagnostic terrain si la discussion porte sur la forme des chiffres
ou sur le prétraitement.

Les fichiers choisis peuvent être copiés dans `rapports/figures/`, et les
tableaux dans `rapports/tableaux/`, en conservant les résultats détaillés dans
`resultats/`. Pour régénérer les synthèses Markdown :

```bash
.venv/bin/python scripts/visualisation/generer_rapports_cascade_md.py
```
