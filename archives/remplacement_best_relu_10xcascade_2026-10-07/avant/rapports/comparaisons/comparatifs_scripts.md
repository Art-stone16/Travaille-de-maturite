# Comparatif des scripts et modules communs

Les commandes sont classées dans `scripts/` selon leur usage. Les modules
réutilisables sont dans `src/reconnaissance_chiffres/` et s’importent via le
package `reconnaissance_chiffres`. Les liens ci-dessous pointent vers leurs
emplacements actuels.

| Nom | Utilité | Auteurs |
|---|---|---|
| [analyser_chiffres_moyens.py](../../scripts/visualisation/analyser_chiffres_moyens.py) | Calcule les images moyennes et la variabilité des chiffres de MNIST, puis les compare à des écritures personnelles. |OpenAI, 2026: GPT-5.5 |
| [analyser_hyperparametres.py](../../scripts/experiences/analyser_hyperparametres.py) | Analyse les résultats des expériences pour étudier les liens entre hyperparamètres, précision, stabilité et coût, et produit les graphiques correspondants. |OpenAI, 2026: Sol-5.6 |
| [comparer_images_ctn.py](../../scripts/visualisation/comparer_images_ctn.py) | Assemble les images annotées des huit modèles en une planche comparative pour chaque feuille Cascade Top-N. |OpenAI,2026:Sol-6.0 |
| [detection.py](../../src/reconnaissance_chiffres/detection.py) | Fournit les fonctions communes de détection des chiffres dans une image et les diagnostics de détection. |OpenAI,2026: Sol-5.6 |
| [config.py](../../src/reconnaissance_chiffres/config.py) | Centralise les chemins des données, modèles, résultats et caches, ainsi que les réglages d’environnement utilisés par les autres scripts. |OpenAI, 2026:GPT-5.5 |
| [exporter_exemples_mnist.py](../../scripts/visualisation/exporter_exemples_mnist.py) | Exporte une planche contenant un exemple réel de chaque chiffre de 0 à 9 issu du jeu d’entraînement MNIST. |OpenAI, 2026: Astra-6.0 |
| [generer_color_map.py](../../scripts/experiences/generer_color_map.py) | Entraîne et compare des configurations de filtres, dropout et activations sur MNIST, avec ajout possible de données de cascade, puis génère des cartes et surfaces de résultats. |OpenAI, 2026: Sol-5.6 |
| [generer_rapports_cascade_md.py](../../scripts/visualisation/generer_rapports_cascade_md.py) | Génère les résumés Cascade Top-N et le comparatif des modèles en Markdown dans `rapports/comparaisons/cascade_top_n/`. |OpenAI, 2026: Astra-6 |
| [rapports.py](../../src/reconnaissance_chiffres/rapports.py) | Crée des résumés Markdown et un comparatif des modèles à partir des résultats Cascade Top-N enregistrés, avec scores Top-N et principales confusions. |OpenAI, 2026: Astra-6 |
| [matrice_confusion.py](../../scripts/evaluation/matrice_confusion.py) | Évalue un modèle sur le jeu de test MNIST et produit les matrices de confusion, les métriques par chiffre et leurs graphiques. |OpenAi, 2026: GPT-5.5 |
| [preparer_dataset_cascade.py](../../scripts/preparation_donnees/preparer_dataset_cascade.py) | Extrait les chiffres des feuilles Cascade Top-N, permet de vérifier et sélectionner les exemples, puis crée un dataset NumPy validé en 28 × 28 pixels. |OpenAI, 2026: Sol-5.6 |
| [preparer_dataset_synthetique.py](../../scripts/preparation_donnees/preparer_dataset_synthetique.py) | Importe, génère, augmente, valide et visualise des données synthétiques de caractères en 28 × 28 pixels. |OpenAI, 2026: Astra-6 |
| [recherche_architectures.py](../../scripts/experiences/recherche_architectures.py) | Entraîne et compare plusieurs architectures sur MNIST en faisant varier les filtres, kernels, activations et dropout, puis sauvegarde les modèles et un bilan CSV. |Arthur Perret et OpenAI, 2026: GPT-5.5 |
| [reconnaissance_webcam.py](../../scripts/webcam/reconnaissance_webcam.py) | Détecte et reconnaît les chiffres en direct avec la webcam, permet de changer de modèle et de sauvegarder des captures ou des vidéos annotées. |OpenAI, 2026: Sol-5.6 |
| [test_cascade_top_n.py](../../scripts/evaluation/test_cascade_top_n.py) | Teste un modèle sur une ou plusieurs feuilles Cascade Top-N et mesure le rang du chiffre attendu parmi les prédictions. |OpenAI, 2026: Sol-5.6 |
| [test_condition_reelle.py](../../scripts/evaluation/test_condition_reelle.py) | Teste la détection et la reconnaissance sur une photographie terrain et sauvegarde l’image annotée, les contrôles du prétraitement et les résultats détaillés. |OpenAI, 2026: Sol-5.6 |
| [test_stabilite.py](../../scripts/experiences/test_stabilite.py) | Répète les entraînements avec plusieurs graines pour mesurer la stabilité, calculer des intervalles de confiance et comparer des configurations. |OpenAI, 2026: GPT-5.5 |
| [train_best_color_map_cascade.py](../../scripts/entrainement/train_best_color_map_cascade.py) | Entraîne Best_COLOR_MAP_cascade avec l’architecture de Best_COLOR_MAP et les données MNIST complétées par la cascade uniquement dans le jeu d’entraînement. |OpenAI, 2026: Sol_6.1 |
| [train_best_relu_10xcascade.py](../../scripts/entrainement/train_best_relu_10xcascade.py) | Entraîne best_relu_10xcascade sur MNIST et un lot de cascade multiplié par dix grâce aux originaux et à neuf variantes par image. |OpenAI, 2026: Sol_6.1 |
| [train_best_relu_2xcascade.py](../../scripts/entrainement/train_best_relu_2xcascade.py) | Entraîne best_relu_2xcascade sur MNIST et les données de cascade présentes deux fois, avec des copies exactes. |OpenAI, 2026: Sol_6.1 |
| [train_modele_principal.py](../../scripts/entrainement/train_modele_principal.py) | Entraîne le modèle défini dans le script, actuellement Best_relu_cascade_V2, avec MNIST et les données de cascade, puis sauvegarde les modèles et les courbes. |Arthur Perret et OpenAI, 2026: GPT-5.5|
| [visualiser_motifs_classes.py](../../scripts/visualisation/visualiser_motifs_classes.py) | Visualise les motifs de classe appris par les modèles à partir des gradients sur des images MNIST et génère des planches comparatives. | OpenAI, 2026: Astra-6 |
| [comparer_test_terrain_tm_v3.py](../../scripts/evaluation/comparer_test_terrain_tm_v3.py) | Compare les résultats sauvegardés de la campagne terrain TM V3 pour les huit modèles et produit les annotations de référence, les CSV et le rapport comparatif. | — |
| [pretraitement.py](../../src/reconnaissance_chiffres/pretraitement.py) | Prépare les zones détectées en 28 × 28 pixels et conserve les images intermédiaires pour le contrôle qualité. | — |
| [datasets.py](../../src/reconnaissance_chiffres/datasets.py) | Centralise l’accès aux images et étiquettes du dataset Cascade préparé. | — |
| [modeles.py](../../src/reconnaissance_chiffres/modeles.py) | Charge les modèles Keras, avec l’adaptation des paramètres incompatibles présents dans certaines anciennes archives. | — |
