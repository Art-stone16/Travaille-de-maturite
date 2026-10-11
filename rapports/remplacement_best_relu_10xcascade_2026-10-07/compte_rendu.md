# Remplacement de best_relu_10xcascade — 7 octobre 2026

## Modèle ×10 remplacé et évaluations régénérées

Le modèle actif utilise dix exemplaires exacts par chiffre Cascade : un original et neuf copies identiques. Le meilleur checkpoint atteint 99,10 % sur les 10 000 images MNIST test, soit 9 910 prédictions correctes et 90 erreurs. Les anciens poids, scripts et résultats sont archivés. Les autres modèles actifs sont conservés. Les contrôles CTN utilisent les sources d’entraînement et doivent être lus séparément des évaluations indépendantes.

## Périmètre et dénominateurs

MNIST test est le jeu officiel de 10 000 images, sans Cascade ; accuracy = nombre de prédictions exactes / 10 000. Pour chaque chiffre, la sensibilité mesure la proportion de vrais chiffres correctement reconnus ; la spécificité mesure la proportion des autres chiffres non confondus avec lui. Les intervalles par chiffre sont les intervalles de Wilson à 95 % du script existant.

CTN Top-N = nombre de zones où le chiffre de la feuille se trouve parmi les N premiers choix / 1 227 zones détectées. Les 1 217 zones validées ont servi à l’entraînement ; dix zones ont été exclues. Les douze feuilles totalisent 1 201 chiffres déclarés attendus : cette différence de comptage ne donne pas un nombre exact de faux positifs. Pour les photos annotées, le taux sur la photo utilise tous les chiffres annotés ; un chiffre non détecté est un échec.

## Validation séparée et copies exactes

La version ×2 historique mélangeait les copies avant la validation : 307 originaux étaient communs aux deux partitions, et 357 copies étaient réservées à la validation. Cette fuite n’a pas été reproduite. Les deux scripts appellent désormais la même routine, avec des facteurs respectifs de 2 et de 10. Une permutation de graine 42 réserve d’abord 9 000 images MNIST à la validation ; toutes les copies Cascade sont ensuite ajoutées au train. Les 1 217 matrices ont été comparées aux lignes du manifeste et aux étiquettes. Les empreintes des images Cascade n’ont aucun recouvrement avec MNIST validation ou test.

Architecture : Conv2D 64 filtres 5×5 ReLU, BatchNormalization, MaxPooling 2×2, Conv2D 64 filtres 3×3 ReLU, BatchNormalization, MaxPooling 2×2, Flatten, Dropout 0,6, Dense 10 softmax. Total : 55 114 paramètres. Adam à 0,001, SparseCategoricalCrossentropy, batch 128, 30 époques maximum. Checkpoint sur val_loss ; EarlyStopping sur val_loss, patience 2, sans restauration automatique des meilleurs poids. La graine globale 42 fixe Python, NumPy et Keras/TensorFlow ; les opérations déterministes sont demandées. Le test utilise explicitement best_model.keras.

| Partition | Nouveau ×10 |
| --- | --- |
| MNIST entraînement | 51000 |
| Cascade originaux | 1217 |
| Copies supplémentaires | 10953 |
| Cascade entraînement total | 12170 |
| Entraînement total | 63170 |
| MNIST validation | 9000 |
| Cascade validation / test | 0 / 0 |
| MNIST test | 10000 |
| Époques réalisées / retenue | 11 / 9 |

Répartition des originaux par chiffre 0 à 9 : 98, 100, 228, 102, 99, 103, 187, 100, 100, 100. Chaque effectif est multiplié par dix dans le train ×10.

## MNIST : meilleur checkpoint et état final distingués

| État | Corrects | Erreurs | Accuracy |
| --- | --- | --- | --- |
| Ancien ×10 augmenté, meilleur checkpoint | 9910 | 90 | 99,10 % |
| Nouveau ×10 exact, meilleur checkpoint | 9910 | 90 | 99,10 % |

L’état final du nouveau ×10 atteint 99,23 %, avec une loss 0.032277. Le meilleur checkpoint a une loss 0.028750. L’ancien score publié de 99,11 % et sa loss 0,030425 concernaient l’état final. L’ancien meilleur checkpoint a été réévalué ici sur les mêmes 10 000 entrées : son accuracy agrégée est aussi de 99,10 %, mais les erreurs et métriques par chiffre diffèrent. 

## Métriques MNIST du nouveau modèle par chiffre

| Chiffre | Effectif | Corrects | FN | Sensibilité | Spécificité | Précision | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 980 | 978 | 2 | 99,80 % | 99,92 % | 99,29 % | 99,54 % |
| 1 | 1135 | 1134 | 1 | 99,91 % | 99,86 % | 98,95 % | 99,43 % |
| 2 | 1032 | 1015 | 17 | 98,35 % | 99,96 % | 99,61 % | 98,98 % |
| 3 | 1010 | 1005 | 5 | 99,50 % | 99,81 % | 98,34 % | 98,92 % |
| 4 | 982 | 974 | 8 | 99,19 % | 99,91 % | 99,19 % | 99,19 % |
| 5 | 892 | 879 | 13 | 98,54 % | 99,92 % | 99,21 % | 98,88 % |
| 6 | 958 | 944 | 14 | 98,54 % | 99,94 % | 99,47 % | 99,00 % |
| 7 | 1028 | 1018 | 10 | 99,03 % | 99,88 % | 98,93 % | 98,98 % |
| 8 | 974 | 963 | 11 | 98,87 % | 99,94 % | 99,48 % | 99,18 % |
| 9 | 1009 | 1000 | 9 | 99,11 % | 99,84 % | 98,62 % | 98,86 % |

Le chiffre 2 a ici la sensibilité la plus faible (1 015/1 032). Les matrices en effectifs et normalisées, les dix scores de chaque image et les intervalles par chiffre sont exportés dans le dossier MNIST.

## CTN : contrôles sur les sources d’entraînement

| Expérience | Top-1 | Top-2 | Top-3 | Top-5 | Top-1 sur validés |
| --- | --- | --- | --- | --- | --- |
| Ancien ×10 augmenté | 1209/1227 (98,53 %) | 1219/1227 (99,35 %) | 1221/1227 (99,51 %) | 1222/1227 (99,59 %) | 1207/1217 |
| Nouveau ×10 exact | 1215/1227 (99,02 %) | 1219/1227 (99,35 %) | 1219/1227 (99,35 %) | 1221/1227 (99,51 %) | 1214/1217 |

Ces résultats mesurent la reconnaissance de données issues des mêmes feuilles que l’entraînement. Les dix zones exclues sont détaillées séparément dans les données du bilan. Les taux complets restent ceux du protocole historique pour permettre la comparaison ; ils ne deviennent pas des scores indépendants.

| Chiffre | Zones | Top-1 | Top-3 | Top-5 |
| --- | --- | --- | --- | --- |
| 0 | 98 | 100,00 % | 100,00 % | 100,00 % |
| 1 | 100 | 100,00 % | 100,00 % | 100,00 % |
| 2 | 232 | 98,28 % | 98,28 % | 98,28 % |
| 3 | 102 | 100,00 % | 100,00 % | 100,00 % |
| 4 | 99 | 98,99 % | 100,00 % | 100,00 % |
| 5 | 105 | 97,14 % | 98,10 % | 100,00 % |
| 6 | 189 | 99,47 % | 99,47 % | 99,47 % |
| 7 | 100 | 100,00 % | 100,00 % | 100,00 % |
| 8 | 101 | 98,02 % | 100,00 % | 100,00 % |
| 9 | 101 | 99,01 % | 99,01 % | 99,01 % |

## Les cinq protocoles terrain sont rejoués

| Photo | Modèle | Détectés | Corrects / photo | Réussite photo |
| --- | --- | --- | --- | --- |
| Test_terrain_TM | Nouveau ×10 exact | 27 | 27/27 | 100,00 % |
| Test_terrain_TM_V2 | Nouveau ×10 exact | 29 | 29/30 | 96,67 % |
| Test_terrain_TM_V3 | Nouveau ×10 exact | 30 | 30/30 | 100,00 % |
| image_reelle_Grand | Nouveau ×10 exact | 22 | Non calculable | Annotations absentes |
| test_terrain | Nouveau ×10 exact | 31 | Non calculable | Annotations absentes |

Les 139 zones des cinq tests ×10 ont les mêmes rectangles et les mêmes entrées 28×28 que leurs exécutions historiques. Les images annotées, contrôles en niveaux de gris, matrices binaires de contrôle, CSV et résumés ont été régénérés. Les trois annotations du TM sont conservées à l’identique ; TM_V2 garde le 4 non détecté de la dernière ligne. Les photos terrain sont distinctes des douze sources Cascade ; l’identité des auteurs des écritures ne peut pas être vérifiée. Les deux photos non annotées permettent de comparer les prédictions et le prétraitement, mais ne permettent pas de calculer un taux de réussite fiable.

## Archive, chemins actifs et résultats conservés

Archive : `archives/remplacement_best_relu_10xcascade_2026-10-07/`. Elle contient le script original, les poids best/final, les métadonnées, les cinq tests terrain, les douze tests CTN, les courbes, motifs et anciens comparatifs. Les anciens dossiers d’exécution ×10 ont été retirés des résultats actifs après vérification des copies archivées. Les anciennes lignes ×10 du journal terrain y sont aussi archivées.

Le nouveau modèle a été enregistré provisoirement, rechargé, puis installé sous `modeles/actifs/best_relu_10xcascade/best_model.keras` et `final_model.keras`. Les empreintes enregistrées dans les tests MNIST, CTN, terrain et motifs correspondent au fichier actif. Les 18 fichiers des autres modèles et 5 693 sorties individuelles sont identiques ; leurs lignes dans les trois comparatifs terrain sont également identiques. Les douze montages CTN à huit modèles, les rapports CTN et terrain et la comparaison de motifs ont été actualisés. Le ×2 actif conserve ses poids historiques.

## Fichiers modifiés et résultats régénérés

- `scripts/entrainement/train_best_relu_10xcascade.py` et `train_best_relu_2xcascade.py` : appels à la routine commune ; seul le facteur varie.
- `src/reconnaissance_chiffres/entrainement_cascade.py` : copies exactes, séparation des partitions, contrôles, entraînement et sauvegardes.
- `tests/test_injection_cascade.py` : dix occurrences par original, labels, validation identique et exclusion de Cascade. Les 43 tests du dépôt passent.
- `scripts/evaluation/test_cascade_top_n.py`, `test_condition_reelle.py` et `src/reconnaissance_chiffres/rapports.py` : empreintes, statut des contrôles et rapports.
- `scripts/evaluation/comparer_test_terrain_tm_v3.py` : accepte les trois schémas terrain, conserve les annotations et les lignes des sept autres modèles.
- `scripts/evaluation/reevaluer_best_relu_10xcascade.py` et `bilan_remplacement_relu_10x.py` : installation, relecture des protocoles, vérifications et bilan reproductible.
- `scripts/visualisation/visualiser_motifs_classes.py` : recalcul sélectif du ×10, réutilisation des autres cartes et provenance.
- `modeles/actifs/best_relu_10xcascade/` : nouveaux best/final, métadonnées, indices de partition et historique CSV.
- `README.md`, `.gitignore`, `modeles/catalogue.md`, comparatifs des scripts et des modèles : documentation du remplacement et conservation des poids archivés.
- Sorties : deux évaluations MNIST complètes (ancien checkpoint et nouveau), douze CTN ×10, cinq terrains ×10, leurs comparatifs, douze montages CTN et deux illustrations de motifs.
- L’inventaire exhaustif des fichiers créés, modifiés ou déplacés est enregistré dans `fichiers_modifies.json`. Les chemins et correspondances avant/après des tests sont dans l’archive.

## Limites de reproduction et de comparaison

- Aucune étude de stabilité enregistrée ne cible le ×10. L’étude présente utilise 16/32 filtres, noyaux 5×5/5×5, dropout 0,4 et les graines 42 à 46 : elle est conservée. Aucun résultat de stabilité ×10 n’est inventé.
- L’initialisation aléatoire des anciennes expériences ×2/×10 n’était pas enregistrée ; leur entraînement exact n’est pas reproductible à partir du dépôt. Les poids archivés permettent de rejouer leurs prédictions.
- Le nouveau ×10 a été entraîné avec une seule graine ; aucune étude de stabilité supplémentaire n’a été réalisée.
- Il n’existe pas de jeu Cascade indépendant dans le dataset préparé : tous les 1 217 exemples inclus sont marqués train. Aucune évaluation finale Cascade indépendante ne peut donc être produite avec ces fichiers.
- Les fichiers de matrices MNIST de l’ancien ×10 n’étaient pas présents. Une évaluation complète de son checkpoint archivé a été effectuée pour rendre la comparaison actuelle explicite.
- `test_terrain.jpg` et `image_reelle_Grand.jpg` n’ont pas d’annotations par chiffre : leurs métriques d’exactitude sont indisponibles.
- Le manuscrit du TM n’est pas présent dans le dépôt. Les numéros de figures, tableaux et pages ne peuvent pas être établis.
- Le rapport HTML est validé structurellement et contient une représentation lisible sans JavaScript. La vérification visuelle automatisée du lecteur HTML n’a pas pu être exécutée : le Chromium de vérification est absent et le Chrome installé n’a pas démarré dans ce runtime. Les figures PNG d’entraînement et MNIST ont été inspectées séparément.

## Éléments du TM à actualiser si tu les as repris

1. La méthode ×10 avec rotations ±8° et translations ±1,5 pixel, ainsi que toute légende parlant de neuf variantes augmentées. La nouvelle méthode est un original et neuf copies identiques.
2. L’ancienne description de la validation : mélange MNIST/Cascade avant réservation des 15 %. Les nouveaux effectifs sont 63 170 en train et 9 000 en validation MNIST ; les 12 170 entrées Cascade sont toutes en train.
3. L’ancien score MNIST final ×10 : 99,11 %, loss 0,030425308 ; et toute confusion entre ce score et best_model.keras. Les nouvelles valeurs sont 99,10 % pour best et 99,23 % pour final.
4. Les courbes ×10 de 14 époques : elles sont remplacées par 11 époques, checkpoint retenu à l’époque 9. La courbe ×2 active reste historique.
5. La ligne ×10 des tableaux CTN : Top-1 98,53 %, Top-2 99,35 %, Top-3 99,51 %, Top-5 99,59 %. Les tableaux par chiffre et de confusions, ainsi que le classement des modèles, ont été régénérés. Les anciens et nouveaux CTN doivent être identifiés comme contrôles sur les sources d’entraînement.
6. Toutes les images annotées et confidences ×10 des douze feuilles CTN, les douze montages à huit modèles et les cinq photos terrain. Les trois scores TM ×10 restent 27/27, 29/30 et 30/30, mais leurs confidences et fichiers d’exécution ont changé.
7. La planche `planche_06_best_relu_10xcascade.png`, ses données de gradients et `comparaison_7_modeles.png`. Les cartes des six autres modèles sont conservées.
8. Toute conclusion attribuant au seul facteur ×2/×10 un écart entre les modèles actifs historiques. Le ×2 actif conserve son protocole historique.

Les nombres de paramètres et l’architecture du ×10 restent valables. Les PDF CTN archivés de juillet et du 5 septembre 2026 précèdent l’ancien ×10 du 25 septembre et ne sont pas concernés par son remplacement.

## Résultats disponibles pour mettre à jour le TM

Utiliser les nouveaux CSV, matrices et illustrations du ×10, en gardant la séparation entre MNIST test, contrôles CTN et terrain annoté. Le bilan et les scripts fournissent la provenance des chiffres sans modifier le manuscrit. Pour répondre à une question de stabilité, il faudrait une nouvelle expérience sur plusieurs graines ; pour mesurer la généralisation Cascade, il faudrait de nouvelles feuilles indépendantes avec des annotations.
