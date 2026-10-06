# Test terrain — Test_terrain_TM

Test du 6 octobre 2026 sur les huit modèles actifs (`best_model.keras`).

La photo contient 27 chiffres : trois lignes de 1 à 9. Les 27 chiffres ont été détectés, sans composante supplémentaire, avec les mêmes rectangles et les mêmes entrées 28×28 pour les huit modèles. La correspondance des rectangles avec les chiffres a été vérifiée sur la photo et le diagnostic de détection.

| Modèle | Corrects | Réussite | Erreurs | Image annotée |
|---|---:|---:|---|---|
| Best_COLOR_MAP_cascade | 27/27 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-50/03_resultats/resultat_annote.jpg) |
| best_relu_10xcascade | 27/27 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-29/03_resultats/resultat_annote.jpg) |
| best_relu_2xcascade | 27/27 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-34/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade | 27/27 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-12/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade_V2 | 27/27 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-18/03_resultats/resultat_annote.jpg) |
| best_relu | 24/27 | 88,89 % | L1 C6: 6→5 ; L3 C4: 4→8 ; L3 C9: 9→5 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-24/03_resultats/resultat_annote.jpg) |
| best_softmax | 21/27 | 77,78 % | L1 C4: 4→8 ; L2 C9: 9→3 ; L2 C1: 1→7 ; L3 C4: 4→8 ; L3 C9: 9→3 ; L3 C7: 7→3 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-39/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP | 19/27 | 70,37 % | L1 C7: 7→2 ; L1 C4: 4→8 ; L2 C9: 9→3 ; L2 C1: 1→7 ; L3 C4: 4→8 ; L3 C9: 9→3 ; L3 C7: 7→2 ; L3 C1: 1→4 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM/2026-10-06_08-49-44/03_resultats/resultat_annote.jpg) |

L = ligne, C = colonne (de gauche à droite).

Le taux de réussite porte uniquement sur cette photo et ne mesure pas les performances générales des modèles. Aucun 0 ne figure sur la photo.

## Fichiers

- `comparaison_8_modeles.csv` : scores et erreurs par modèle.
- `predictions_8_modeles_evaluees.csv` : les 216 prédictions avec valeur réelle et résultat de la comparaison.
- `annotations_reference.csv` : chiffres réels et correspondance avec les rectangles détectés.
- `comparaison_provenance.json` : provenance et empreintes des fichiers utilisés.
- Chaque dossier horodaté contient les diagnostics de détection, les contrôles du prétraitement, une photo annotée et les résultats du script.

Les exécutions originales utilisent `--chiffre-reel inconnu`, car cette option accepte une seule valeur pour toute la photo. Leur CSV, leur résumé et le journal global conservent donc la valeur réelle vide. Les annotations par chiffre et les scores calculés à partir de celles-ci sont enregistrés séparément dans les fichiers comparatifs ci-dessus.

Données et provenance : [dossier de cette photo](../../../resultats/photos_terrain/Test_terrain_TM).
