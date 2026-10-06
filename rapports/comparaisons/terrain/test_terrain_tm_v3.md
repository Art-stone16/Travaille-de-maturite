# Test terrain — Test_terrain_TM_V3

Test du 6 octobre 2026 sur les huit modèles actifs (`best_model.keras`), avec les mêmes réglages que les tests précédents.

La photo contient 30 chiffres : trois lignes de 0 à 9. Les 30 chiffres sont détectés sans composante supplémentaire. Les annotations ont été vérifiées visuellement sur la photo et le diagnostic de détection. Les huit modèles reçoivent les mêmes rectangles et les mêmes entrées 28×28.

| Modèle | Corrects | Réussite | Erreurs de reconnaissance | Image annotée |
|---|---:|---:|---|---|
| Best_COLOR_MAP_cascade | 30/30 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-56/03_resultats/resultat_annote.jpg) |
| best_relu_10xcascade | 30/30 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-39/03_resultats/resultat_annote.jpg) |
| best_relu_2xcascade | 30/30 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-43/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade_V2 | 30/30 | 100,00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-31/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade | 28/30 | 93,33 % | L2 C5: 4→8 ; L3 C5: 4→7 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-27/03_resultats/resultat_annote.jpg) |
| best_softmax | 27/30 | 90,00 % | L1 C5: 4→8 ; L2 C2: 1→4 ; L2 C5: 4→8 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-48/03_resultats/resultat_annote.jpg) |
| best_relu | 26/30 | 86,67 % | L1 C2: 1→7 ; L1 C5: 4→8 ; L2 C5: 4→8 ; L3 C5: 4→8 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-35/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP | 23/30 | 76,67 % | L1 C2: 1→7 ; L1 C5: 4→8 ; L1 C8: 7→3 ; L2 C2: 1→7 ; L2 C5: 4→8 ; L3 C2: 1→4 ; L3 C8: 7→8 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-52/03_resultats/resultat_annote.jpg) |

L = ligne, C = colonne, de gauche à droite. La colonne 1 correspond au chiffre 0. Les scores concernent uniquement cette photo.

## Fichiers

- `comparaison_8_modeles.csv` : scores et erreurs par modèle.
- `predictions_8_modeles_evaluees.csv` : les 240 prédictions évaluées.
- `annotations_reference.csv` : chiffres réels et correspondance avec les rectangles.
- `comparaison_provenance.json` : provenance et empreintes des fichiers.
- Chaque dossier horodaté contient les diagnostics, les contrôles du prétraitement, une photo annotée et les résultats originaux.

Les exécutions originales utilisent `--chiffre-reel inconnu`, car cette option accepte une seule valeur pour toute la photo. Leur CSV, leur résumé et le journal global conservent donc la valeur réelle vide. Les annotations par chiffre et les scores sont enregistrés séparément dans les fichiers comparatifs.

Données et provenance : [dossier de cette photo](../../../resultats/photos_terrain/Test_terrain_TM_V3).
