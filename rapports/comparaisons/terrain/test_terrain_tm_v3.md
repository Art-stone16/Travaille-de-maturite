# Test terrain — Test_terrain_TM_V3

Comparatif actualisé le 7 octobre 2026 après remplacement du modèle ×10. Les sept autres résultats et les annotations de référence sont conservés.

La photo contient 30 chiffres ; 30 sont détectés. Les rectangles et les entrées 28×28 ont été vérifiés identiques à ceux du test historique. Chaque chiffre non détecté est compté comme un échec sur la photo.

| Modèle | Corrects / photo | Réussite photo | Réussite détectés | Erreurs de reconnaissance | Image annotée |
|---|---:|---:|---:|---|---|
| Best_COLOR_MAP_cascade | 30/30 | 100.00 % | 100.00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-56/03_resultats/resultat_annote.jpg) |
| best_relu_10xcascade | 30/30 | 100.00 % | 100.00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-07_13-24-42/03_resultats/resultat_annote.jpg) |
| best_relu_2xcascade | 30/30 | 100.00 % | 100.00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-43/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade_V2 | 30/30 | 100.00 % | 100.00 % | Aucune | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-31/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade | 28/30 | 93.33 % | 93.33 % | L2 C5: 4→8 ; L3 C5: 4→7 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-27/03_resultats/resultat_annote.jpg) |
| best_softmax | 27/30 | 90.00 % | 90.00 % | L1 C5: 4→8 ; L2 C2: 1→4 ; L2 C5: 4→8 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-48/03_resultats/resultat_annote.jpg) |
| best_relu | 26/30 | 86.67 % | 86.67 % | L1 C2: 1→7 ; L1 C5: 4→8 ; L2 C5: 4→8 ; L3 C5: 4→8 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-35/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP | 23/30 | 76.67 % | 76.67 % | L1 C2: 1→7 ; L1 C5: 4→8 ; L1 C8: 7→3 ; L2 C2: 1→7 ; L2 C5: 4→8 ; L3 C2: 1→4 ; L3 C8: 7→8 | [Voir](../../../resultats/photos_terrain/Test_terrain_TM_V3/2026-10-06_09-23-52/03_resultats/resultat_annote.jpg) |

L = ligne, C = colonne. Les taux portent uniquement sur cette photo.
Aucun chiffre non détecté.

Ces photos ne figurent pas dans les sources Cascade du nouveau modèle. Ce sont des contrôles terrain sur des photos distinctes ; un recouvrement d’auteur des écritures ne peut pas être établi avec les métadonnées disponibles.

Le ×2 enregistré conserve son protocole historique ; la comparaison ×2/×10 ne permet pas d’isoler causalement le facteur de répétition.

Les CSV bruts utilisent `--chiffre-reel inconnu` : les annotations par chiffre et les scores sont dans `annotations_reference.csv`, `predictions_8_modeles_evaluees.csv` et `comparaison_8_modeles.csv`. Le nouveau résumé concerne le nouveau ×10 ; les autres exécutions gardent leur date historique.
