# Test terrain — Test_terrain_TM_V2

Comparatif actualisé le 7 octobre 2026 après remplacement du modèle ×10. Les sept autres résultats et les annotations de référence sont conservés.

La photo contient 30 chiffres ; 29 sont détectés. Les rectangles et les entrées 28×28 ont été vérifiés identiques à ceux du test historique. Chaque chiffre non détecté est compté comme un échec sur la photo.

| Modèle | Corrects / photo | Réussite photo | Réussite détectés | Erreurs de reconnaissance | Image annotée |
|---|---:|---:|---:|---|---|
| best_relu_10xcascade | 29/30 | 96.67 % | 100.00 % | Aucune | [Voir](2026-10-07_13-24-37/03_resultats/resultat_annote.jpg) |
| best_relu_2xcascade | 29/30 | 96.67 % | 100.00 % | Aucune | [Voir](2026-10-06_08-57-16/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade_V2 | 29/30 | 96.67 % | 100.00 % | Aucune | [Voir](2026-10-06_08-57-02/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade | 28/30 | 93.33 % | 96.55 % | L3 C10: 9→0 | [Voir](2026-10-06_08-56-57/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP_cascade | 26/30 | 86.67 % | 89.66 % | L1 C5: 4→9 ; L2 C6: 5→3 ; L2 C8: 7→1 | [Voir](2026-10-06_08-57-31/03_resultats/resultat_annote.jpg) |
| best_softmax | 26/30 | 86.67 % | 89.66 % | L1 C5: 4→8 ; L3 C8: 7→8 ; L3 C10: 9→5 | [Voir](2026-10-06_08-57-22/03_resultats/resultat_annote.jpg) |
| best_relu | 23/30 | 76.67 % | 79.31 % | L1 C2: 1→7 ; L1 C5: 4→8 ; L2 C2: 1→7 ; L2 C8: 7→1 ; L3 C2: 1→7 ; L3 C10: 9→3 | [Voir](2026-10-06_08-57-07/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP | 20/30 | 66.67 % | 68.97 % | L1 C5: 4→8 ; L1 C7: 6→5 ; L1 C8: 7→2 ; L2 C2: 1→7 ; L2 C10: 9→3 ; L3 C2: 1→7 ; L3 C7: 6→8 ; L3 C8: 7→2 ; L3 C10: 9→3 | [Voir](2026-10-06_08-57-26/03_resultats/resultat_annote.jpg) |

L = ligne, C = colonne. Les taux portent uniquement sur cette photo.
Chiffres non détectés : L3 C5: 4 (rejet hauteur)

Ces photos ne figurent pas dans les sources Cascade du nouveau modèle. Ce sont des contrôles terrain sur des photos distinctes ; un recouvrement d’auteur des écritures ne peut pas être établi avec les métadonnées disponibles.

Le ×2 enregistré conserve son protocole historique ; la comparaison ×2/×10 ne permet pas d’isoler causalement le facteur de répétition.

Les CSV bruts utilisent `--chiffre-reel inconnu` : les annotations par chiffre et les scores sont dans `annotations_reference.csv`, `predictions_8_modeles_evaluees.csv` et `comparaison_8_modeles.csv`. Le nouveau résumé concerne le nouveau ×10 ; les autres exécutions gardent leur date historique.
