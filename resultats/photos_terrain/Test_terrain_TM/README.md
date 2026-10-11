# Test terrain — Test_terrain_TM

Comparatif actualisé le 7 octobre 2026 après remplacement du modèle ×10. Les sept autres résultats et les annotations de référence sont conservés.

La photo contient 27 chiffres ; 27 sont détectés. Les rectangles et les entrées 28×28 ont été vérifiés identiques à ceux du test historique. Chaque chiffre non détecté est compté comme un échec sur la photo.

| Modèle | Corrects / photo | Réussite photo | Réussite détectés | Erreurs de reconnaissance | Image annotée |
|---|---:|---:|---:|---|---|
| Best_COLOR_MAP_cascade | 27/27 | 100.00 % | 100.00 % | Aucune | [Voir](2026-10-06_08-49-50/03_resultats/resultat_annote.jpg) |
| best_relu_10xcascade | 27/27 | 100.00 % | 100.00 % | Aucune | [Voir](2026-10-07_13-24-30/03_resultats/resultat_annote.jpg) |
| best_relu_2xcascade | 27/27 | 100.00 % | 100.00 % | Aucune | [Voir](2026-10-06_08-49-34/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade | 27/27 | 100.00 % | 100.00 % | Aucune | [Voir](2026-10-06_08-49-12/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade_V2 | 27/27 | 100.00 % | 100.00 % | Aucune | [Voir](2026-10-06_08-49-18/03_resultats/resultat_annote.jpg) |
| best_relu | 24/27 | 88.89 % | 88.89 % | L1 C6: 6→5 | L3 C4: 4→8 | L3 C9: 9→5 | [Voir](2026-10-06_08-49-24/03_resultats/resultat_annote.jpg) |
| best_softmax | 21/27 | 77.78 % | 77.78 % | L1 C4: 4→8 | L2 C9: 9→3 | L2 C1: 1→7 | L3 C4: 4→8 | L3 C9: 9→3 | L3 C7: 7→3 | [Voir](2026-10-06_08-49-39/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP | 19/27 | 70.37 % | 70.37 % | L1 C7: 7→2 | L1 C4: 4→8 | L2 C9: 9→3 | L2 C1: 1→7 | L3 C4: 4→8 | L3 C9: 9→3 | L3 C7: 7→2 | L3 C1: 1→4 | [Voir](2026-10-06_08-49-44/03_resultats/resultat_annote.jpg) |

L = ligne, C = colonne. Les taux portent uniquement sur cette photo.
Aucun chiffre non détecté.

Ces photos ne figurent pas dans les sources Cascade du nouveau modèle. Ce sont des contrôles terrain sur des photos distinctes ; un recouvrement d’auteur des écritures ne peut pas être établi avec les métadonnées disponibles.

Le ×2 enregistré conserve son protocole historique ; la comparaison ×2/×10 ne permet pas d’isoler causalement le facteur de répétition.

Les CSV bruts utilisent `--chiffre-reel inconnu` : les annotations par chiffre et les scores sont dans `annotations_reference.csv`, `predictions_8_modeles_evaluees.csv` et `comparaison_8_modeles.csv`. Le nouveau résumé concerne le nouveau ×10 ; les autres exécutions gardent leur date historique.
