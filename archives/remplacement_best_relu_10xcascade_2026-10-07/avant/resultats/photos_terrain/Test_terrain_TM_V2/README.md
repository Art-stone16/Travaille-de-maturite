# Test terrain — Test_terrain_TM_V2

Test du 6 octobre 2026 sur les huit modèles actifs (`best_model.keras`), avec les mêmes réglages que le test précédent.

La photo contient 30 chiffres : trois lignes de 0 à 9. Le détecteur accepte 29 chiffres et rejette le 4 de la dernière ligne à cause de sa hauteur (318 pixels, contre un maximum de 260,4 pixels, soit 20 % de la hauteur de la photo). Ce chiffre ne reçoit donc aucune prédiction. Aucune composante supplémentaire n’est détectée.

Les huit modèles ont reçu les mêmes 29 rectangles et les mêmes entrées 28×28. Les annotations ont été vérifiées visuellement sur la photo et le diagnostic de détection.

| Modèle | Corrects / photo | Réussite sur la photo | Réussite sur les 29 détectés | Erreurs de reconnaissance | Image annotée |
|---|---:|---:|---:|---|---|
| best_relu_10xcascade | 29/30 | 96,67 % | 100,00 % | Aucune | [Voir](2026-10-06_08-57-12/03_resultats/resultat_annote.jpg) |
| best_relu_2xcascade | 29/30 | 96,67 % | 100,00 % | Aucune | [Voir](2026-10-06_08-57-16/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade_V2 | 29/30 | 96,67 % | 100,00 % | Aucune | [Voir](2026-10-06_08-57-02/03_resultats/resultat_annote.jpg) |
| Best_relu_cascade | 28/30 | 93,33 % | 96,55 % | L3 C10: 9→0 | [Voir](2026-10-06_08-56-57/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP_cascade | 26/30 | 86,67 % | 89,66 % | L1 C5: 4→9 ; L2 C6: 5→3 ; L2 C8: 7→1 | [Voir](2026-10-06_08-57-31/03_resultats/resultat_annote.jpg) |
| best_softmax | 26/30 | 86,67 % | 89,66 % | L1 C5: 4→8 ; L3 C8: 7→8 ; L3 C10: 9→5 | [Voir](2026-10-06_08-57-22/03_resultats/resultat_annote.jpg) |
| best_relu | 23/30 | 76,67 % | 79,31 % | L1 C2: 1→7 ; L1 C5: 4→8 ; L2 C2: 1→7 ; L2 C8: 7→1 ; L3 C2: 1→7 ; L3 C10: 9→3 | [Voir](2026-10-06_08-57-07/03_resultats/resultat_annote.jpg) |
| Best_COLOR_MAP | 20/30 | 66,67 % | 68,97 % | L1 C5: 4→8 ; L1 C7: 6→5 ; L1 C8: 7→2 ; L2 C2: 1→7 ; L2 C10: 9→3 ; L3 C2: 1→7 ; L3 C7: 6→8 ; L3 C8: 7→2 ; L3 C10: 9→3 | [Voir](2026-10-06_08-57-26/03_resultats/resultat_annote.jpg) |

L = ligne, C = colonne (de gauche à droite). La colonne 1 correspond au chiffre 0, la colonne 5 au chiffre 4.

Le taux sur la photo utilise les 30 chiffres comme dénominateur et compte le chiffre non détecté comme un échec du système complet. Le taux sur les détectés utilise seulement les 29 chiffres effectivement soumis aux modèles. Ces scores concernent uniquement cette photo.

## Fichiers

- `comparaison_8_modeles.csv` : scores, erreurs de reconnaissance et chiffre non détecté.
- `predictions_8_modeles_evaluees.csv` : 240 lignes, soit les 30 chiffres pour chacun des 8 modèles ; la prédiction du chiffre non détecté est vide.
- `annotations_reference.csv` : chiffres réels, rectangles et motif du rejet.
- `comparaison_provenance.json` : provenance, empreintes et limite de hauteur.
- Chaque dossier horodaté contient les diagnostics, les contrôles du prétraitement, une photo annotée et les résultats originaux.

Les exécutions originales utilisent `--chiffre-reel inconnu`, car cette option accepte une seule valeur pour toute la photo. Leur CSV, leur résumé et le journal global conservent donc la valeur réelle vide. Les annotations par chiffre et les scores sont enregistrés séparément dans les fichiers comparatifs.
