# Rapports globaux Cascade Top N

Les bilans actualisés sont disponibles en PDF, avec graphiques intégrés :

- [Campagne des 24–25 juillet 2026](../../../../output/pdf/bilan_cascade_top_n_2026-07-24_25.pdf) — Best_COLOR_MAP, 13 exécutions et 12 feuilles distinctes.
- [Campagne du 5 septembre 2026](../../../../output/pdf/bilan_cascade_top_n_2026-09-05.pdf) — best_relu_MNIST_cascade_V2, 12 exécutions.

Chaque rapport contient la synthèse, la couverture Top-N, les résultats par feuille, la matrice de confusion, les écarts de détection et les limites d'interprétation. Les deux exécutions de CTN_3 en juillet restent incluses ; le score avec une seule exécution par feuille est également indiqué.

Les scores sont recalculés depuis les fichiers `predictions.csv` et les effectifs attendus sont lus dans les fichiers `resume.txt`. Le fichier `output/pdf/sources_cascade_top_n.json` consigne les sources et les agrégats. L'ancien HTML est une archive de juillet uniquement.

Régénération depuis la racine du projet :

```sh
.venv/bin/python scripts/generer_rapports_cascade_pdf.py
```

Les graphiques utilisent Matplotlib et sont incorporés en vectoriel dans les PDF. Aucun export HTML intermédiaire n'est utilisé.

Choix des graphiques : barres de couverture pour comparer quatre seuils (Top-1, 2, 3, 5), barres empilées pour décomposer les rangs par feuille, matrice d'effectifs pour les confusions et barres divergentes pour les écarts de comptage. Les pourcentages partagent une échelle de 0 à 100 ; les hachures distinguent les rangs en niveaux de gris. Les tableaux servent à consulter les valeurs exactes. Les deux rapports suivent la même structure pour faciliter leur lecture, sans présenter les campagnes comme des essais indépendants et contrôlés.
