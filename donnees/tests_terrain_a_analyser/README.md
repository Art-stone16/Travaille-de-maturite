# Photographies terrain à analyser

Range ici les nouvelles photographies, de préférence dans un sous-dossier par
protocole :

```text
tests_terrain_a_analyser/
└── papier_blanc_stylo_noir/
    ├── classe_0.jpg
    ├── classe_1.jpg
    └── classe_9.jpg
```

Une photo `classe_7.jpg` doit contenir uniquement des 7 si elle est évaluée avec
`--chiffre-reel 7`. Conserve aussi les photos ratées : elles font partie du test
des limites du système.

Le script ne modifie pas ces fichiers. Toutes les sorties vont dans
`sorties/tests_condition_reelle/`.
