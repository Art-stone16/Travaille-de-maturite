# Tests terrain

Toutes les exécutions sont classées directement par photographie :

```text
01_TESTS_TERRAIN/
├── journal_global_tests_terrain.csv
├── image_reelle_9/
│   └── AAAA-MM-JJ_HH-MM-SS/
└── test_terrain/
    └── AAAA-MM-JJ_HH-MM-SS/
```

Chaque dossier horodaté contient la source et les paramètres, les étapes de
détection, les contrôles du prétraitement et les résultats. Le nom de
l'expérience et celui du modèle figurent dans `00_source/parametres.json`,
`03_resultats/resultats_terrain.csv` et le journal global ; ils ne créent donc
pas de dossiers intermédiaires supplémentaires.
