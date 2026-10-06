# Archives du projet

Ce dossier conserve les éléments historiques du travail de maturité.
Les huit modèles utilisés actuellement se trouvent dans `../modeles/actifs/`.

| Dossier | Contenu |
|---|---|
| `modeles/modeles_historiques/` | Anciennes architectures et leurs fichiers Keras |
| `resultats/graphiques/` | Anciennes courbes d'entraînement |
| `resultats/matrices_confusion/` | Anciennes évaluations |
| `resultats/tests_condition_reelle/` | Anciens essais sur photographies |
| `pdf/` | Deux rapports PDF historiques, leurs sources et les rendus de pages conservés |
| `structure_historique/` | Anciens dossiers vides conservés après la réorganisation |

Les scripts actuels n'utilisent pas automatiquement ces modèles et n'écrivent
pas leurs nouvelles sorties ici. Un ancien modèle peut être sélectionné
explicitement avec `--modele`, s'il possède les dix sorties attendues.
`Best_TEST` possède vingt sorties et ne convient pas à l'évaluation à dix classes.

Les modèles Keras et résultats historiques de ce dossier sont autorisés par
`.gitignore` pour publication. Ils ne sont disponibles après un clone que s'ils
figurent dans le commit publié.

## Documents conservés depuis l’historique Git

Les documents suivants étaient absents du workspace avant la publication de la nouvelle structure. Ils sont conservés ici avec leur contenu original :

- `pdf/` : deux bilans Cascade Top-N et leur fichier de sources ;
- `rapports_cascade_historiques/` : ancien rapport HTML et ses fichiers associés ;
- `documentation_synthetique/` : ancien guide et prompt d’import des données synthétiques.

Les rendus de pages PNG, planches de contact et extractions texte auparavant rangés dans `tmp/pdfs/` sont conservés dans `pdf/rendus/`, à côté des PDF originaux.
