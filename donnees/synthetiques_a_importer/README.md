# Entrées synthétiques à importer

Ce dossier reçoit uniquement les fichiers bruts produits à la main ou par un
outil externe. Le script ne modifie pas ces fichiers. Les datasets contrôlés
sont créés séparément dans `donnees/datasets_synthetiques/<nom_du_dataset>/`.

## Organisation conseillée

Pour des fichiers image, NumPy, CSV ou texte, créez un sous-dossier par classe :

```text
donnees/synthetiques_a_importer/
├── 0/
│   ├── style_fin_01.png
│   └── style_rond_02.npy
├── 1/
│   └── style_fin_01.csv
└── A/
    └── manuscrit_01.png
```

L'étiquette est alors déduite du premier sous-dossier (`0`, `1`, `A`, etc.).
Pour une entrée isolée, utilisez `--etiquette`. Un fichier JSON peut aussi
porter une étiquette différente pour chaque échantillon.

## Formats acceptés

- Images : `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`.
- Matrices : `.npy`, `.npz`, `.csv`, `.txt`.
- Lots structurés : `.json` avec un champ `samples`.

Une matrice doit avoir exactement 28 lignes et 28 colonnes. Les pixels peuvent
être compris entre 0 et 1 ou entre 0 et 255. Pour faciliter le contrôle, le
format recommandé est : fond `0`, trait `1`. Les photographies et autres images
sont recadrées et redimensionnées par le script.

Schéma JSON recommandé :

```json
{
  "format_version": 1,
  "samples": [
    {
      "id": "7_style_01",
      "label": "7",
      "split": "non_attribue",
      "pixels": [[0, 0, 0]]
    }
  ]
}
```

Dans le vrai fichier, `pixels` doit contenir 28 lignes de 28 valeurs. Le petit
extrait ci-dessus illustre seulement la structure.

Les champs supplémentaires, comme `style` sur un échantillon ou
`generation_notes` à la racine, sont conservés dans la provenance du dataset
préparé.

## Commandes

Depuis la racine du projet :

```bash
.venv/bin/python scripts/preparer_dataset_synthetique.py importer \
  --entree donnees/synthetiques_a_importer \
  --nom-dataset claude_lot_01 \
  --origine claude \
  --description "Premier lot, plusieurs styles manuscrits"
```

Puis contrôlez explicitement le résultat :

```bash
.venv/bin/python scripts/preparer_dataset_synthetique.py valider \
  --dataset donnees/datasets_synthetiques/claude_lot_01
```

Le dossier créé contient `manifest.csv`, `manifest.json`, les matrices, les PNG,
une grille d'aperçu et les rapports de validation ou de rejet.

Le fichier [PROMPT_CLAUDE.md](PROMPT_CLAUDE.md) contient un modèle de demande.
Le script n'appelle pas Claude automatiquement : il importe seulement les
fichiers que vous déposez ici.
