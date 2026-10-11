# Comparaisons pour le travail de maturité

- [Configuration des huit modèles](comparatifs_modeles.md).
- [Rôle des scripts et modules](comparatifs_scripts.md).
- [Comparaison Cascade Top-N](cascade_top_n/comparaison_modeles.md).
- [Photo Test_terrain_TM](terrain/test_terrain_tm.md).
- [Photo Test_terrain_TM_V2](terrain/test_terrain_tm_v2.md).
- [Photo Test_terrain_TM_V3](terrain/test_terrain_tm_v3.md).

Les résultats détaillés, CSV, images et paramètres restent dans `resultats/`. Les trois comparatifs terrain sont des copies destinées au TM des comptes rendus conservés auprès de leurs données. Les anciens PDF sont rangés dans `archives/pdf/` lorsqu’ils sont présents.

Pour régénérer les comparatifs Cascade depuis la racine :

```bash
.venv/bin/python scripts/visualisation/generer_rapports_cascade_md.py
```
