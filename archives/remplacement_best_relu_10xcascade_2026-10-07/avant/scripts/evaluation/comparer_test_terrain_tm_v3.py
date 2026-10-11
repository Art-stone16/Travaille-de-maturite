
# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401

import csv
import hashlib
import json
from pathlib import Path

root = Path('resultats/photos_terrain/Test_terrain_TM_V3')
source = Path('donnees/brutes/terrain/historiques/Test_terrain_TM_V3.jpg')

def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter=';'))

def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
        w.writeheader()
        w.writerows(rows)

files = sorted(root.glob('*/03_resultats/resultats_terrain.csv'))
assert len(files) == 8
reference = read(files[0])
assert len(reference) == 30
ordered = sorted(reference, key=lambda r: int(r['y']))
annotations = []
for line in range(3):
    group = ordered[line*10:(line+1)*10]
    assert max(int(r['y']) for r in group) - min(int(r['y']) for r in group) < 150
    for col, r in enumerate(sorted(group, key=lambda r: int(r['x'])), 1):
        annotations.append({'numero': int(r['numero']), 'ligne': line+1, 'colonne': col, 'chiffre_reel': col-1, 'detecte': 1, 'origine_chiffre_reel': 'lecture_visuelle_photo_3_lignes_0_a_9', **{k: int(r[k]) for k in ['x','y','largeur','hauteur']}})
write(root / 'annotations_reference.csv', annotations)
summary, details = [], []
for file in files:
    rows = read(file)
    assert len(rows) == 30
    lookup = {int(r['numero']): r for r in rows}
    assert set(lookup) == {a['numero'] for a in annotations}
    name = rows[0]['modele']
    score, errors = 0, []
    for a in annotations:
        r = lookup[a['numero']]
        assert all(int(r[k]) == a[k] for k in ['x','y','largeur','hauteur'])
        for k in ['planche_qc','matrice_0_1','image_28_gris']:
            assert (file.parent.parent / r[k]).is_file()
        correct = int(int(r['prediction']) == a['chiffre_reel'])
        score += correct
        details.append({'modele': name, 'execution': r['execution'], 'numero': a['numero'], 'ligne': a['ligne'], 'colonne': a['colonne'], 'chiffre_reel': a['chiffre_reel'], 'detecte': 1, 'prediction': int(r['prediction']), 'correct': correct, 'confiance': r['confiance'], 'origine_chiffre_reel': a['origine_chiffre_reel']})
        if not correct:
            errors.append(f"L{a['ligne']} C{a['colonne']}: {a['chiffre_reel']}→{r['prediction']}")
    summary.append({'modele': name, 'chiffres_photo': 30, 'chiffres_detectes': 30, 'non_detectes': 0, 'corrects': score, 'erreurs_reconnaissance': len(errors), 'accuracy_detectes_pourcent': f'{score/30*100:.2f}', 'accuracy_photo_pourcent': f'{score/30*100:.2f}', 'detail_erreurs_reconnaissance': ' ; '.join(errors), 'execution': rows[0]['execution'], 'image_annotee': (file.parent / 'resultat_annote.jpg').relative_to(root).as_posix(), 'csv_original': file.relative_to(root).as_posix(), 'chemin_modele': rows[0]['chemin_modele']})
summary.sort(key=lambda r: (-r['corrects'], r['modele'].lower()))
assert len({r['modele'] for r in summary}) == 8
write(root / 'comparaison_8_modeles.csv', summary)
write(root / 'predictions_8_modeles_evaluees.csv', details)
metadata = {'image_source': source.as_posix(), 'sha256_image': hashlib.sha256(source.read_bytes()).hexdigest(), 'nombre_chiffres_photo': 30, 'nombre_detectes': 30, 'annotation': 'Lecture visuelle de la photo et du diagnostic: trois lignes de 0 à 9. Rectangles triés par y, groupes de 10 triés par x.', 'nombre_modeles': 8, 'executions': [{'modele': r['modele'], 'execution': r['execution'], 'sha256_modele': hashlib.sha256(Path(r['chemin_modele']).read_bytes()).hexdigest()} for r in summary]}
(root / 'comparaison_provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
lines = ['# Test terrain — Test_terrain_TM_V3', '', 'Test du 6 octobre 2026 sur les huit modèles actifs (`best_model.keras`), avec les mêmes réglages que les tests précédents.', '', 'La photo contient 30 chiffres : trois lignes de 0 à 9. Les 30 chiffres sont détectés sans composante supplémentaire. Les annotations ont été vérifiées visuellement sur la photo et le diagnostic de détection. Les huit modèles reçoivent les mêmes rectangles et les mêmes entrées 28×28.', '', '| Modèle | Corrects | Réussite | Erreurs de reconnaissance | Image annotée |', '|---|---:|---:|---|---|']
for r in summary:
    lines.append(f"| {r['modele']} | {r['corrects']}/30 | {r['accuracy_photo_pourcent'].replace('.', ',')} % | {r['detail_erreurs_reconnaissance'] or 'Aucune'} | [Voir]({r['image_annotee']}) |")
lines += ['', 'L = ligne, C = colonne, de gauche à droite. La colonne 1 correspond au chiffre 0. Les scores concernent uniquement cette photo.', '', '## Fichiers', '', '- `comparaison_8_modeles.csv` : scores et erreurs par modèle.', '- `predictions_8_modeles_evaluees.csv` : les 240 prédictions évaluées.', '- `annotations_reference.csv` : chiffres réels et correspondance avec les rectangles.', '- `comparaison_provenance.json` : provenance et empreintes des fichiers.', '- Chaque dossier horodaté contient les diagnostics, les contrôles du prétraitement, une photo annotée et les résultats originaux.', '', 'Les exécutions originales utilisent `--chiffre-reel inconnu`, car cette option accepte une seule valeur pour toute la photo. Leur CSV, leur résumé et le journal global conservent donc la valeur réelle vide. Les annotations par chiffre et les scores sont enregistrés séparément dans les fichiers comparatifs.', '']
(root / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
assert len(details) == 240
journal = read(root.parent / 'journal_global_tests_terrain.csv')
new_rows = [r for r in journal if Path(r['image_source']).name == source.name and r['experience'] == 'Test_terrain_TM_V3_8_modeles']
assert len(new_rows) == 240
for r in summary:
    print(f"{r['modele']}: {r['corrects']}/30 ({r['accuracy_photo_pourcent']} %) — {r['detail_erreurs_reconnaissance'] or 'aucune erreur'}")
print('Vérification réussie : 8 exécutions, 240 prédictions évaluées et 240 lignes dans le journal global, fichiers de contrôle présents.')
