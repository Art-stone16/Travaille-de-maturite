"""Actualise les comparatifs terrain à partir des annotations existantes.

Le nom historique du script est conservé. --dossier accepte TM, TM_V2 ou TM_V3.
Seules les lignes du modèle actualisé sont remplacées ; les autres sont gardées.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401

import argparse
import csv
import hashlib
import json
from pathlib import Path
from reconnaissance_chiffres import config


def lire(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter=';'))


def ecrire(path, rows, fields=None):
    temporaire = path.with_suffix(path.suffix + '.tmp')
    with temporaire.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), delimiter=';')
        writer.writeheader()
        writer.writerows(rows)
    temporaire.replace(path)


def actualiser(root, nom='best_relu_10xcascade'):
    annotations = lire(root / 'annotations_reference.csv')
    anciens_scores = lire(root / 'comparaison_8_modeles.csv')
    anciens_details = lire(root / 'predictions_8_modeles_evaluees.csv')
    provenance = json.loads((root / 'comparaison_provenance.json').read_text())
    source = config.PROJECT_ROOT / provenance['image_source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance['sha256_image']
    candidats = []
    for path in root.glob('*/03_resultats/resultats_terrain.csv'):
        rows = lire(path)
        if rows and rows[0]['modele'] == nom:
            candidats.append((path, rows))
    path, rows = max(candidats, key=lambda pair: pair[0].parent.parent.name)
    lookup = {r['numero']: r for r in rows}
    assert len(lookup) == len(rows) == sum(int(a.get('detecte', 1)) for a in annotations)
    details, erreurs, manques = [], [], []
    corrects = 0
    champs_details = list(anciens_details[0])
    for a in annotations:
        detecte = int(a.get('detecte', 1))
        if detecte:
            r = lookup[a['numero']]
            assert all(int(r[k]) == int(a[k]) for k in ['x', 'y', 'largeur', 'hauteur'])
            for k in ['planche_qc', 'matrice_0_1', 'image_28_gris']:
                assert (path.parent.parent / r[k]).is_file()
            correct = int(r['prediction'] == a['chiffre_reel'])
            corrects += correct
            if not correct:
                erreurs.append(f"L{a['ligne']} C{a['colonne']}: {a['chiffre_reel']}→{r['prediction']}")
        else:
            r, correct = {}, 0
            manques.append(f"L{a['ligne']} C{a['colonne']}: {a['chiffre_reel']} (rejet {a.get('raison_rejet', '')})")
        entree = {'modele': nom, 'execution': rows[0]['execution'], 'numero': a['numero'],
                  'ligne': a['ligne'], 'colonne': a['colonne'], 'chiffre_reel': a['chiffre_reel'],
                  'detecte': str(detecte), 'prediction': r.get('prediction', ''), 'correct': str(correct),
                  'confiance': r.get('confiance', ''), 'raison_rejet': a.get('raison_rejet', ''),
                  'origine_chiffre_reel': a['origine_chiffre_reel']}
        details.append({k: entree[k] for k in champs_details})
    score = next(r.copy() for r in anciens_scores if r['modele'] == nom)
    nouvelles_valeurs = {'corrects': str(corrects), 'erreurs_reconnaissance': str(len(erreurs)),
                  'accuracy_detectes_pourcent': f'{corrects / len(rows) * 100:.2f}',
                  'accuracy_photo_pourcent': f'{corrects / len(annotations) * 100:.2f}',
                  'detail_erreurs_reconnaissance': ' ; '.join(erreurs),
                  'erreurs': str(len(erreurs)), 'accuracy_pourcent': f'{corrects / len(annotations) * 100:.2f}',
                  'detail_erreurs': ' ; '.join(erreurs),
                  'execution': rows[0]['execution'],
                  'image_annotee': (path.parent / 'resultat_annote.jpg').relative_to(root).as_posix(),
                  'csv_original': path.relative_to(root).as_posix(), 'chemin_modele': rows[0]['chemin_modele']}
    score.update({k: v for k, v in nouvelles_valeurs.items() if k in score})
    if 'detail_non_detectes' in score:
        score['detail_non_detectes'] = ' ; '.join(manques)
    scores = [r for r in anciens_scores if r['modele'] != nom] + [score]
    scores.sort(key=lambda r: (-int(r['corrects']), r['modele'].lower()))
    details = [r for r in anciens_details if r['modele'] != nom] + details
    assert len(scores) == 8 and len(details) == len(annotations) * 8
    ecrire(root / 'comparaison_8_modeles.csv', scores)
    ecrire(root / 'predictions_8_modeles_evaluees.csv', details, champs_details)
    for execution in provenance['executions']:
        if execution['modele'] == nom:
            execution.update(execution=score['execution'], sha256_modele=hashlib.sha256(Path(score['chemin_modele']).read_bytes()).hexdigest())
    provenance['actualisation'] = {'date': '2026-10-07', 'modele': nom,
                                 'annotations_conservees': True, 'autres_modeles_conserves': True}
    (root / 'comparaison_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    lines = [f'# Test terrain — {root.name}', '',
             'Comparatif actualisé le 7 octobre 2026 après remplacement du modèle ×10. Les sept autres résultats et les annotations de référence sont conservés.', '',
             f"La photo contient {len(annotations)} chiffres ; {len(rows)} sont détectés. Les rectangles et les entrées 28×28 ont été vérifiés identiques à ceux du test historique. Chaque chiffre non détecté est compté comme un échec sur la photo.", '',
             '| Modèle | Corrects / photo | Réussite photo | Réussite détectés | Erreurs de reconnaissance | Image annotée |',
             '|---|---:|---:|---:|---|---|']
    for r in scores:
        photo = r.get('accuracy_photo_pourcent', r.get('accuracy_pourcent'))
        detectes = r.get('accuracy_detectes_pourcent', r.get('accuracy_pourcent'))
        erreurs_modele = r.get('detail_erreurs_reconnaissance', r.get('detail_erreurs'))
        lines.append(f"| {r['modele']} | {r['corrects']}/{len(annotations)} | {photo} % | {detectes} % | {erreurs_modele or 'Aucune'} | [Voir]({r['image_annotee']}) |")
    lines += ['', 'L = ligne, C = colonne. Les taux portent uniquement sur cette photo.',
              ('Chiffres non détectés : ' + ' ; '.join(manques)) if manques else 'Aucun chiffre non détecté.', '',
              'Ces photos ne figurent pas dans les sources Cascade du nouveau modèle. Ce sont des contrôles terrain sur des photos distinctes ; un recouvrement d’auteur des écritures ne peut pas être établi avec les métadonnées disponibles.', '',
              'Le ×2 enregistré conserve son protocole historique ; la comparaison ×2/×10 ne permet pas d’isoler causalement le facteur de répétition.', '',
              'Les CSV bruts utilisent `--chiffre-reel inconnu` : les annotations par chiffre et les scores sont dans `annotations_reference.csv`, `predictions_8_modeles_evaluees.csv` et `comparaison_8_modeles.csv`. Le nouveau résumé concerne le nouveau ×10 ; les autres exécutions gardent leur date historique.', '']
    texte = '\n'.join(lines)
    (root / 'README.md').write_text(texte, encoding='utf-8')
    # Version du rapport dans le dossier habituel, avec les mêmes cibles de liens.
    for r in scores:
        texte = texte.replace(f"]({r['image_annotee']})", f"](../../../resultats/photos_terrain/{root.name}/{r['image_annotee']})")
    (config.RAPPORTS_COMPARAISONS / 'terrain' / f'{root.name.lower()}.md').write_text(texte, encoding='utf-8')
    print(f"{root.name}: {corrects}/{len(annotations)} ; erreurs : {' ; '.join(erreurs) or 'aucune'}")
    return score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dossier', type=Path, default=config.SORTIES_TESTS_TERRAIN / 'Test_terrain_TM_V3')
    parser.add_argument('--modele-actualise', default='best_relu_10xcascade')
    args = parser.parse_args(argv)
    actualiser(args.dossier, args.modele_actualise)


if __name__ == '__main__':
    main()
