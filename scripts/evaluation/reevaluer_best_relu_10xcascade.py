"""Remplace un ×10 vérifié et rejoue les protocoles répertoriés dans l'archive.

L'archive est créée avant la modification du script. Ce programme refuse un
remplacement déjà effectué et conserve une trace des entrées, sorties et hashes.
"""

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from reconnaissance_chiffres import config

NOM = 'best_relu_10xcascade'
ROOT = config.PROJECT_ROOT
ARCHIVE = ROOT / 'archives/remplacement_best_relu_10xcascade_2026-10-07'
STAGE = config.SORTIES_ENTRAINEMENTS_MANUELS / 'remplacement_best_relu_10xcascade_2026-10-07'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lire_csv(path, delimiter=';'):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def ecrire_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def installer(archive, stage):
    import numpy as np
    import keras
    from reconnaissance_chiffres.modeles import charger_modele
    active = config.MODELES_VALIDES / NOM
    source = stage / 'modele_provisoire'
    metadata = json.loads((source / 'entrainement.json').read_text())
    assert metadata['facteur_repetition'] == 10 and metadata['cascade_total'] == 12170
    assert metadata['augmentation'] is None and metadata['cascade_validation'] == metadata['cascade_test'] == 0
    with np.load(ROOT / '.cache/keras/datasets/mnist.npz') as data:
        entrees = data['x_test'][:32].astype('float32')[..., None] / 255
    for filename in ['best_model.keras', 'final_model.keras']:
        assert sha(source / filename) == metadata['verification_rechargement'][filename]['sha256']
        model = charger_modele(source / filename)
        preds = model.predict(entrees, verbose=0)
        assert preds.shape == (32, 10) and np.isfinite(preds).all()
        np.testing.assert_allclose(preds.sum(axis=1), 1, atol=1e-5)
    original = archive / 'avant/modeles/actifs' / NOM
    assert all(sha(p) == sha(original / p.name) for p in active.iterdir() if p.is_file())
    temp = config.MODELES_VALIDES / f'.{NOM}_nouveau'
    shutil.copytree(source, temp)
    ancien = archive / 'ancien_dossier_actif'
    active.rename(ancien)
    try:
        temp.rename(active)
    except BaseException:
        ancien.rename(active)
        raise
    for filename in ['best_model.keras', 'final_model.keras']:
        assert sha(active / filename) == sha(source / filename)
        model = charger_modele(active / filename)
        assert np.isfinite(model.predict(entrees, verbose=0)).all()
    shutil.copy2(stage / 'courbes' / f'{NOM}_training_curves.png',
                 config.SORTIES_ENTRAINEMENTS_MANUELS / 'courbes' / f'{NOM}_training_curves.png')
    ecrire_json(archive / 'installation.json', {
        'date': datetime.now(ZoneInfo('Europe/Zurich')).isoformat(),
        'actif': str(active.relative_to(ROOT)), 'provisoire': str(source.relative_to(ROOT)),
        'ancien_sha256': sha(original / 'best_model.keras'),
        'nouveau_sha256': sha(active / 'best_model.keras'),
        'verification_rechargement_avant_et_apres': True})
    keras.backend.clear_session()


def retirer_resultats_historiques(archive, inventaire):
    chemins = inventaire['cascade'] + [t['dossier'] for t in inventaire['terrain']]
    for chemin in chemins:
        actuel = ROOT / chemin
        copie = archive / 'avant' / chemin
        if not actuel.exists():
            continue
        assert copie.is_dir()
        assert all(sha(f) == sha(copie / f.relative_to(actuel))
                   for f in actuel.rglob('*') if f.is_file())
        shutil.rmtree(actuel)
    journal = config.SORTIES_TESTS_TERRAIN / 'journal_global_tests_terrain.csv'
    rows = lire_csv(journal)
    executions_historiques = {t['parametres']['execution']['identifiant'] for t in inventaire['terrain']}
    with journal.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
        writer.writeheader()
        writer.writerows(r for r in rows if not (r['modele'] == NOM and r['execution'] in executions_historiques))


def executer(archive, nom_log, arguments):
    commande = [sys.executable, *map(str, arguments)]
    env = {**os.environ, 'TF_NUM_INTRAOP_THREADS': '6', 'TF_NUM_INTEROP_THREADS': '2'}
    with (archive / f'{nom_log}.log').open('w') as log:
        subprocess.run(commande, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'OK : {nom_log}', flush=True)
    return commande


def verifier_terrain(ancien, nouveau):
    old = lire_csv(ancien / '03_resultats/resultats_terrain.csv')
    new = lire_csv(nouveau / '03_resultats/resultats_terrain.csv')
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert all(a[k] == b[k] for k in ['numero', 'x', 'y', 'largeur', 'hauteur'])
        for key in ['matrice_0_1', 'image_28_gris']:
            assert sha(ancien / a[key]) == sha(nouveau / b[key])
    return len(new)


def reevaluer(archive):
    inventaire = json.loads((archive / 'inventaire.json').read_text())
    if (archive / 'evaluations_reexecutees.json').exists():
        raise FileExistsError("La réévaluation est déjà enregistrée dans cette archive.")
    active = config.MODELES_VALIDES / NOM / 'best_model.keras'
    installation = json.loads((archive / 'installation.json').read_text())
    assert sha(active) == installation['nouveau_sha256']
    retirer_resultats_historiques(archive, inventaire)
    evaluations = []
    commande = executer(archive, 'mnist_best', ['scripts/evaluation/matrice_confusion.py',
        '--modele', active, '--mnist-path', '.cache/keras/datasets/mnist.npz',
        '--nom-evaluation', f'{NOM}_copies_exactes_2026-10-07', '--sauver-predictions'])
    evaluations.append({'type': 'MNIST test best', 'commande': commande,
                        'evaluation_independante': True, 'images': 10000})
    commande = executer(archive, 'cascade', ['scripts/evaluation/test_cascade_top_n.py',
        '--toutes', '--modele', active, '--nom-modele', NOM])
    # Vérifier les mêmes zones dans chaque feuille et préserver toutes les variantes.
    from reconnaissance_chiffres.rapports import charger
    nouveaux_ctn = charger(config.SORTIES_CASCADE_TOP_N)[NOM]
    for execution in nouveaux_ctn:
        ancien = next(p for p in inventaire['cascade'] if Path(p).parent.name == execution['feuille'])
        a = lire_csv(archive / 'avant' / ancien / 'predictions.csv')
        b = execution['lignes']
        assert len(a) == len(b)
        assert all(all(x[k] == y[k] for k in ['numero', 'x', 'y', 'largeur', 'hauteur', 'chiffre_attendu'])
                   for x, y in zip(a, b))
        evaluations.append({'type': 'Cascade Top-N', 'feuille': execution['feuille'],
                            'ancien': ancien, 'nouveau': str(execution['path'].parent.relative_to(ROOT)),
                            'zones': len(b), 'rectangles_identiques': True,
                            'evaluation_independante': False})
    for numero, t in enumerate(inventaire['terrain']):
        p = t['parametres']
        assert p['detection']['diagnostics_sauvegardes']
        source = Path(p['entrees']['image_source'])
        chiffre = p['entrees']['chiffre_reel']
        before = set(config.SORTIES_TESTS_TERRAIN.glob(f'{source.stem}/*/00_source/parametres.json'))
        commande = executer(archive, f'terrain_{numero}_{source.stem}', [
            'scripts/evaluation/test_condition_reelle.py', '--image', source,
            '--modele', active, '--nom-modele', NOM,
            '--nom-experience', p['execution']['experience'],
            '--chiffre-reel', 'inconnu' if chiffre is None else str(chiffre)])
        after = set(config.SORTIES_TESTS_TERRAIN.glob(f'{source.stem}/*/00_source/parametres.json'))
        nouveau_json, = after - before
        nouveau = nouveau_json.parents[1]
        actuel = json.loads(nouveau_json.read_text())
        assert actuel['detection']['configuration'] == p['detection']['configuration']
        assert actuel['pretraitement'] == p['pretraitement']
        assert actuel['entrees']['modele_sha256'] == sha(active)
        zones = verifier_terrain(archive / 'avant' / t['dossier'], nouveau)
        evaluations.append({'type': 'Terrain', 'image': source.name, 'ancien': t['dossier'],
                            'nouveau': str(nouveau.relative_to(ROOT)), 'zones': zones,
                            'commande': commande, 'pretraitement_identique': True,
                            'source_distincte_du_dataset_cascade': True,
                            'annotations_disponibles': (nouveau.parent / 'annotations_reference.csv').exists()})
        ecrire_json(archive / 'evaluations_en_cours.json', evaluations)
    for nom in ['Test_terrain_TM', 'Test_terrain_TM_V2', 'Test_terrain_TM_V3']:
        executer(archive, f'comparatif_{nom}', ['scripts/evaluation/comparer_test_terrain_tm_v3.py',
                 '--dossier', config.SORTIES_TESTS_TERRAIN / nom])
    executer(archive, 'comparaisons_images_ctn', ['scripts/visualisation/comparer_images_ctn.py'])
    ecrire_json(archive / 'evaluations_reexecutees.json', evaluations)
    return evaluations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--provisoire', type=Path, default=STAGE)
    parser.add_argument('--installer', action='store_true')
    parser.add_argument('--reevaluer', action='store_true')
    args = parser.parse_args(argv)
    if args.installer:
        installer(args.archive, args.provisoire)
        print('Modèle installé et rechargé.', flush=True)
    if args.reevaluer:
        reevaluer(args.archive)


if __name__ == '__main__':
    main()
