"""Recalcule le bilan du remplacement depuis les sources, sans réentraîner."""

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401

import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from contextlib import redirect_stdout

import numpy as np
from reconnaissance_chiffres import config
from reconnaissance_chiffres.rapports import charger
from scripts.evaluation.reevaluer_best_relu_10xcascade import verifier_terrain

ROOT = config.PROJECT_ROOT
ARCHIVE = ROOT / 'archives/remplacement_best_relu_10xcascade_2026-10-07'
SORTIE = ROOT / 'rapports/remplacement_best_relu_10xcascade_2026-10-07'
STAGE = config.SORTIES_ENTRAINEMENTS_MANUELS / 'remplacement_best_relu_10xcascade_2026-10-07'
EVAL = config.SORTIES_PERFORMANCES_PAR_CHIFFRE
NOM = 'best_relu_10xcascade'
DATE = datetime.now(ZoneInfo('Europe/Zurich')).isoformat(timespec='seconds')


def lire(path, separateur=','):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=separateur))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normaliser_valeurs(rows):
    result = []
    for row in rows:
        new = {}
        for k, v in row.items():
            try:
                new[k] = float(v) if '.' in v else int(v)
            except (ValueError, TypeError):
                new[k] = v
        result.append(new)
    return result


def calculer():
    nouveau = json.loads((config.MODELES_VALIDES / NOM / 'entrainement.json').read_text())
    ancien = json.loads((ARCHIVE / 'avant/modeles/actifs' / NOM / 'entrainement.json').read_text())
    evaluations = [
        ('Ancien ×10 augmenté, meilleur checkpoint', 'best_relu_10xcascade_ancien_checkpoint_reevalue_2026-10-07'),
        ('Nouveau ×10 exact, meilleur checkpoint', 'best_relu_10xcascade_copies_exactes_2026-10-07'),
    ]
    mnist, protocoles = [], []
    for nom, dossier in evaluations:
        d = EVAL / dossier
        rows = lire(d / 'donnees/predictions_test.csv')
        corrects = sum(r['classe_reelle'] == r['classe_predite'] for r in rows)
        resume = lire(d / 'donnees/resume_global.csv')[0]
        assert len(rows) == 10000 and abs(float(resume['accuracy_globale']) - corrects/10000) < 1e-10
        meta = json.loads((d / 'configuration.json').read_text())
        protocoles.append(meta['protocol_id'])
        mnist.append({'modele': nom, 'images': len(rows), 'corrects': corrects,
                      'erreurs': len(rows) - corrects, 'accuracy': corrects/len(rows),
                      'f1_macro': float(resume['f1_macro']), 'source': str(d.relative_to(ROOT)),
                      'modele_sha256': meta['modele']['sha256']})
    assert len(set(protocoles)) == 1
    par_chiffre = normaliser_valeurs(lire(EVAL / evaluations[1][1] / 'donnees/metriques_par_classe.csv'))
    dataset = config.DONNEES_PREPAREES / 'cascade/cascade_top_n_v1'
    manifest = lire(dataset / 'manifest.csv')
    lookup = {(r['source_image'], *(int(r[k]) for k in ['x','y','largeur','hauteur'])): r
              for r in manifest}
    sources_ctn = [
        ('Ancien ×10 augmenté', charger(ARCHIVE / 'avant/resultats/cascade_top_n')[NOM]),
        ('Nouveau ×10 exact', charger(config.SORTIES_CASCADE_TOP_N)[NOM]),
    ]
    cascade, cascade_chiffres = [], []
    for nom, feuilles in sources_ctn:
        rows = [r for f in feuilles for r in f['lignes']]
        inclus, exclus = [], []
        for r in rows:
            cle = (r['image_source'], *(int(r[k]) for k in ['x','y','largeur','hauteur']))
            assert cle in lookup
            entree = lookup[cle]
            assert int(entree['label']) == int(r['chiffre_attendu'])
            (inclus if entree['decision'] == 'inclure' else exclus).append(r)
        assert len(rows) == 1227 and len(inclus) == 1217 and len(exclus) == 10
        ligne = {'modele': nom, 'zones': len(rows), 'inclus_train': len(inclus),
                 'zones_exclues': len(exclus), 'evaluation_independante': False}
        for n in [1,2,3,5,10]:
            bons = sum(int(r['rang_chiffre_attendu']) <= n for r in rows)
            ligne[f'top_{n}_corrects'] = bons
            ligne[f'top_{n}'] = bons / len(rows)
        ligne['top_1_inclus_train_corrects'] = sum(int(r['rang_chiffre_attendu']) == 1 for r in inclus)
        ligne['top_1_exclus_corrects'] = sum(int(r['rang_chiffre_attendu']) == 1 for r in exclus)
        cascade.append(ligne)
        if nom == 'Nouveau ×10 exact':
            for chiffre in range(10):
                rs = [r for r in rows if int(r['chiffre_attendu']) == chiffre]
                cascade_chiffres.append({'chiffre': chiffre, 'zones': len(rs),
                    'top_1_corrects': sum(int(r['rang_chiffre_attendu']) == 1 for r in rs),
                    **{f'top_{n}': sum(int(r['rang_chiffre_attendu']) <= n for r in rs) / len(rs) for n in [1,3,5]}})
    inv = json.loads((ARCHIVE / 'inventaire.json').read_text())
    terrain = []
    for t in inv['terrain']:
        image = Path(t['parametres']['entrees']['image_source'])
        ds = []
        for racine, label in [(config.SORTIES_TESTS_TERRAIN, 'Nouveau ×10 exact')]:
            name = NOM
            candidats = [q for q in (racine / image.stem).glob('*/00_source/parametres.json')
                         if json.loads(q.read_text())['entrees']['nom_modele'] == name]
            q = max(candidats)
            d = q.parents[1]
            raw = lire(d / '03_resultats/resultats_terrain.csv', ';')
            ds.append(d)
            ligne = {'photo': image.stem, 'modele': label, 'zones_detectees': len(raw),
                     'dossier': str(d.relative_to(ROOT)), 'source_distincte_cascade': True}
            annotations = config.SORTIES_TESTS_TERRAIN / image.stem / 'annotations_reference.csv'
            if annotations.exists():
                truth = lire(annotations, ';'); pred = {r['numero']: r for r in raw}
                bons = 0
                for a in truth:
                    if int(a.get('detecte', 1)):
                        r = pred[a['numero']]
                        assert all(int(a[k]) == int(r[k]) for k in ['x','y','largeur','hauteur'])
                        bons += int(a['chiffre_reel'] == r['prediction'])
                ligne.update(chiffres_photo=len(truth), corrects=bons,
                             accuracy_photo=bons/len(truth), accuracy_detectes=bons/len(raw),
                             non_detectes=len(truth)-len(raw), annotations_disponibles=True)
            else:
                ligne.update(annotations_disponibles=False, corrects=None, accuracy_photo=None)
            terrain.append(ligne)
        verifier_terrain(ds[0], ARCHIVE / 'avant' / t['dossier'])
    audit = json.loads((ARCHIVE / 'audit_donnees.json').read_text())
    preservation = json.loads((ARCHIVE / 'controle_preservation.json').read_text())
    return {'date': DATE, 'nouveau': nouveau, 'ancien': ancien,
            'mnist': mnist, 'mnist_par_chiffre': par_chiffre, 'cascade': cascade,
            'cascade_par_chiffre': cascade_chiffres, 'terrain': terrain,
            'audit': audit, 'preservation': preservation,
            'protocole_mnist_commun': protocoles[0]}


def tableau(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def produire(data):
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / 'resultats_verifies.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    n = data['nouveau']
    pct = lambda x: f'{100*x:.2f} %'.replace('.', ',')
    sections = []
    sections.append(('resume', 'Modèle ×10 remplacé et évaluations régénérées',
        'Le modèle actif utilise dix exemplaires exacts par chiffre Cascade : un original et neuf copies identiques. '
        f"Le meilleur checkpoint atteint {pct(n['mnist_test_accuracy'])} sur les 10 000 images MNIST test, soit 9 910 prédictions correctes et 90 erreurs. "
        'Les anciens poids, scripts et résultats sont archivés. Les autres modèles actifs sont conservés. '
        'Les contrôles CTN utilisent les sources d’entraînement et doivent être lus séparément des évaluations indépendantes.'))
    sections.append(('definitions', 'Périmètre et dénominateurs',
        'MNIST test est le jeu officiel de 10 000 images, sans Cascade ; accuracy = nombre de prédictions exactes / 10 000. '
        'Pour chaque chiffre, la sensibilité mesure la proportion de vrais chiffres correctement reconnus ; la spécificité mesure la proportion des autres chiffres non confondus avec lui. '
        'Les intervalles par chiffre sont les intervalles de Wilson à 95 % du script existant.\n\n'
        'CTN Top-N = nombre de zones où le chiffre de la feuille se trouve parmi les N premiers choix / 1 227 zones détectées. '
        'Les 1 217 zones validées ont servi à l’entraînement ; dix zones ont été exclues. Les douze feuilles totalisent 1 201 chiffres déclarés attendus : cette différence de comptage ne donne pas un nombre exact de faux positifs. '
        'Pour les photos annotées, le taux sur la photo utilise tous les chiffres annotés ; un chiffre non détecté est un échec.'))
    sections.append(('methode', 'Validation séparée et copies exactes',
        'La version ×2 historique mélangeait les copies avant la validation : 307 originaux étaient communs aux deux partitions, et 357 copies étaient réservées à la validation. '
        'Cette fuite n’a pas été reproduite. Les deux scripts appellent désormais la même routine, avec des facteurs respectifs de 2 et de 10. '
        'Une permutation de graine 42 réserve d’abord 9 000 images MNIST à la validation ; toutes les copies Cascade sont ensuite ajoutées au train. '
        'Les 1 217 matrices ont été comparées aux lignes du manifeste et aux étiquettes. Les empreintes des images Cascade n’ont aucun recouvrement avec MNIST validation ou test.\n\n'
        'Architecture : Conv2D 64 filtres 5×5 ReLU, BatchNormalization, MaxPooling 2×2, Conv2D 64 filtres 3×3 ReLU, BatchNormalization, MaxPooling 2×2, Flatten, Dropout 0,6, Dense 10 softmax. '
        'Total : 55 114 paramètres. Adam à 0,001, SparseCategoricalCrossentropy, batch 128, 30 époques maximum. '
        'Checkpoint sur val_loss ; EarlyStopping sur val_loss, patience 2, sans restauration automatique des meilleurs poids. '
        'La graine globale 42 fixe Python, NumPy et Keras/TensorFlow ; les opérations déterministes sont demandées. Le test utilise explicitement best_model.keras.\n\n' +
        tableau(['Partition', 'Nouveau ×10'], [
            ['MNIST entraînement', n['mnist_train']],
            ['Cascade originaux', 1217], ['Copies supplémentaires', 10953],
            ['Cascade entraînement total', 12170], ['Entraînement total', 63170],
            ['MNIST validation', 9000], ['Cascade validation / test', '0 / 0'],
            ['MNIST test', 10000], ['Époques réalisées / retenue', '11 / 9']]) +
        '\n\nRépartition des originaux par chiffre 0 à 9 : 98, 100, 228, 102, 99, 103, 187, 100, 100, 100. Chaque effectif est multiplié par dix dans le train ×10.'))
    sections.append(('mnist', 'MNIST : meilleur checkpoint et état final distingués',
        tableau(['État', 'Corrects', 'Erreurs', 'Accuracy'], [[r['modele'], r['corrects'], r['erreurs'], pct(r['accuracy'])] for r in data['mnist']]) +
        f"\n\nL’état final du nouveau ×10 atteint {pct(n['mnist_test_final']['accuracy'])}, avec une loss {n['mnist_test_final']['loss']:.6f}. "
        f"Le meilleur checkpoint a une loss {n['mnist_test_loss']:.6f}. L’ancien score publié de 99,11 % et sa loss 0,030425 concernaient l’état final. "
        'L’ancien meilleur checkpoint a été réévalué ici sur les mêmes 10 000 entrées : son accuracy agrégée est aussi de 99,10 %, mais les erreurs et métriques par chiffre diffèrent. '
        ''))
    sections.append(('classes', 'Métriques MNIST du nouveau modèle par chiffre',
        tableau(['Chiffre', 'Effectif', 'Corrects', 'FN', 'Sensibilité', 'Spécificité', 'Précision', 'F1'],
            [[r['classe'],r['support'],r['tp'],r['fn'],pct(r['sensibilite']),pct(r['specificite']),pct(r['precision']),pct(r['f1'])] for r in data['mnist_par_chiffre']]) +
        '\n\nLe chiffre 2 a ici la sensibilité la plus faible (1 015/1 032). Les matrices en effectifs et normalisées, les dix scores de chaque image et les intervalles par chiffre sont exportés dans le dossier MNIST.'))
    sections.append(('cascade', 'CTN : contrôles sur les sources d’entraînement',
        tableau(['Expérience', 'Top-1', 'Top-2', 'Top-3', 'Top-5', 'Top-1 sur validés'],
            [[r['modele'], *(f"{r[f'top_{k}_corrects']}/1227 ({pct(r[f'top_{k}'])})" for k in [1,2,3,5]),
              f"{r['top_1_inclus_train_corrects']}/1217"] for r in data['cascade']]) +
        '\n\nCes résultats mesurent la reconnaissance de données issues des mêmes feuilles que l’entraînement. '
        'Les dix zones exclues sont détaillées séparément dans les données du bilan. Les taux complets restent ceux du protocole historique pour permettre la comparaison ; ils ne deviennent pas des scores indépendants.\n\n' +
        tableau(['Chiffre','Zones','Top-1','Top-3','Top-5'], [[r['chiffre'],r['zones'],pct(r['top_1']),pct(r['top_3']),pct(r['top_5'])] for r in data['cascade_par_chiffre']])))
    terrain_rows = []
    for r in data['terrain']:
        terrain_rows.append([r['photo'],r['modele'],r['zones_detectees'],
                             f"{r['corrects']}/{r.get('chiffres_photo')}" if r['annotations_disponibles'] else 'Non calculable',
                             pct(r['accuracy_photo']) if r['annotations_disponibles'] else 'Annotations absentes'])
    sections.append(('terrain', 'Les cinq protocoles terrain sont rejoués',
        tableau(['Photo','Modèle','Détectés','Corrects / photo','Réussite photo'],terrain_rows) +
        '\n\nLes 139 zones des cinq tests ×10 ont les mêmes rectangles et les mêmes entrées 28×28 que leurs exécutions historiques. '
        'Les images annotées, contrôles en niveaux de gris, matrices binaires de contrôle, CSV et résumés ont été régénérés. '
        'Les trois annotations du TM sont conservées à l’identique ; TM_V2 garde le 4 non détecté de la dernière ligne. '
        'Les photos terrain sont distinctes des douze sources Cascade ; l’identité des auteurs des écritures ne peut pas être vérifiée. '
        'Les deux photos non annotées permettent de comparer les prédictions et le prétraitement, mais ne permettent pas de calculer un taux de réussite fiable.'))
    sections.append(('preservation', 'Archive, chemins actifs et résultats conservés',
        'Archive : `archives/remplacement_best_relu_10xcascade_2026-10-07/`. '
        'Elle contient le script original, les poids best/final, les métadonnées, les cinq tests terrain, les douze tests CTN, les courbes, motifs et anciens comparatifs. '
        'Les anciens dossiers d’exécution ×10 ont été retirés des résultats actifs après vérification des copies archivées. Les anciennes lignes ×10 du journal terrain y sont aussi archivées.\n\n'
        'Le nouveau modèle a été enregistré provisoirement, rechargé, puis installé sous `modeles/actifs/best_relu_10xcascade/best_model.keras` et `final_model.keras`. '
        'Les empreintes enregistrées dans les tests MNIST, CTN, terrain et motifs correspondent au fichier actif. '
        'Les 18 fichiers des autres modèles et 5 693 sorties individuelles sont identiques ; leurs lignes dans les trois comparatifs terrain sont également identiques. '
        'Les douze montages CTN à huit modèles, les rapports CTN et terrain et la comparaison de motifs ont été actualisés. '
        'Le ×2 actif conserve ses poids historiques.'))
    sections.append(('fichiers', 'Fichiers modifiés et résultats régénérés',
        '- `scripts/entrainement/train_best_relu_10xcascade.py` et `train_best_relu_2xcascade.py` : appels à la routine commune ; seul le facteur varie.\n'
        '- `src/reconnaissance_chiffres/entrainement_cascade.py` : copies exactes, séparation des partitions, contrôles, entraînement et sauvegardes.\n'
        '- `tests/test_injection_cascade.py` : dix occurrences par original, labels, validation identique et exclusion de Cascade. Les 43 tests du dépôt passent.\n'
        '- `scripts/evaluation/test_cascade_top_n.py`, `test_condition_reelle.py` et `src/reconnaissance_chiffres/rapports.py` : empreintes, statut des contrôles et rapports.\n'
        '- `scripts/evaluation/comparer_test_terrain_tm_v3.py` : accepte les trois schémas terrain, conserve les annotations et les lignes des sept autres modèles.\n'
        '- `scripts/evaluation/reevaluer_best_relu_10xcascade.py` et `bilan_remplacement_relu_10x.py` : installation, relecture des protocoles, vérifications et bilan reproductible.\n'
        '- `scripts/visualisation/visualiser_motifs_classes.py` : recalcul sélectif du ×10, réutilisation des autres cartes et provenance.\n'
        '- `modeles/actifs/best_relu_10xcascade/` : nouveaux best/final, métadonnées, indices de partition et historique CSV.\n'
        '- `README.md`, `.gitignore`, `modeles/catalogue.md`, comparatifs des scripts et des modèles : documentation du remplacement et conservation des poids archivés.\n'
        '- Sorties : deux évaluations MNIST complètes (ancien checkpoint et nouveau), douze CTN ×10, cinq terrains ×10, leurs comparatifs, douze montages CTN et deux illustrations de motifs.\n'
        '- L’inventaire exhaustif des fichiers créés, modifiés ou déplacés est enregistré dans `fichiers_modifies.json`. Les chemins et correspondances avant/après des tests sont dans l’archive.'))
    sections.append(('limites', 'Limites de reproduction et de comparaison',
        '- Aucune étude de stabilité enregistrée ne cible le ×10. L’étude présente utilise 16/32 filtres, noyaux 5×5/5×5, dropout 0,4 et les graines 42 à 46 : elle est conservée. Aucun résultat de stabilité ×10 n’est inventé.\n'
        '- L’initialisation aléatoire des anciennes expériences ×2/×10 n’était pas enregistrée ; leur entraînement exact n’est pas reproductible à partir du dépôt. Les poids archivés permettent de rejouer leurs prédictions.\n'
        '- Le nouveau ×10 a été entraîné avec une seule graine ; aucune étude de stabilité supplémentaire n’a été réalisée.\n'
        '- Il n’existe pas de jeu Cascade indépendant dans le dataset préparé : tous les 1 217 exemples inclus sont marqués train. Aucune évaluation finale Cascade indépendante ne peut donc être produite avec ces fichiers.\n'
        '- Les fichiers de matrices MNIST de l’ancien ×10 n’étaient pas présents. Une évaluation complète de son checkpoint archivé a été effectuée pour rendre la comparaison actuelle explicite.\n'
        '- `test_terrain.jpg` et `image_reelle_Grand.jpg` n’ont pas d’annotations par chiffre : leurs métriques d’exactitude sont indisponibles.\n'
        '- Le manuscrit du TM n’est pas présent dans le dépôt. Les numéros de figures, tableaux et pages ne peuvent pas être établis.'))
    sections[-1] = (sections[-1][0], sections[-1][1], sections[-1][2] +
        '\n- Le rapport HTML est validé structurellement et contient une représentation lisible sans JavaScript. La vérification visuelle automatisée du lecteur HTML n’a pas pu être exécutée : le Chromium de vérification est absent et le Chrome installé n’a pas démarré dans ce runtime. Les figures PNG d’entraînement et MNIST ont été inspectées séparément.')
    sections.append(('obsolete', 'Éléments du TM à actualiser si tu les as repris',
        '1. La méthode ×10 avec rotations ±8° et translations ±1,5 pixel, ainsi que toute légende parlant de neuf variantes augmentées. La nouvelle méthode est un original et neuf copies identiques.\n'
        '2. L’ancienne description de la validation : mélange MNIST/Cascade avant réservation des 15 %. Les nouveaux effectifs sont 63 170 en train et 9 000 en validation MNIST ; les 12 170 entrées Cascade sont toutes en train.\n'
        '3. L’ancien score MNIST final ×10 : 99,11 %, loss 0,030425308 ; et toute confusion entre ce score et best_model.keras. Les nouvelles valeurs sont 99,10 % pour best et 99,23 % pour final.\n'
        '4. Les courbes ×10 de 14 époques : elles sont remplacées par 11 époques, checkpoint retenu à l’époque 9. La courbe ×2 active reste historique.\n'
        '5. La ligne ×10 des tableaux CTN : Top-1 98,53 %, Top-2 99,35 %, Top-3 99,51 %, Top-5 99,59 %. Les tableaux par chiffre et de confusions, ainsi que le classement des modèles, ont été régénérés. Les anciens et nouveaux CTN doivent être identifiés comme contrôles sur les sources d’entraînement.\n'
        '6. Toutes les images annotées et confidences ×10 des douze feuilles CTN, les douze montages à huit modèles et les cinq photos terrain. Les trois scores TM ×10 restent 27/27, 29/30 et 30/30, mais leurs confidences et fichiers d’exécution ont changé.\n'
        '7. La planche `planche_06_best_relu_10xcascade.png`, ses données de gradients et `comparaison_7_modeles.png`. Les cartes des six autres modèles sont conservées.\n'
        '8. Toute conclusion attribuant au seul facteur ×2/×10 un écart entre les modèles actifs historiques. Le ×2 actif conserve son protocole historique.\n\n'
        'Les nombres de paramètres et l’architecture du ×10 restent valables. Les PDF CTN archivés de juillet et du 5 septembre 2026 précèdent l’ancien ×10 du 25 septembre et ne sont pas concernés par son remplacement.'))
    sections.append(('suite', 'Résultats disponibles pour mettre à jour le TM',
        'Utiliser les nouveaux CSV, matrices et illustrations du ×10, en gardant la séparation entre MNIST test, contrôles CTN et terrain annoté. '
        'Le bilan et les scripts fournissent la provenance des chiffres sans modifier le manuscrit. '
        'Pour répondre à une question de stabilité, il faudrait une nouvelle expérience sur plusieurs graines ; pour mesurer la généralisation Cascade, il faudrait de nouvelles feuilles indépendantes avec des annotations.'))
    title = 'Remplacement de best_relu_10xcascade — 7 octobre 2026'
    markdown = '# '+title+'\n\n'+'\n\n'.join('## '+titre+'\n\n'+body for _,titre,body in sections)+'\n'
    (SORTIE / 'compte_rendu.md').write_text(markdown,encoding='utf-8')
    # A single canonical artifact powers the portable report; no custom renderer.
    source = {'id':'bilan_verifie','label':'Bilan recalculé depuis les données et poids enregistrés',
              'path':str((SORTIE/'resultats_verifies.json').relative_to(ROOT)),
              'query':{'engine':'Python','language':'python','executed_at':DATE,
                       'description':'Agrège les CSV MNIST, CTN et terrain ; vérifie les annotations et la conservation des autres modèles.',
                       'sql':'from scripts.evaluation.bilan_remplacement_relu_10x import calculer\nbilan = calculer()',
                       'tables_used':['predictions_test.csv','metriques_par_classe.csv','predictions.csv','resultats_terrain.csv','annotations_reference.csv'],
                       'filters':['MNIST test complet ; aucun sous-échantillonnage','CTN : douze feuilles dont deux variantes, toutes les zones du protocole historique','Terrain : annotations historiques conservées, non-détection comptée comme échec'],
                       'metric_definitions':['accuracy = prédictions exactes / images évaluées','sensibilité = TP / (TP + FN)','CTN Top-N = rang attendu ≤ N / toutes les zones détectées']}}
    blocs=[{'id':'titre','type':'markdown','body':'# '+title}]
    charts=[{'id':'sensibilite','title':'Sensibilité MNIST par chiffre','description':'Nouveau meilleur checkpoint ; dix classes, 10 000 images MNIST test. Les effectifs de classe sont conservés dans les données.',
             'type':'bar','dataset':'classes','encodings':{'x':{'field':'classe'},'y':{'field':'sensibilite'}},
             'options':{'orientation':'vertical','grouping':'single'},'sourceId':'bilan_verifie'},
            {'id':'validation','title':'Perte de validation MNIST par époque','description':'Nouveau ×10 ; 9 000 images de validation, 11 époques observées. Le checkpoint retenu est l’époque 9.',
             'type':'line','dataset':'courbes','encodings':{'x':{'field':'epoque'},'y':{'field':'val_loss'}},'sourceId':'bilan_verifie'}]
    for ident,titre,body in sections:
        blocs.append({'id':ident,'type':'markdown','sourceId':'bilan_verifie','body':'## '+titre+'\n\n'+body})
        if ident=='classes':blocs.append({'id':'graph_classes','type':'chart','chartId':'sensibilite'})
        if ident=='methode':blocs.append({'id':'graph_validation','type':'chart','chartId':'validation'})
    courbes=[{'epoque':i+1,**{k:v[i] for k,v in n['history'].items()},'mnist_validation':9000,'cascade_train':12170}
             for i in range(n['epochs_realises'])]
    # Le lecteur natif demande une requête SQL effectivement exécutée pour
    # chaque graphique : conserver un extrait vérifié et relire ses lignes.
    database = SORTIE / 'extrait_verifie.sqlite'
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        datasets = {}
        sources_sql = []
        for table, ident, rows in [('metriques_mnist', 'classes', data['mnist_par_chiffre']),
                                   ('courbes_entrainement', 'courbes', courbes)]:
            db.execute(f'DROP TABLE IF EXISTS {table}')
            columns = list(rows[0])
            db.execute(f"CREATE TABLE {table} ({', '.join(columns)})")
            db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})",
                           [[r[k] for k in columns] for r in rows])
            ordre = 'classe' if ident == 'classes' else 'epoque'
            query = f'SELECT * FROM {table} ORDER BY {ordre};'
            datasets[ident] = [dict(r) for r in db.execute(query).fetchall()]
            sources_sql.append({'id':ident+'_sql','label':table,'path':str(database.relative_to(ROOT)),
                                'query':{'engine':'SQLite','language':'sql','sql':query,
                                         'executed_at':DATE,'tables_used':[table],
                                         'description':'Extrait vérifié des résultats du nouveau modèle ×10.',
                                         'metric_definitions':['sensibilite = TP / support ; pertes issues de history sans transformation'],
                                         'filters':['Toutes les classes MNIST test ; toutes les époques observées']}})
    charts[0]['sourceId'] = 'classes_sql'
    charts[1]['sourceId'] = 'courbes_sql'
    artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':title,'description':'Méthode, contrôles, résultats et éléments du TM à actualiser.','generatedAt':DATE,'sources':[source],'charts':charts,'blocks':blocs},
              'snapshot':{'version':1,'status':'ready','generatedAt':DATE,'datasets':datasets}}
    artifact['manifest']['sources'].extend(sources_sql)
    (SORTIE/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    notes={'audience':'technical','mode':'html','sections': [s[0] for s in sections],
           'structure':'Résumé, définitions, protocole, résultats, contrôles, limites, éléments obsolètes et suite. Les définitions sont avancées pour éviter des dénominateurs ambigus.',
           'chart_map':[{'id':'sensibilite','famille':'bar','question':'Quels chiffres sont le moins bien reconnus dans MNIST test ?','source':'metriques_par_classe.csv','palette':'Palette native commune, distinction par position et labels.'},
                        {'id':'validation','famille':'line','question':'À quelle époque la perte de validation est-elle minimale ?','points':11,'source':'entrainement.json','palette':'Une série native, sans légende redondante.'}],
           'omissions':'Comparaison des deux versions ×10 sous forme de table exacte. Les illustrations exportées du projet gardent leurs protocoles et styles historiques.'}
    (SORTIE/'notes_methode.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
    return markdown


def notebook():
    codes=[
        "from pathlib import Path\nimport sys, json, csv, hashlib\nimport numpy as np\nROOT = Path.cwd()\nassert (ROOT / 'scripts').is_dir(), 'Exécuter depuis la racine du dépôt.'\nsys.path.insert(0, str(ROOT))\nfrom scripts import _bootstrap\nfrom reconnaissance_chiffres.entrainement_cascade import repeter_cascade, empreintes_images\np = ROOT / 'donnees/preparees/cascade/cascade_top_n_v1'\nx = np.load(p / 'dataset_numpy/x_train_cascade.npy')\ny = np.load(p / 'dataset_numpy/y_train_cascade.npy')\nimages, labels = repeter_cascade(x, y, 10)\nassert images.shape == (12170,28,28,1) and labels.shape == (12170,)\nprint({'originaux': len(x), 'copies':len(images)-len(x), 'classes':np.bincount(y).tolist()})",
        "with np.load(ROOT / '.cache/keras/datasets/mnist.npz') as mnist:\n    test = mnist['x_test'].astype('float32')[...,None] / 255\n    train_mnist = mnist['x_train'].astype('float32')[...,None] / 255\na = np.load(ROOT / 'modeles/actifs/best_relu_10xcascade/indices_split.npz')\nassert not set(a['mnist_train']) & set(a['mnist_validation'])\nassert not empreintes_images(x) & empreintes_images(test)\nassert not empreintes_images(x) & empreintes_images(train_mnist[a['mnist_validation']])\nprint({'mnist_train':len(a['mnist_train']),'mnist_validation':len(a['mnist_validation']),'cascade_validation_test':0})",
        "from scripts.evaluation.bilan_remplacement_relu_10x import calculer\nbilan = calculer()\nprint(json.dumps({'mnist':[{k:r[k] for k in ['modele','corrects','erreurs','accuracy']} for r in bilan['mnist']], 'cascade':[{k:r[k] for k in ['modele','top_1_corrects','top_5_corrects','evaluation_independante']} for r in bilan['cascade']], 'preservation':bilan['preservation']},ensure_ascii=False,indent=2))"
    ]
    cells=[{'cell_type':'markdown','id':'resume','metadata':{},'source':['# Vérifications du remplacement ×10\n',
        '## tl;dr\n','Le nouveau meilleur checkpoint donne 9 910/10 000 sur MNIST. Les 12 170 entrées Cascade sont des copies exactes, réservées au train.\n',
        '## Context & Methods\n','Compagnon du compte rendu, exécuté cellule par cellule avec Python. Exécuter depuis la racine du dépôt avec l’environnement `.venv`. Les contrôles ne réentraînent aucun modèle.\n',
        '### Key Assumptions\n','Les annotations terrain historiques sont conservées. Les tests CTN ne sont pas indépendants. Le nouveau ×10 est entraîné avec une seule graine.\n',
        '## Data\n','Les CSV MNIST, CTN, terrain et le manifeste du dataset sont les sources des chiffres.']}]
    namespace={}
    for i,code in enumerate(codes,1):
        cells.append({'cell_type':'markdown','id':f'intro-{i}','metadata':{},
                      'source':[(['### Vérifier les copies exactes','### Vérifier les partitions','## Results'][i-1])+'\n']})
        output=io.StringIO()
        with redirect_stdout(output):exec(compile(code,f'cellule_{i}','exec'),namespace)
        cells.append({'cell_type':'code','id':f'code-{i}','metadata':{},'source':code.splitlines(keepends=True),'execution_count':i,
                      'outputs':[{'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(keepends=True)}]})
    cells.append({'cell_type':'markdown','id':'conclusions','metadata':{},'source':['## Takeaways\n',
        'Les copies sont exactes, les labels sont alignés et les partitions MNIST sont disjointes. Les résultats CTN utilisent les sources d’entraînement ; les photos non annotées n’ont pas de score d’exactitude.']})
    obj={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.9'}},'cells':cells}
    (SORTIE/'verifications.ipynb').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    data=calculer()
    produire(data)
    notebook()
    print('Bilan et notebook vérifiés :',SORTIE)


if __name__ == '__main__':
    main()
