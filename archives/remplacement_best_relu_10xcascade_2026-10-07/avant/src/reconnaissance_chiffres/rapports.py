"""Génère les résumés Markdown depuis les résultats CTN sauvegardés.

Usage : python scripts/visualisation/generer_rapports_cascade_md.py
"""
from pathlib import Path
import os

from . import config
from collections import Counter, defaultdict
import csv
import re

ROOT = config.PROJECT_ROOT
SOURCE = config.SORTIES_CASCADE_TOP_N


def pourcentage(n, total):
    return f'{100 * n / total:.2f} %'.replace('.', ',') if total else '—'


def tableau(entetes, lignes):
    return '\n'.join(['| ' + ' | '.join(map(str, entetes)) + ' |',
                      '| ' + ' | '.join(['---'] * len(entetes)) + ' |'] +
                     ['| ' + ' | '.join(map(str, ligne)) + ' |' for ligne in lignes])


def charger(source):
    derniers = {}
    for path in sorted(source.glob('CTN_*/*/predictions.csv')):
        with path.open(encoding='utf-8-sig', newline='') as fichier:
            lignes = list(csv.DictReader(fichier, delimiter=';'))
        resume = (path.parent / 'resume.txt').read_text(encoding='utf-8')
        modele = re.search(r'^Modele\s*:\s*(.+)$', resume, re.M).group(1).strip()
        attendu = int(re.search(r'Nombre attendu\s*:\s*(\d+)', resume).group(1))
        detecte = int(re.search(r'Nombre detecte\s*:\s*(\d+)', resume).group(1))
        if len(lignes) != detecte:
            raise ValueError(f'Effectif incohérent : {path}')
        numeros = set()
        for ligne in lignes:
            rang = int(ligne['rang_chiffre_attendu'])
            if (ligne['modele'] != modele or not 1 <= rang <= 10
                    or int(ligne[f'top_{rang}_chiffre']) != int(ligne['chiffre_attendu'])
                    or (int(ligne['prediction_top_1']) == int(ligne['chiffre_attendu'])) != (rang == 1)
                    or ligne['numero'] in numeros):
                raise ValueError(f'Prédiction incohérente : {path}')
            numeros.add(ligne['numero'])
        cle = (modele, path.parent.parent.name)
        execution = dict(path=path, feuille=cle[1], date=path.parent.name,
                         attendu=attendu, lignes=lignes)
        if cle not in derniers or execution['date'] > derniers[cle]['date']:
            derniers[cle] = execution
    groupes = defaultdict(list)
    for (modele, _), execution in sorted(derniers.items()):
        groupes[modele].append(execution)
    if not groupes:
        raise ValueError(f'Aucun résultat dans {source}')
    return groupes


def score(lignes, n):
    return sum(int(ligne['rang_chiffre_attendu']) <= n for ligne in lignes)


METHODE = ('Les scores portent sur les zones détectées : Top-1 est la proportion dont le chiffre attendu '
           'est le premier choix ; Top-N indique sa présence parmi les N premières propositions. '
           'Toutes les zones reçoivent le chiffre attendu de leur feuille, y compris les éventuels fragments ou bruits. '
           'Ces scores ne mesurent donc pas une précision de détection annotée manuellement. '
           'Seule la dernière exécution de chaque couple modèle/feuille est retenue. '
           'Les variantes CTN_2.2 et CTN_6.2 sont conservées et regroupées avec leur chiffre pour les scores par chiffre. '
           'La précision globale est pondérée par le nombre de zones.')


def generer(source=SOURCE, sortie=None):
    source = Path(source)
    groupes = charger(source)
    sortie = Path(sortie) if sortie is not None else (
        config.RAPPORTS_CASCADE if source.resolve() == SOURCE.resolve() else source / "resume_global"
    )
    textes = {}
    donnees = {}
    for modele, executions in groupes.items():
        if Path(modele).name != modele or modele in ('.', '..'):
            raise ValueError(f'Nom de modèle invalide : {modele}')
        lignes = [ligne for execution in executions for ligne in execution['lignes']]
        par_chiffre = {chiffre: [l for l in lignes if int(l['chiffre_attendu']) == chiffre] for chiffre in range(10)}
        total = len(lignes)
        attendus = sum(e['attendu'] for e in executions)
        donnees[modele] = (lignes, par_chiffre, executions)
        niveaux = [1, 2, 3, 5, 10]
        parties = [f'# Résultats — {modele}',
                   f'{len(executions)} feuilles · Tests du {min(e["date"][:10] for e in executions)} au '
                   f'{max(e["date"][:10] for e in executions)} · {attendus} chiffres attendus · '
                   f'{total} zones détectées · Écart net : {total-attendus:+d}.',
                   '## Résultats globaux',
                   tableau(['Indicateur', 'Corrects / zones', 'Taux'],
                           [[f'Top-{n}', f'{score(lignes,n)}/{total}', pourcentage(score(lignes,n),total)] for n in niveaux]),
                   f'Le Top-5 ajoute {score(lignes,5)-score(lignes,1)} zones au Top-1 '
                   f'(+{100*(score(lignes,5)-score(lignes,1))/total:.2f} points). '
                   f'{total-score(lignes,5)} zones restent hors du Top-5.' if total else 'Aucune zone détectée.',
                   '## Résultats par chiffre',
                   tableau(['Chiffre', 'Zones', 'Corrects Top-1', 'Top-1', 'Top-3', 'Top-5'],
                           [[c, len(ls), score(ls,1), *[pourcentage(score(ls,n),len(ls)) for n in (1,3,5)]]
                            for c, ls in par_chiffre.items()])]
        disponibles = [c for c in par_chiffre if par_chiffre[c]]
        if disponibles:
            taux = {c: score(par_chiffre[c],1)/len(par_chiffre[c]) for c in disponibles}
            meilleurs = [str(c) for c in disponibles if taux[c] == max(taux.values())]
            faibles = [str(c) for c in disponibles if taux[c] == min(taux.values())]
            parties.append(f'Meilleur taux Top-1 : chiffre(s) {", ".join(meilleurs)} '
                           f'({pourcentage(score(par_chiffre[int(meilleurs[0])],1),len(par_chiffre[int(meilleurs[0])]))}). '
                           f'Plus faible : chiffre(s) {", ".join(faibles)} '
                           f'({pourcentage(score(par_chiffre[int(faibles[0])],1),len(par_chiffre[int(faibles[0])]))}).')
        erreurs = Counter((int(l['chiffre_attendu']),int(l['prediction_top_1'])) for l in lignes
                          if int(l['rang_chiffre_attendu']) != 1)
        parties += ['## Principales confusions Top-1',
                    tableau(['Attendu → prédit', 'Nombre de zones'],
                            [[f'{a} → {b}', n] for (a,b), n in erreurs.most_common(5)]) if erreurs else 'Aucune confusion.',
                    '## Feuilles et sources',
                    tableau(['Feuille / prédictions', 'Exécution', 'Attendus', 'Détectés', 'Écart', 'Top-1', 'Top-5'],
                            [[f'[{e["feuille"]}]({os.path.relpath(e["path"], sortie / modele)})', e['date'],
                              e['attendu'], len(e['lignes']), f'{len(e["lignes"])-e["attendu"]:+d}',
                              *[pourcentage(score(e['lignes'],n),len(e['lignes'])) for n in (1,5)]] for e in executions]),
                    '## Lecture des résultats', METHODE,
                    '[Comparaison des modèles](../comparaison_modeles.md)']
        textes[sortie / modele / 'resultats.md'] = '\n\n'.join(parties) + '\n'
    ordre = sorted(donnees, key=lambda m: score(donnees[m][0],1)/len(donnees[m][0]) if donnees[m][0] else -1, reverse=True)
    parties = ['# Comparaison des modèles — Cascade Top-N', '## Résultats globaux',
               tableau(['Modèle', 'Feuilles', 'Attendus', 'Zones', 'Top-1', 'Top-2', 'Top-3', 'Top-5'],
                       [[f'[{m}]({m}/resultats.md)', len(donnees[m][2]), sum(e['attendu'] for e in donnees[m][2]),
                         len(donnees[m][0]), *[pourcentage(score(donnees[m][0],n),len(donnees[m][0])) for n in (1,2,3,5)]] for m in ordre]),
               'Modèles classés par précision globale Top-1 décroissante.', '## Précision Top-1 par chiffre',
               tableau(['Modèle'] + list(range(10)),
                       [[m] + [pourcentage(score(donnees[m][1][c],1),len(donnees[m][1][c])) for c in range(10)] for m in ordre]),
               '## Méthode', METHODE,
               'Les campagnes ont été réalisées à des dates différentes ; la comparaison décrit les résultats sauvegardés. '
               'Un écart entre zones détectées et chiffres attendus est un bilan de comptage, pas un nombre exact de faux positifs ou de chiffres manqués.',
               'Régénération depuis la racine du projet : `python scripts/visualisation/generer_rapports_cascade_md.py`.']
    textes[sortie / 'comparaison_modeles.md'] = '\n\n'.join(parties) + '\n'
    for path, texte in textes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texte, encoding='utf-8')
    return sortie


if __name__ == '__main__':
    print(generer())
