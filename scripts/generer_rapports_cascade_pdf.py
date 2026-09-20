"""Reconstruit les deux bilans PDF directement depuis les résultats sauvegardés.

Usage : .venv/bin/python scripts/generer_rapports_cascade_pdf.py
Aucun entraînement ni fichier HTML n'est nécessaire.
"""
from pathlib import Path
from collections import Counter
import csv
import json
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sorties/03_TESTS_PHOTOS/02_CASCADE_TOP_N"
OUT = ROOT / "output/pdf"
BLUE, LIGHT, GOLD, INK = "#3569a8", "#b9cce4", "#bc862d", "#223247"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "text.color": INK, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def load_runs(prefix):
    runs = []
    for path in sorted(SOURCE.glob(f"CTN_*/{prefix}*/predictions.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        summary = (path.parent / "resume.txt").read_text(encoding="utf-8")
        expected = int(re.search(r"Nombre attendu\s*:\s*(\d+)", summary)[1])
        detected = int(re.search(r"Nombre detecte\s*:\s*(\d+)", summary)[1])
        assert rows and len(rows) == detected, path
        assert len({r['numero'] for r in rows}) == len(rows), path
        ranks = np.array([int(r['rang_chiffre_attendu']) for r in rows])
        assert ((ranks >= 1) & (ranks <= 10)).all(), path
        for r in rows:
            assert int(r[f"top_{r['rang_chiffre_attendu']}_chiffre"]) == int(r['chiffre_attendu'])
            assert (int(r['prediction_top_1']) == int(r['chiffre_attendu'])) == (int(r['rang_chiffre_attendu']) == 1)
        runs.append(dict(image=path.parent.parent.name, timestamp=path.parent.name,
                         model=rows[0]['modele'], expected=expected, n=len(rows),
                         ranks=ranks, rows=rows, path=str(path.relative_to(ROOT))))
    assert runs, prefix
    assert len({r['model'] for r in runs}) == 1
    return sorted(runs, key=lambda r: (float(r['image'][4:]), r['timestamp']))


def pct(v):
    return f"{v:.2f}".replace('.', ',')


def paragraph(fig, y, text, size=10, width=99):
    lines = []
    for part in text.split('\n'):
        lines.extend(textwrap.wrap(part, width=width) or [''])
    fig.text(.085, y, '\n'.join(lines), fontsize=size, va='top', linespacing=1.5)


def page(title, subtitle, number):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    fig.text(.085, .954, 'CASCADE TOP N  /  RAPPORT DE TEST', color=BLUE, size=10, weight='bold')
    fig.text(.085, .908, title, size=21, weight='bold', va='top')
    fig.text(.085, .864, subtitle, size=10, color='#647286')
    fig.add_artist(plt.Line2D([.085, .915], [.842, .842], color=LIGHT, lw=1))
    fig.text(.085, .038, 'Travail de maturité · Résultats descriptifs sur les zones détectées', size=8, color='#647286')
    fig.text(.915, .038, f'{number} / 5', ha='right', size=8, color='#647286')
    return fig


def save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def generate(prefix, date_label, filename):
    runs = load_runs(prefix)
    ranks = np.concatenate([r['ranks'] for r in runs])
    n, expected = len(ranks), sum(r['expected'] for r in runs)
    counts = [int((ranks <= k).sum()) for k in range(1, 11)]
    cover = np.array(counts) / n * 100
    labels = [r['image'].replace('_', ' ') for r in runs]
    duplicates = Counter(r['image'] for r in runs)
    for i, r in enumerate(runs):
        if duplicates[r['image']] > 1:
            labels[i] += ' · ' + r['timestamp'][11:].replace('-', ':')
    lowest = min(runs, key=lambda r: (r['ranks'] <= 1).mean())
    latest = {r['image']: r for r in sorted(runs, key=lambda r: r['timestamp'])}
    dedup = np.concatenate([r['ranks'] for r in latest.values()])
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / filename
    with PdfPages(output, metadata={'Title': f'Cascade Top N - {date_label}', 'Author': 'Travail de maturité',
                                   'Subject': 'Couverture, classement, confusions et détection'}) as pdf:
        fig = page('Bilan global des cascades Top N', date_label, 1)
        fig.text(.085, .804, 'Synthèse des résultats', size=14, weight='bold')
        paragraph(fig, .773, f"Le chiffre attendu est classé premier dans {pct(cover[0])} % des zones détectées "
                  f"({counts[0]}/{n}). En élargissant aux cinq premières propositions, la couverture atteint "
                  f"{pct(cover[4])} % ({counts[4]}/{n}), soit {counts[4]-counts[0]} zones supplémentaires.")
        paragraph(fig, .678, f"Périmètre : {len(runs)} exécutions, {len(latest)} feuilles distinctes, {n} zones détectées "
                  f"pour {expected} chiffres attendus cumulés. Modèle : {runs[0]['model']}.")
        ax = fig.add_axes([.12, .335, .78, .265])
        k = [1, 2, 3, 5]
        bars = ax.bar([f'Top-{i}' for i in k], [cover[i-1] for i in k], color=BLUE, width=.58)
        ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel('Couverture (%)')
        ax.set_title('Couverture cumulée du chiffre attendu', loc='left', pad=18, size=12)
        for bar, i in zip(bars, k):
            ax.text(bar.get_x()+bar.get_width()/2, cover[i-1]+2, f'{pct(cover[i-1])} %\n{counts[i-1]}/{n}', ha='center', size=9)
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.18)
        paragraph(fig, .272, 'Lecture : Top-N signifie que le chiffre attendu figure parmi les N classes les mieux '
                  'classées pour une zone. Le dénominateur est le nombre de zones détectées, et non le nombre de chiffres attendus.')
        paragraph(fig, .173, 'Une couverture Top-5 élevée indique que la bonne classe reste souvent disponible parmi '
                  'les propositions. Elle ne signifie pas que le système choisit automatiquement la bonne réponse. '
                  'Les zones de bruit éventuelles sont incluses dans ces scores.', size=9, width=109)
        save(pdf, fig)

        fig = page('Les écarts entre feuilles', date_label, 2)
        paragraph(fig, .801, f"La feuille la moins bien classée en Top-1 est {lowest['image']} : "
                  f"{pct(100*(lowest['ranks'] <= 1).mean())} %. Le graphique sépare les réponses correctes dès "
                  'le premier choix des gains apportés par les rangs suivants.')
        ax = fig.add_axes([.255, .275, .64, .435])
        y = np.arange(len(runs)); left = np.zeros(len(runs))
        for lo, hi, color, hatch, label in [(0,1,BLUE,None,'Rang 1'), (1,3,LIGHT,'//','Rangs 2-3'),
                                           (3,5,'#e4c792','..','Rangs 4-5')]:
            values = np.array([100*((r['ranks']>lo)&(r['ranks']<=hi)).mean() for r in runs])
            ax.barh(y, values, left=left, color=color, edgecolor='white', hatch=hatch, height=.65, label=label)
            left += values
        ax.set_yticks(y, labels, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0,100)
        ax.set_xlabel('Part des zones détectées (%)'); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        ax.set_title('Couverture par feuille et par rang', loc='left', pad=33, size=12)
        ax.legend(loc='lower left', bbox_to_anchor=(0,1.005), ncol=3, fontsize=8, frameon=False)
        paragraph(fig, .202, 'La partie non couverte jusqu’à 100 % correspond aux rangs 6 à 10. Les variantes '
                  'CTN 2.2 et CTN 6.2 restent séparées : elles représentent des feuilles différentes. '
                  'Chaque barre utilise son propre nombre de zones comme dénominateur.', size=9, width=109)
        paragraph(fig, .119, 'Priorité : examiner les découpes des feuilles les moins performantes pour déterminer '
                  'si les erreurs viennent du classement, de la forme des chiffres ou du prétraitement.', size=9, width=109)
        save(pdf, fig)

        matrix = np.zeros((10,10), dtype=int)
        for r in runs:
            for row in r['rows']:
                matrix[int(row['chiffre_attendu']), int(row['prediction_top_1'])] += 1
        errors = [(int(matrix[a,b]),a,b) for a in range(10) for b in range(10) if a != b and matrix[a,b]]
        errors.sort(reverse=True)
        fig = page('Comprendre les confusions', date_label, 3)
        e, a, b = errors[0]
        paragraph(fig, .801, f"La confusion la plus fréquente est {a} vers {b}, avec {e} zones. Au total, "
                  f"{n-counts[0]} zones n’ont pas le chiffre attendu en premier choix. Les valeurs ci-dessous "
                  'sont des effectifs ; la diagonale correspond aux réponses conformes à la classe de la feuille.')
        ax = fig.add_axes([.16,.305,.68,.395])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
        for a in range(10):
            for b in range(10):
                ax.text(b,a,str(matrix[a,b]),ha='center',va='center',size=8,
                        color='white' if matrix[a,b] > matrix.max()*.55 else INK)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xlabel('Classe prédite en Top-1'); ax.set_ylabel('Classe attendue de la feuille')
        ax.set_title('Matrice de confusion - effectifs par zone', pad=14, size=12)
        paragraph(fig,.244, 'Confusions principales : ' + ' ; '.join(f'{a} vers {b} : {e}' for e,a,b in errors[:4]) + '.')
        paragraph(fig,.174, 'Les feuilles de même classe sont regroupées ici, variantes et réexécutions comprises. '
                  'Les classes 2 et 6 disposent donc de davantage de zones ; en juillet, la classe 3 est aussi '
                  'surreprésentée. Ces effectifs ne constituent pas une comparaison équilibrée entre classes.', size=9,width=109)
        paragraph(fig,.095, 'La classe attendue est attribuée à toutes les zones de la feuille. Une annotation manuelle '
                  'est nécessaire pour distinguer les vrais chiffres des faux positifs de détection.', size=9,width=109)
        save(pdf, fig)

        fig = page('Séparer détection et classement', date_label, 4)
        delta = [r['n']-r['expected'] for r in runs]
        paragraph(fig,.801, f"Le bilan de comptage est de {n} zones pour {expected} chiffres attendus "
                  f"(écart net : {n-expected:+d}). La somme des écarts absolus par exécution est de "
                  f"{sum(abs(d) for d in delta)}. Le total net masque donc des excès et des déficits selon les feuilles.")
        ax=fig.add_axes([.255,.33,.62,.39])
        ax.barh(np.arange(len(runs)),delta,color=[BLUE if d>=0 else GOLD for d in delta],height=.65)
        ax.set_yticks(np.arange(len(runs)),labels,fontsize=8); ax.invert_yaxis()
        ax.axvline(0,color=INK,lw=.8); ax.set_xlim(-15,37); ax.set_xlabel('Zones détectées - chiffres attendus')
        ax.set_title('Écart de comptage par feuille',loc='left',pad=16,size=12)
        for i,d in enumerate(delta): ax.text(d+(1 if d>=0 else -1),i,f'{d:+d}',va='center',ha='left' if d>=0 else 'right',size=9)
        paragraph(fig,.265, 'Un excès peut provenir de fragments ou de bruit ; un déficit peut provenir de chiffres '
                  'manqués ou fusionnés. Un écart nul ne garantit pas une détection parfaite : plusieurs types '
                  'd’erreurs peuvent se compenser.')
        paragraph(fig,.157, 'Il serait trompeur de diviser les succès Top-1 par le nombre attendu pour annoncer une '
                  'exactitude de bout en bout : certaines feuilles produisent plus de zones que de chiffres. '
                  'La priorité est de vérifier les découpes de CTN_2 (+30 zones) et de CTN_6 (-9 zones), '
                  'puis d’établir une correspondance entre chaque zone et chaque chiffre réel.',size=9,width=109)
        save(pdf,fig)

        fig = page('Détail et conditions de lecture',date_label,5)
        paragraph(fig,.802,'Chaque taux ci-dessous est calculé sur les zones détectées de l’exécution. '
                  'Les colonnes Att. et Dét. donnent les effectifs attendus et détectés.',size=9,width=109)
        table_data=[]
        for label,r in zip(labels,runs):
            table_data.append([label,str(r['expected']),str(r['n'])]+[pct(100*(r['ranks']<=k).mean()) for k in (1,2,3,5)])
        ax=fig.add_axes([.085,.438,.83,.31]); ax.axis('off')
        table=ax.table(cellText=table_data,colLabels=['Feuille / heure','Att.','Dét.','Top-1 %','Top-2 %','Top-3 %','Top-5 %'],
                       colWidths=[.28,.08,.08,.14,.14,.14,.14],cellLoc='center',bbox=[0,0,1,1])
        table.auto_set_font_size(False); table.set_fontsize(8)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor(BLUE if row==0 else ('#edf2f8' if row%2 else '#f7f9fc'))
            if row==0: cell.set_text_props(color='white',weight='bold')
        fig.text(.085,.404,'Méthode et robustesse',size=12,weight='bold')
        text = 'Les scores globaux sont pondérés par le nombre de zones : somme des succès divisée par somme des zones. '
        if len(runs)>len(latest):
            text += (f"Juillet inclut deux exécutions de CTN_3. En ne conservant que la dernière exécution par feuille, "
                     f"le Top-1 vaut {pct(100*(dedup<=1).mean())} % et le Top-5 {pct(100*(dedup<=5).mean())} % "
                     f"sur {len(dedup)} zones, contre {pct(cover[0])} % et {pct(cover[4])} % pour toutes les exécutions.")
        else:
            text += 'Cette campagne comporte une exécution par feuille. Les variantes 2.2 et 6.2 sont conservées. '
            text += 'Pour comparer à juillet, il faut retenir une seule exécution de CTN_3 dans la campagne initiale.'
        paragraph(fig,.379,text,size=9,width=109)
        paragraph(fig,.267,'Ces mesures décrivent les fichiers disponibles, sans démontrer une généralisation à de '
                  'nouvelles écritures. Les zones d’une même feuille ne sont pas des essais indépendants. '
                  'Aucun intervalle de confiance fondé sur cette indépendance n’est donc présenté.',size=9,width=109)
        paragraph(fig,.185,'Suite proposée : annoter les zones ambiguës, vérifier que les feuilles de test sont absentes '
                  'de l’entraînement, puis mesurer séparément la détection et la classification sur de nouvelles '
                  'feuilles. La question restante est la performance sur des écritures jamais vues.',size=9,width=109)
        paragraph(fig,.099,'Sources : predictions.csv et resume.txt de chaque exécution de la campagne, sous '
                  'sorties/03_TESTS_PHOTOS/02_CASCADE_TOP_N/CTN_*. Calculs refaits à partir des prédictions individuelles.',size=8,width=121)
        save(pdf,fig)

    return dict(pdf=str(output.relative_to(ROOT)), period=date_label, model=runs[0]['model'],
                runs=len(runs), distinct_images=len(latest), detected=n, expected=expected,
                top_counts=counts, top_percent=cover.tolist(),
                latest_only=dict(detected=len(dedup),top1=float(100*(dedup<=1).mean()),top5=float(100*(dedup<=5).mean())),
                sources=[r['path'] for r in runs])


def main():
    results=[generate('2026-07','Campagne des 24 et 25 juillet 2026','bilan_cascade_top_n_2026-07-24_25.pdf'),
             generate('2026-09-05','Campagne du 5 septembre 2026','bilan_cascade_top_n_2026-09-05.pdf')]
    (OUT/'sources_cascade_top_n.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in results:
        print(f"{r['pdf']} : {r['runs']} exécutions, {r['detected']} zones, Top-1 {r['top_percent'][0]:.2f} %")


if __name__ == '__main__':
    main()
