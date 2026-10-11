"""Assemble les resultats des huit modeles annotes de chaque feuille CTN en une image."""

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

from reconnaissance_chiffres import config as env_config

ROOT = env_config.SORTIES_CASCADE_TOP_N
MODELS = (
    'Best_COLOR_MAP',
    'best_relu',
    'Best_relu_cascade',
    'Best_relu_cascade_V2',
    'best_relu_2xcascade',
    'best_relu_10xcascade',
    'Best_COLOR_MAP_cascade',
    'best_softmax',
)
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'
FONT = ImageFont.truetype(FONT_PATH, 34)
TITLE_FONT = ImageFont.truetype(FONT_PATH, 46)
PANEL_WIDTH = 1500
GAP = 28
HEADER = 110
LABEL_HEIGHT = 70
BG = '#eef2f7'


def run():
    for folder in sorted(ROOT.glob('CTN_*')):
        if not folder.is_dir():
            continue
        sources = {}
        for resume in folder.glob('*/resume.txt'):
            model = next((line.split(':', 1)[1].strip() for line in resume.read_text(encoding='utf-8').splitlines() if line.startswith('Modele ')), None)
            image_path = resume.parent / 'resultat_annote.jpg'
            if model in MODELS and image_path.is_file():
                old = sources.get(model)
                if old is None or resume.parent.name > old.parent.name:
                    sources[model] = image_path
        missing = set(MODELS) - set(sources)
        if missing:
            raise RuntimeError(f'{folder.name}: resultats manquants: {sorted(missing)}')
        with Image.open(sources[MODELS[0]]) as reference:
            height = round(reference.height * PANEL_WIDTH / reference.width)
        panel_height = LABEL_HEIGHT + height
        width = PANEL_WIDTH * 2 + GAP * 3
        total_height = HEADER + panel_height * ((len(MODELS) + 1) // 2) + GAP * (((len(MODELS) + 1) // 2) + 1)
        canvas = Image.new('RGB', (width, total_height), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((GAP, 27), f'{folder.name} — comparaison des {len(MODELS)} modeles', font=TITLE_FONT, fill='#17253a')
        for index, model in enumerate(MODELS):
            col, row = index % 2, index // 2
            x = GAP + col * (PANEL_WIDTH + GAP)
            y = HEADER + GAP + row * (panel_height + GAP)
            draw.rounded_rectangle((x, y, x + PANEL_WIDTH, y + panel_height), radius=12, fill='white')
            draw.text((x + 20, y + 16), model, font=FONT, fill='#17253a')
            with Image.open(sources[model]) as source:
                resized = ImageOps.contain(source.convert('RGB'), (PANEL_WIDTH, height), Image.Resampling.LANCZOS)
                canvas.paste(resized, (x + (PANEL_WIDTH - resized.width) // 2, y + LABEL_HEIGHT))
        output = folder / f'comparaison_{len(MODELS)}_modeles.jpg'
        canvas.save(output, quality=90, optimize=True)
        print(f'{folder.name}: {output} ({width}x{total_height})')


if __name__ == '__main__':
    run()
