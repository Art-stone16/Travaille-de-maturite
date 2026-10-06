"""Génère les rapports comparatifs Cascade Top-N au format Markdown."""

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401

from reconnaissance_chiffres.rapports import generer

if __name__ == '__main__':
    print(generer())
