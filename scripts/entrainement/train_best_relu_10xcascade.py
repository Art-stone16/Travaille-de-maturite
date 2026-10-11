"""Entraîne best_relu_10xcascade avec 10 exemplaires exacts par image Cascade.

La routine partagée réserve la validation MNIST avant l'injection de Cascade.
Voir le compte rendu du remplacement pour les incohérences du protocole historique.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401
from reconnaissance_chiffres.entrainement_cascade import entrainer

FACTEUR_REPETITION = 10
MODEL_NAME = "best_relu_10xcascade"

if __name__ == "__main__":
    entrainer(FACTEUR_REPETITION, MODEL_NAME)
