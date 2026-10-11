"""Accès commun aux tableaux du dataset Cascade préparé."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    import numpy as np


def chemin_dataset_cascade(nom: str = "cascade_top_n_v1") -> Path:
    return config.DONNEES_PREPAREES / "cascade" / nom / "dataset_numpy"


def charger_cascade_entrainement(dossier: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Charge les images et étiquettes, sans changer leur ordre ni leurs valeurs."""
    import numpy as np

    dossier = chemin_dataset_cascade() if dossier is None else Path(dossier)
    return (
        np.load(dossier / "x_train_cascade.npy", allow_pickle=False),
        np.load(dossier / "y_train_cascade.npy", allow_pickle=False),
    )
