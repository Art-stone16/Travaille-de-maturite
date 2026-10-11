"""Exporte une planche de dix exemples authentiques du jeu d'entrainement MNIST."""

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401


from pathlib import Path
import json

from reconnaissance_chiffres import config as env_config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    archive = Path.home() / ".keras" / "datasets" / "mnist.npz"
    with np.load(archive, allow_pickle=False) as data:
        images, labels = data["x_train"], data["y_train"]
    indices = [int(np.flatnonzero(labels == digit)[0]) for digit in range(10)]
    output = env_config.SORTIES_VISUALISATIONS / "exemples_mnist"
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(6, 2.9), dpi=300)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.98, bottom=0.08,
                        wspace=0.20, hspace=0.38)
    for digit, (ax, index) in enumerate(zip(axes.flat, indices)):
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(str(digit), fontsize=11, labelpad=4)
    destination = output / "planche_mnist_0_a_9.png"
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    (output / "provenance.json").write_text(json.dumps({
        "dataset": "MNIST", "partition": "entrainement",
        "selection": "Premiere image de chaque classe dans l'archive",
        "indices_base_zero": dict(zip(map(str, range(10)), indices)),
        "resolution_originale": [28, 28],
        "affichage": "Niveaux de gris, fond noir, agrandissement sans lissage"
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
