"""Visualise les motifs de classe appris par les sept modeles de chiffres.

Pour chaque chiffre, la carte represente le gradient moyen du score avant
softmax par rapport aux pixels d'entree. Les memes images MNIST sont utilisees
pour tous les modeles afin de rendre la comparaison equitable.
"""

from __future__ import annotations

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401


import argparse
import json
import os
from pathlib import Path

from reconnaissance_chiffres import config as env_config
from reconnaissance_chiffres.modeles import charger_modele as _charger_modele

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(env_config.PROJECT_ROOT / ".cache" / "matplotlib"))

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tensorflow as tf


MODEL_PATHS = (
    ("Best_COLOR_MAP", env_config.PROJECT_ROOT / "modeles/actifs/Best_COLOR_MAP/best_model.keras"),
    ("best_relu", env_config.PROJECT_ROOT / "modeles/actifs/best_relu/best_model.keras"),
    (
        "Best_relu_cascade",
        env_config.PROJECT_ROOT
        / "modeles/actifs/Best_relu_cascade/best_model.keras",
    ),
    (
        "Best_relu_cascade_V2",
        env_config.PROJECT_ROOT
        / "modeles/actifs/Best_relu_cascade_V2/best_model.keras",
    ),
    (
        "best_relu_2xcascade",
        env_config.PROJECT_ROOT
        / "modeles/actifs/best_relu_2xcascade/best_model.keras",
    ),
    (
        "best_relu_10xcascade",
        env_config.PROJECT_ROOT
        / "modeles/actifs/best_relu_10xcascade/best_model.keras",
    ),
    ("best_softmax", env_config.PROJECT_ROOT / "modeles/actifs/best_softmax/best_model.keras"),
)

PALETTE = LinearSegmentedColormap.from_list(
    "influence_classe",
    ("#174EA6", "#87A9E6", "#F7F7F7", "#F29B88", "#C62828"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Creer les planches des motifs de classe des sept modeles."
    )
    parser.add_argument(
        "--echantillons-par-classe",
        type=int,
        default=256,
        help="Nombre d'images MNIST utilisees pour chaque chiffre (defaut: 256).",
    )
    parser.add_argument(
        "--sortie",
        type=Path,
        default=env_config.SORTIES_VISUALISATIONS / "motifs_classes",
        help="Dossier de destination.",
    )
    return parser.parse_args()


def charger_mnist(nombre: int) -> tuple[list[np.ndarray], dict[str, list[int]], Path]:
    if nombre <= 0:
        raise ValueError("--echantillons-par-classe doit etre strictement positif")

    candidats = (
        env_config.PROJECT_ROOT / ".cache/keras/datasets/mnist.npz",
        Path.home() / ".keras/datasets/mnist.npz",
    )
    archive = next((chemin for chemin in candidats if chemin.is_file()), None)
    if archive is None:
        raise FileNotFoundError(
            "Archive MNIST introuvable. Placez mnist.npz dans .cache/keras/datasets/."
        )

    with np.load(archive, allow_pickle=False) as donnees:
        images = donnees["x_train"].astype("float32") / 255.0
        etiquettes = donnees["y_train"]

    lots: list[np.ndarray] = []
    indices: dict[str, list[int]] = {}
    for chiffre in range(10):
        selection = np.flatnonzero(etiquettes == chiffre)[:nombre]
        if selection.size < nombre:
            raise ValueError(f"Seulement {selection.size} images disponibles pour le chiffre {chiffre}")
        lots.append(images[selection, ..., np.newaxis])
        indices[str(chiffre)] = selection.astype(int).tolist()
    return lots, indices, archive


def couche_sortie(model: keras.Model) -> keras.layers.Dense:
    for couche in reversed(model.layers):
        if isinstance(couche, keras.layers.Dense) and couche.units == 10:
            return couche
    raise ValueError(f"Aucune couche Dense de 10 sorties dans {model.name}")


def charger_modele(chemin: Path) -> keras.Model:
    """Charge aussi les archives creees par une version Keras plus recente."""
    return _charger_modele(chemin, compile=False)


def calculer_cartes(model: keras.Model, lots: list[np.ndarray]) -> np.ndarray:
    sortie = couche_sortie(model)
    extracteur = keras.Model(inputs=model.inputs, outputs=sortie.input)
    poids, biais = sortie.get_weights()
    cartes = []

    for chiffre, images in enumerate(lots):
        gradients_lots = []
        for debut in range(0, len(images), 128):
            entree = tf.convert_to_tensor(images[debut : debut + 128])
            with tf.GradientTape() as ruban:
                ruban.watch(entree)
                caracteristiques = extracteur(entree, training=False)
                score = tf.linalg.matvec(caracteristiques, poids[:, chiffre]) + biais[chiffre]
                somme_scores = tf.reduce_sum(score)
            gradients = ruban.gradient(somme_scores, entree)
            gradients_lots.append(gradients.numpy())
        carte = np.concatenate(gradients_lots, axis=0).mean(axis=0)[..., 0]
        cartes.append(carte)

    return np.stack(cartes).astype("float32")


def normaliser_par_modele(cartes: np.ndarray) -> tuple[np.ndarray, float]:
    limite = float(np.percentile(np.abs(cartes), 99.0))
    if not np.isfinite(limite) or limite <= 0:
        limite = float(np.max(np.abs(cartes))) or 1.0
    return np.clip(cartes / limite, -1.0, 1.0), limite


def style_case(ax: plt.Axes, carte: np.ndarray, titre: str | None = None) -> None:
    ax.imshow(carte, cmap=PALETTE, vmin=-1, vmax=1, interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])
    for bordure in ax.spines.values():
        bordure.set_color("#9CA3AF")
        bordure.set_linewidth(0.65)
    if titre is not None:
        ax.set_title(titre, fontsize=10, color="#252A34", pad=5)


def ajouter_legende(fig: plt.Figure, position: list[float]) -> None:
    axe = fig.add_axes(position)
    gradient = np.linspace(-1, 1, 512)[np.newaxis, :]
    axe.imshow(gradient, aspect="auto", cmap=PALETTE, vmin=-1, vmax=1)
    axe.set_yticks([])
    axe.set_xticks([0, 256, 511], ["Défavorise", "Neutre", "Favorise"], fontsize=8)
    axe.tick_params(axis="x", length=0, pad=4, colors="#4B5563")
    for bordure in axe.spines.values():
        bordure.set_color("#9CA3AF")
        bordure.set_linewidth(0.6)


def sauver_planche_individuelle(
    nom: str, cartes: np.ndarray, destination: Path, nombre: int
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(8.27, 5.15), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.80, bottom=0.22, wspace=0.28, hspace=0.38)
    fig.suptitle(f"Motifs de classe — {nom}", fontsize=17, fontweight="bold", color="#1F2937", y=0.955)
    fig.text(
        0.5,
        0.885,
        f"Influence moyenne des pixels sur le score de chaque chiffre · {nombre} images MNIST par classe",
        ha="center",
        fontsize=9.2,
        color="#4B5563",
    )
    for chiffre, ax in enumerate(axes.flat):
        style_case(ax, cartes[chiffre], f"Chiffre {chiffre}")
    ajouter_legende(fig, [0.31, 0.115, 0.38, 0.026])
    fig.text(
        0.5,
        0.055,
        "Couleurs normalisées par modèle : le signe est comparable, pas l’amplitude brute entre modèles.",
        ha="center",
        fontsize=7.7,
        color="#6B7280",
    )
    fig.savefig(destination, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def sauver_comparaison(
    noms: list[str], cartes_modeles: list[np.ndarray], destination: Path, nombre: int
) -> None:
    fig, axes = plt.subplots(7, 10, figsize=(14.8, 10.2), dpi=240)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.88, bottom=0.12, wspace=0.12, hspace=0.28)
    fig.suptitle("Comparaison des motifs de classe des sept modèles", fontsize=20, fontweight="bold", color="#1F2937", y=0.965)
    fig.text(
        0.58,
        0.925,
        f"Même sélection de {nombre} images MNIST par chiffre · normalisation indépendante pour chaque modèle",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )

    for ligne, (nom, cartes) in enumerate(zip(noms, cartes_modeles)):
        for chiffre, ax in enumerate(axes[ligne]):
            style_case(ax, cartes[chiffre], str(chiffre) if ligne == 0 else None)
            if chiffre == 0:
                ax.set_ylabel(nom, rotation=0, ha="right", va="center", labelpad=13, fontsize=8.2, color="#252A34")

    ajouter_legende(fig, [0.42, 0.055, 0.32, 0.018])
    fig.text(
        0.58,
        0.018,
        "Rouge : augmenter la luminosité du pixel favorise la classe · Bleu : elle la défavorise",
        ha="center",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.savefig(destination, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def sauver_legende(destination: Path) -> None:
    fig = plt.figure(figsize=(7.2, 1.15), dpi=300, facecolor="white")
    fig.text(0.5, 0.83, "Influence d'un pixel sur le score de la classe", ha="center", fontsize=11, fontweight="bold", color="#1F2937")
    ajouter_legende(fig, [0.12, 0.37, 0.76, 0.18])
    fig.savefig(destination, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    options = arguments()
    options.sortie.mkdir(parents=True, exist_ok=True)
    (env_config.PROJECT_ROOT / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)

    lots, indices, archive = charger_mnist(options.echantillons_par_classe)
    noms: list[str] = []
    cartes_normalisees: list[np.ndarray] = []
    limites: dict[str, float] = {}

    for numero, (nom, chemin) in enumerate(MODEL_PATHS, start=1):
        if not chemin.is_file():
            raise FileNotFoundError(chemin)
        print(f"[{numero}/7] {nom}", flush=True)
        model = charger_modele(chemin)
        cartes_brutes = calculer_cartes(model, lots)
        cartes, limite = normaliser_par_modele(cartes_brutes)
        noms.append(nom)
        cartes_normalisees.append(cartes)
        limites[nom] = limite
        np.savez_compressed(
            options.sortie / f"donnees_{nom}.npz",
            gradients_bruts=cartes_brutes,
            gradients_normalises=cartes,
        )
        sauver_planche_individuelle(
            nom,
            cartes,
            options.sortie / f"planche_{numero:02d}_{nom}.png",
            options.echantillons_par_classe,
        )
        keras.backend.clear_session()

    sauver_comparaison(
        noms,
        cartes_normalisees,
        options.sortie / "comparaison_7_modeles.png",
        options.echantillons_par_classe,
    )
    sauver_legende(options.sortie / "legende_couleurs.png")

    provenance = {
        "methode": "Gradient moyen du score avant softmax par rapport aux pixels d'entree",
        "interpretation": {
            "rouge": "Augmenter la luminosite du pixel favorise le score de la classe",
            "bleu": "Augmenter la luminosite du pixel defavorise le score de la classe",
            "blanc": "Influence locale faible ou nulle",
        },
        "normalisation": "Par modele, percentile 99 de la valeur absolue des 10 cartes, puis bornage entre -1 et 1",
        "echantillons_par_classe": options.echantillons_par_classe,
        "archive_mnist": str(archive.relative_to(env_config.PROJECT_ROOT) if archive.is_relative_to(env_config.PROJECT_ROOT) else archive),
        "indices_mnist": indices,
        "modeles": {nom: str(chemin.relative_to(env_config.PROJECT_ROOT)) for nom, chemin in MODEL_PATHS},
        "limites_brutes_percentile_99": limites,
    }
    (options.sortie / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Images creees dans {options.sortie}")


if __name__ == "__main__":
    main()
