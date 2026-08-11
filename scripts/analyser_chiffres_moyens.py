"""Analyse les chiffres moyens de MNIST et les compare a une ecriture personnelle.

Ce script ne lance aucun entrainement. Il calcule, pour chaque classe, la moyenne
et l'ecart-type de chaque pixel puis range toutes les donnees, figures et metadonnees
dans un dossier d'experience autonome.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import env_config

import cv2
import matplotlib
import numpy as np

if "--afficher" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import detection_chiffres as detection


SORTIE_PAR_DEFAUT = env_config.SORTIES_CHIFFRES_MOYENS
COULEUR_MNIST = np.array([0.18, 0.43, 0.72], dtype=np.float32)
COULEUR_PERSONNELLE = np.array([0.93, 0.49, 0.16], dtype=np.float32)


def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule le chiffre moyen et l'ecart-type pixel par pixel de "
            "MNIST, puis compare facultativement des images personnelles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("entrainement", "test", "tout"),
        default="entrainement",
        help="Partie de MNIST utilisee pour les statistiques.",
    )
    parser.add_argument(
        "--mnist-npz",
        "--mnist-path",
        dest="mnist_npz",
        type=Path,
        help=(
            "Archive MNIST locale contenant x_train/y_train/x_test/y_test. "
            "Sans cette option, le cache local ~/.keras/datasets/mnist.npz "
            "est utilise s'il existe, puis Keras est employe en dernier recours."
        ),
    )
    parser.add_argument(
        "--personnel",
        action="append",
        default=[],
        metavar="CHIFFRE=IMAGE",
        help=(
            "Image d'un seul chiffre a comparer a sa classe. L'option peut "
            "etre repetee, par exemple "
            "--personnel 7=donnees/ecritures_personnelles/mon_7.JPG."
        ),
    )
    parser.add_argument(
        "--sortie",
        type=Path,
        default=SORTIE_PAR_DEFAUT,
        help="Dossier racine dans lequel creer l'experience.",
    )
    parser.add_argument(
        "--nom-experience",
        help=(
            "Nom du sous-dossier de sortie. Par defaut, un horodatage unique "
            "est utilise. Un nom existant n'est jamais ecrase."
        ),
    )
    parser.add_argument(
        "--limite-par-classe",
        type=int,
        help=(
            "Nombre maximal d'images MNIST par classe. Utile seulement pour "
            "un test rapide; omettre pour l'analyse complete."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Resolution des figures PNG.",
    )
    parser.add_argument(
        "--afficher",
        action="store_true",
        help="Ouvre les figures apres leur sauvegarde (si l'environnement le permet).",
    )
    return parser


def _verifier_images(images: np.ndarray, etiquettes: np.ndarray) -> None:
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(
            "Les images MNIST doivent avoir la forme (N, 28, 28); "
            f"forme recue: {images.shape}."
        )
    if etiquettes.ndim != 1 or len(etiquettes) != len(images):
        raise ValueError(
            "Les etiquettes MNIST doivent etre un vecteur de meme longueur "
            "que les images."
        )
    classes = set(np.unique(etiquettes).tolist())
    if not set(range(10)).issubset(classes):
        manquantes = sorted(set(range(10)) - classes)
        raise ValueError(f"Classes MNIST absentes: {manquantes}.")


def _normaliser_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    if not np.issubdtype(images.dtype, np.number):
        raise ValueError("Les pixels MNIST doivent etre numeriques.")
    images = images.astype(np.float32)
    if not np.isfinite(images).all():
        raise ValueError("Les pixels MNIST contiennent NaN ou une valeur infinie.")
    minimum = float(images.min())
    maximum = float(images.max())
    if minimum < 0:
        raise ValueError(f"Pixel MNIST negatif detecte: {minimum}.")
    if maximum > 1:
        if maximum <= 255:
            images /= 255.0
        else:
            raise ValueError(f"Pixel MNIST superieur a 255 detecte: {maximum}.")
    return np.clip(images, 0.0, 1.0)


def charger_mnist(
    source: str,
    chemin_npz: Path | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Charge MNIST depuis une archive locale ou via Keras."""
    if chemin_npz is None:
        cache_keras = Path.home() / ".keras" / "datasets" / "mnist.npz"
        if cache_keras.is_file():
            chemin_npz = cache_keras

    if chemin_npz is not None:
        chemin_npz = chemin_npz.expanduser().resolve()
        if not chemin_npz.is_file():
            raise FileNotFoundError(f"Archive MNIST introuvable: {chemin_npz}")
        with np.load(chemin_npz, allow_pickle=False) as archive:
            attendues = {"x_train", "y_train", "x_test", "y_test"}
            absentes = attendues - set(archive.files)
            if absentes:
                raise ValueError(
                    "Archive MNIST incomplete; cles absentes: "
                    + ", ".join(sorted(absentes))
                )
            x_train = archive["x_train"]
            y_train = archive["y_train"]
            x_test = archive["x_test"]
            y_test = archive["y_test"]
        origine = str(chemin_npz)
    else:
        try:
            import keras
        except ImportError as exc:
            raise RuntimeError(
                "Keras est necessaire pour charger MNIST automatiquement. "
                "Installez les dependances ou utilisez --mnist-npz."
            ) from exc
        try:
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        except Exception as exc:
            raise RuntimeError(
                "MNIST n'est pas disponible localement et son chargement a echoue. "
                "Placez mnist.npz dans ~/.keras/datasets/ ou indiquez son chemin "
                "avec --mnist-path."
            ) from exc
        origine = "keras.datasets.mnist"

    if source == "entrainement":
        images, etiquettes = x_train, y_train
    elif source == "test":
        images, etiquettes = x_test, y_test
    else:
        images = np.concatenate((x_train, x_test), axis=0)
        etiquettes = np.concatenate((y_train, y_test), axis=0)

    etiquettes = np.asarray(etiquettes, dtype=np.int64).reshape(-1)
    _verifier_images(np.asarray(images), etiquettes)
    return _normaliser_images(images), etiquettes, origine


def limiter_par_classe(
    images: np.ndarray,
    etiquettes: np.ndarray,
    limite: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if limite is None:
        return images, etiquettes
    if limite <= 0:
        raise ValueError("--limite-par-classe doit etre strictement positif.")
    indices = []
    for classe in range(10):
        indices.extend(np.flatnonzero(etiquettes == classe)[:limite].tolist())
    indices = np.asarray(indices, dtype=np.int64)
    return images[indices], etiquettes[indices]


def calculer_statistiques(
    images: np.ndarray,
    etiquettes: np.ndarray,
) -> dict[int, dict[str, np.ndarray | int]]:
    statistiques: dict[int, dict[str, np.ndarray | int]] = {}
    for classe in range(10):
        groupe = images[etiquettes == classe]
        if len(groupe) == 0:
            raise ValueError(f"Aucune image disponible pour la classe {classe}.")
        statistiques[classe] = {
            "nombre": int(len(groupe)),
            "moyenne": groupe.mean(axis=0, dtype=np.float64).astype(np.float32),
            "ecart_type": groupe.std(axis=0, dtype=np.float64).astype(np.float32),
        }
    return statistiques


def nom_sur(texte: str) -> str:
    """Produit un nom de dossier portable sans masquer le nom original du manifeste."""
    texte = re.sub(r"[^A-Za-z0-9._-]+", "_", texte.strip())
    return texte.strip("._-") or "element"


def creer_dossier_experience(base: Path, nom: str | None) -> Path:
    base = base.expanduser().resolve()
    nom_final = nom_sur(nom) if nom else datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier = base / nom_final
    try:
        dossier.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Le dossier d'experience existe deja: {dossier}. "
            "Choisissez un autre --nom-experience."
        ) from exc
    for sous_dossier in (
        "statistiques/matrices",
        "statistiques/images",
        "figures",
        "personnels",
        "rapports",
    ):
        (dossier / sous_dossier).mkdir(parents=True, exist_ok=True)
    return dossier


def sauvegarder_statistiques(
    dossier: Path,
    statistiques: dict[int, dict[str, np.ndarray | int]],
) -> list[dict[str, float | int | str]]:
    lignes = []
    for classe, valeurs in statistiques.items():
        moyenne = np.asarray(valeurs["moyenne"])
        ecart_type = np.asarray(valeurs["ecart_type"])
        np.save(
            dossier / "statistiques" / "matrices" / f"classe_{classe}_moyenne.npy",
            moyenne,
        )
        np.save(
            dossier
            / "statistiques"
            / "matrices"
            / f"classe_{classe}_ecart_type.npy",
            ecart_type,
        )
        cv2.imwrite(
            str(
                dossier
                / "statistiques"
                / "images"
                / f"classe_{classe}_moyenne.png"
            ),
            np.rint(moyenne * 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(
                dossier
                / "statistiques"
                / "images"
                / f"classe_{classe}_ecart_type_echelle_commune.png"
            ),
            np.rint(np.clip(ecart_type / 0.5, 0, 1) * 255).astype(np.uint8),
        )
        lignes.append(
            {
                "classe": classe,
                "nombre_images": int(valeurs["nombre"]),
                "intensite_moyenne": float(moyenne.mean()),
                "ecart_type_moyen": float(ecart_type.mean()),
                "ecart_type_maximum": float(ecart_type.max()),
            }
        )

    chemin_csv = dossier / "statistiques" / "resume_par_classe.csv"
    with chemin_csv.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        writer.writeheader()
        writer.writerows(lignes)
    return lignes


def creer_figure_statistiques(
    dossier: Path,
    statistiques: dict[int, dict[str, np.ndarray | int]],
    source: str,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(4, 5, figsize=(12, 9), constrained_layout=True)
    for classe in range(10):
        groupe = classe // 5
        colonne = classe % 5
        axe_moyenne = axes[groupe * 2, colonne]
        axe_ecart = axes[groupe * 2 + 1, colonne]
        valeurs = statistiques[classe]
        axe_moyenne.imshow(valeurs["moyenne"], cmap="gray", vmin=0, vmax=1)
        axe_ecart.imshow(
            valeurs["ecart_type"], cmap="magma", vmin=0, vmax=0.5
        )
        axe_moyenne.set_title(
            f"Classe {classe} — moyenne\n(n={valeurs['nombre']})",
            fontsize=10,
        )
        axe_ecart.set_title(f"Classe {classe} — ecart-type", fontsize=10)
        for axe in (axe_moyenne, axe_ecart):
            axe.set_xticks([])
            axe.set_yticks([])

    figure.suptitle(
        "MNIST : moyenne et ecart-type pixel par pixel\n"
        f"Source : {source} | valeurs normalisees entre 0 et 1",
        fontsize=14,
    )
    chemin = dossier / "figures" / "moyennes_et_ecarts_types_mnist.png"
    figure.savefig(chemin, dpi=dpi, facecolor="white")
    return chemin


def analyser_specifications_personnelles(
    specifications: Iterable[str],
) -> list[tuple[int, Path]]:
    resultats = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Specification personnelle invalide: {specification!r}. "
                "Format attendu: CHIFFRE=IMAGE."
            )
        etiquette_texte, chemin_texte = specification.split("=", 1)
        try:
            etiquette = int(etiquette_texte)
        except ValueError as exc:
            raise ValueError(
                f"La classe personnelle doit etre un entier entre 0 et 9: "
                f"{etiquette_texte!r}."
            ) from exc
        if etiquette not in range(10):
            raise ValueError(f"Classe personnelle hors de 0..9: {etiquette}.")
        chemin = Path(chemin_texte).expanduser().resolve()
        if not chemin.is_file():
            raise FileNotFoundError(f"Image personnelle introuvable: {chemin}")
        resultats.append((etiquette, chemin))
    return resultats


def preparer_image_personnelle(chemin: Path) -> np.ndarray:
    image = detection.charger_image(chemin)
    hauteur, largeur = image.shape[:2]
    preparee = detection.preparer_chiffre_pour_modele(
        image,
        (0, 0, largeur, hauteur),
    )
    matrice = np.asarray(preparee[0, :, :, 0], dtype=np.float32)
    if matrice.shape != (28, 28) or not np.isfinite(matrice).all():
        raise ValueError(
            f"Le pretraitement de {chemin} n'a pas produit une matrice 28 x 28 valide."
        )
    if float(matrice.max()) < 0.05:
        raise ValueError(
            f"Aucun trait exploitable n'a ete trouve dans l'image personnelle: {chemin}"
        )
    return np.clip(matrice, 0, 1)


def creer_superposition(moyenne: np.ndarray, personnelle: np.ndarray) -> np.ndarray:
    fond = np.ones((28, 28, 3), dtype=np.float32)
    fond -= moyenne[..., None] * (1 - COULEUR_MNIST)
    fond -= personnelle[..., None] * (1 - COULEUR_PERSONNELLE)
    return np.clip(fond, 0, 1)


def comparer_images_personnelles(
    dossier: Path,
    specifications: list[tuple[int, Path]],
    statistiques: dict[int, dict[str, np.ndarray | int]],
    dpi: int,
) -> list[dict[str, float | int | str]]:
    lignes: list[dict[str, float | int | str]] = []
    occurrences: dict[tuple[int, str], int] = {}

    for etiquette, chemin in specifications:
        cle = (etiquette, chemin.stem)
        occurrences[cle] = occurrences.get(cle, 0) + 1
        identifiant = (
            f"classe_{etiquette}_{nom_sur(chemin.stem)}_"
            f"{occurrences[cle]:02d}"
        )
        dossier_element = dossier / "personnels" / identifiant
        dossier_element.mkdir(parents=True, exist_ok=False)
        personnelle = preparer_image_personnelle(chemin)
        moyenne = np.asarray(statistiques[etiquette]["moyenne"])
        difference = np.abs(personnelle - moyenne)
        mae = float(difference.mean())
        rmse = float(np.sqrt(np.mean((personnelle - moyenne) ** 2)))
        aplaties = np.stack((personnelle.ravel(), moyenne.ravel()))
        correlation = float(np.corrcoef(aplaties)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

        np.save(dossier_element / "matrice_28x28.npy", personnelle)
        np.savetxt(
            dossier_element / "matrice_28x28.csv",
            personnelle,
            delimiter=",",
            fmt="%.6f",
        )
        cv2.imwrite(
            str(dossier_element / "image_28x28.png"),
            np.rint(personnelle * 255).astype(np.uint8),
        )

        figure, axes = plt.subplots(1, 4, figsize=(12, 3.4), constrained_layout=True)
        axes[0].imshow(moyenne, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Moyenne MNIST {etiquette}")
        axes[1].imshow(personnelle, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ecriture pretraitee")
        axes[2].imshow(creer_superposition(moyenne, personnelle))
        axes[2].set_title("Superposition")
        axes[2].legend(
            handles=(
                Patch(facecolor=COULEUR_MNIST, label="Moyenne MNIST"),
                Patch(facecolor=COULEUR_PERSONNELLE, label="Ecriture"),
            ),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.28),
            fontsize=8,
            frameon=False,
        )
        image_difference = axes[3].imshow(
            difference, cmap="magma", vmin=0, vmax=1
        )
        axes[3].set_title(f"Difference absolue\nMAE={mae:.3f}")
        figure.colorbar(image_difference, ax=axes[3], fraction=0.046, pad=0.04)
        for axe in axes:
            axe.set_xticks([])
            axe.set_yticks([])
        figure.suptitle(
            f"Comparaison de {chemin.name} avec la classe {etiquette}\n"
            f"RMSE={rmse:.3f} | correlation={correlation:.3f}",
            fontsize=12,
        )
        chemin_figure = dossier_element / "comparaison.png"
        figure.savefig(chemin_figure, dpi=dpi, facecolor="white")

        lignes.append(
            {
                "identifiant": identifiant,
                "classe_attendue": etiquette,
                "image_source": str(chemin),
                "matrice_28x28": str(
                    (dossier_element / "matrice_28x28.npy").relative_to(dossier)
                ),
                "figure": str(chemin_figure.relative_to(dossier)),
                "mae_vers_moyenne": mae,
                "rmse_vers_moyenne": rmse,
                "correlation_avec_moyenne": correlation,
            }
        )
        plt.close(figure)

    if lignes:
        chemin_csv = dossier / "rapports" / "comparaisons_personnelles.csv"
        with chemin_csv.open("w", newline="", encoding="utf-8") as fichier:
            writer = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
            writer.writeheader()
            writer.writerows(lignes)
    return lignes


def sauvegarder_manifeste(
    dossier: Path,
    arguments: argparse.Namespace,
    origine: str,
    nombre_images: int,
    resume_classes: list[dict[str, float | int | str]],
    comparaisons: list[dict[str, float | int | str]],
) -> None:
    manifeste = {
        "type": "analyse_chiffres_moyens_mnist",
        "cree_le": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dossier_experience": str(dossier),
        "entrees": {
            "origine_mnist": origine,
            "partition": arguments.source,
            "nombre_images_utilisees": nombre_images,
            "limite_par_classe": arguments.limite_par_classe,
            "images_personnelles": [
                ligne["image_source"] for ligne in comparaisons
            ],
        },
        "sorties": {
            "figure_generale": "figures/moyennes_et_ecarts_types_mnist.png",
            "resume_classes": "statistiques/resume_par_classe.csv",
            "matrices": "statistiques/matrices/",
            "images_statistiques": "statistiques/images/",
            "comparaisons_personnelles": (
                "rapports/comparaisons_personnelles.csv" if comparaisons else None
            ),
        },
        "classes": resume_classes,
        "comparaisons_personnelles": comparaisons,
        "notes": [
            "Les intensites sont normalisees entre 0 (fond) et 1 (encre).",
            "La distance a la moyenne est descriptive et ne remplace pas "
            "l'evaluation du modele de classification.",
        ],
    }
    with (dossier / "manifest.json").open("w", encoding="utf-8") as fichier:
        json.dump(manifeste, fichier, ensure_ascii=False, indent=2)


def sauvegarder_guide_sortie(dossier: Path, comparaisons_presentes: bool) -> None:
    texte = """# Analyse des chiffres moyens

Ce dossier est une experience autonome. Toutes les intensites vont de 0 (fond)
a 1 (encre).

## Repertoires

- `manifest.json` : provenance, parametres et index des sorties.
- `figures/` : synthese visuelle des moyennes et ecarts-types MNIST.
- `statistiques/matrices/` : matrices NumPy 28 x 28 reutilisables.
- `statistiques/images/` : apercus PNG des matrices.
- `statistiques/resume_par_classe.csv` : effectifs et statistiques agregees.
- `personnels/` : matrice pretraitee et comparaison de chaque image personnelle.
- `rapports/` : tableau des distances pour les images personnelles.

L'ecart a un chiffre moyen sert a visualiser le decalage avec MNIST. Ce n'est
pas, a lui seul, une explication de la prediction d'un reseau convolutif.
"""
    if not comparaisons_presentes:
        texte += (
            "\nAucune image personnelle n'a ete fournie pour cette experience.\n"
        )
    (dossier / "README.md").write_text(texte, encoding="utf-8")


def main() -> int:
    parser = construire_parser()
    arguments = parser.parse_args()
    if arguments.dpi <= 0:
        parser.error("--dpi doit etre strictement positif.")

    dossier: Path | None = None
    try:
        specifications = analyser_specifications_personnelles(arguments.personnel)
        images, etiquettes, origine = charger_mnist(
            arguments.source,
            arguments.mnist_npz,
        )
        images, etiquettes = limiter_par_classe(
            images,
            etiquettes,
            arguments.limite_par_classe,
        )
        statistiques = calculer_statistiques(images, etiquettes)
        dossier = creer_dossier_experience(
            arguments.sortie,
            arguments.nom_experience,
        )
        resume_classes = sauvegarder_statistiques(dossier, statistiques)
        chemin_figure = creer_figure_statistiques(
            dossier,
            statistiques,
            arguments.source,
            arguments.dpi,
        )
        comparaisons = comparer_images_personnelles(
            dossier,
            specifications,
            statistiques,
            arguments.dpi,
        )
        sauvegarder_manifeste(
            dossier,
            arguments,
            origine,
            len(images),
            resume_classes,
            comparaisons,
        )
        sauvegarder_guide_sortie(dossier, bool(comparaisons))

        print(f"Analyse terminee: {dossier}")
        print(f"Figure generale: {chemin_figure}")
        print(f"Images MNIST analysees: {len(images)}")
        print(f"Images personnelles comparees: {len(comparaisons)}")
        print(f"Index complet: {dossier / 'manifest.json'}")

        if arguments.afficher:
            plt.show()
        plt.close("all")
        return 0
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        if dossier is not None:
            print(
                "Attention: un dossier de sortie partiel a pu etre cree ici: "
                f"{dossier}"
            )
        parser.exit(2, f"Erreur: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
