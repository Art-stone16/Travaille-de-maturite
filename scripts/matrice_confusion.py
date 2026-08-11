#!/usr/bin/env python3
"""Évaluer un modèle MNIST et détailler ses performances pour chaque chiffre.

Le script crée un dossier d'évaluation autonome contenant la configuration,
les matrices de confusion, les métriques one-vs-rest par classe et les figures.
Il ne réentraîne jamais le modèle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import env_config


os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
import numpy as np
import pandas as pd

if "--afficher" not in sys.argv and "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix


CLASSES = tuple(range(10))
SORTIE_PAR_DEFAUT = env_config.SORTIES_PERFORMANCES_PAR_CHIFFRE
MODELE_PAR_DEFAUT = (
    env_config.MODELES_VALIDES
    / "Best_COLOR_MAP"
    / "best_model.keras"
)
Z_95 = 1.959963984540054
GRAINE_SOUS_ECHANTILLON = 20260810


class CheminsEvaluation:
    """Tous les chemins produits par une évaluation."""

    def __init__(self, racine: Path) -> None:
        self.racine = racine
        self.donnees = racine / "donnees"
        self.graphiques = racine / "graphiques"
        self.configuration = racine / "configuration.json"
        self.guide = racine / "LISEZ_MOI.txt"
        self.catalogue = racine / "catalogue_sorties.csv"
        self.matrice_effectifs_csv = self.donnees / "matrice_confusion_effectifs.csv"
        self.matrice_reel_csv = (
            self.donnees / "matrice_confusion_normalisee_par_classe_reelle.csv"
        )
        self.matrice_predit_csv = (
            self.donnees / "matrice_confusion_normalisee_par_classe_predite.csv"
        )
        self.metriques_csv = self.donnees / "metriques_par_classe.csv"
        self.resume_csv = self.donnees / "resume_global.csv"
        self.indices_csv = self.donnees / "indices_images_test.csv"
        self.predictions_csv = self.donnees / "predictions_test.csv"
        self.matrice_effectifs_png = (
            self.graphiques / "matrice_confusion_effectifs.png"
        )
        self.matrice_reel_png = (
            self.graphiques / "matrice_confusion_normalisee_reel.png"
        )
        self.metriques_png = self.graphiques / "sensibilite_specificite_par_classe.png"
        self.erreurs_png = self.graphiques / "taux_erreurs_par_classe.png"

    def creer(self) -> None:
        self.donnees.mkdir(parents=True, exist_ok=False)
        self.graphiques.mkdir(parents=True, exist_ok=False)


def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Évalue un modèle MNIST sans le réentraîner, puis exporte la matrice "
            "de confusion et les métriques détaillées pour chaque chiffre."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modele",
        "--model",
        type=Path,
        default=MODELE_PAR_DEFAUT,
        help="fichier .keras/.h5 ou dossier contenant best_model.keras",
    )
    parser.add_argument(
        "--nom-evaluation",
        help=(
            "nom du sous-dossier ; sans valeur, le nom du modèle et un "
            "horodatage sont utilisés"
        ),
    )
    parser.add_argument(
        "--sortie",
        type=Path,
        default=SORTIE_PAR_DEFAUT,
        help="dossier racine des évaluations",
    )
    parser.add_argument(
        "--mnist-path",
        "--mnist-npz",
        dest="mnist_path",
        type=Path,
        help=(
            "archive MNIST locale ; sinon utilise ~/.keras/datasets/mnist.npz "
            "avant de demander le chargement à Keras"
        ),
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        help="nombre maximal d'images de test, uniquement pour une vérification rapide",
    )
    parser.add_argument(
        "--sauver-predictions",
        action="store_true",
        help="exporte aussi la prédiction et les dix scores de chaque image",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="résolution des figures PNG",
    )
    parser.add_argument(
        "--afficher",
        "--show",
        dest="afficher",
        action="store_true",
        help="ouvre les figures après leur sauvegarde",
    )
    return parser


def valider_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.test_limit is not None and args.test_limit <= 0:
        parser.error("--test-limit doit être strictement positif")
    if args.dpi <= 0:
        parser.error("--dpi doit être strictement positif")


def maintenant_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugifier(texte: str) -> str:
    normalise = unicodedata.normalize("NFKD", texte)
    ascii_texte = normalise.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_texte).strip("._-").lower()
    if not slug:
        raise ValueError("le nom doit contenir au moins une lettre ou un chiffre")
    return slug[:100]


def resoudre_modele(chemin: Path) -> Path:
    chemin = chemin.expanduser().resolve()
    if chemin.is_dir():
        favori = chemin / "best_model.keras"
        if favori.is_file():
            chemin = favori
        else:
            candidats = sorted((*chemin.glob("*.keras"), *chemin.glob("*.h5")))
            if len(candidats) != 1:
                raise FileNotFoundError(
                    f"Le dossier {chemin} doit contenir best_model.keras ou un seul modèle."
                )
            chemin = candidats[0]
    if not chemin.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {chemin}")
    return chemin


def sha256_fichier(chemin: Path) -> str:
    empreinte = hashlib.sha256()
    with chemin.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            empreinte.update(bloc)
    return empreinte.hexdigest()


def sha256_tableaux(*tableaux: np.ndarray) -> str:
    """Empreinte stable incluant forme, type et octets de chaque tableau."""
    empreinte = hashlib.sha256()
    for tableau in tableaux:
        contigu = np.ascontiguousarray(tableau)
        empreinte.update(str(contigu.shape).encode("ascii"))
        empreinte.update(str(contigu.dtype).encode("ascii"))
        empreinte.update(contigu.tobytes(order="C"))
    return empreinte.hexdigest()


def creer_chemins(args: argparse.Namespace, modele: Path) -> CheminsEvaluation:
    racine_sortie = args.sortie.expanduser().resolve()
    if args.nom_evaluation:
        nom = slugifier(args.nom_evaluation)
    else:
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        nom = slugifier(f"{modele.stem}_{horodatage}")
    racine = racine_sortie / nom
    if racine.exists():
        raise FileExistsError(
            f"Le dossier d'évaluation existe déjà et ne sera pas écrasé : {racine}"
        )
    return CheminsEvaluation(racine)


def ecrire_json_atomique(chemin: Path, contenu: dict[str, Any]) -> None:
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    temporaire.write_text(
        json.dumps(contenu, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporaire.replace(chemin)


def ecrire_csv_atomique(tableau: pd.DataFrame, chemin: Path, **kwargs: Any) -> None:
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    tableau.to_csv(temporaire, index=False, **kwargs)
    temporaire.replace(chemin)


def charger_mnist_test(
    chemin_npz: Path | None,
    limite: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, dict[str, Any]]:
    if chemin_npz is None:
        cache = Path.home() / ".keras" / "datasets" / "mnist.npz"
        if cache.is_file():
            chemin_npz = cache

    if chemin_npz is not None:
        chemin_npz = chemin_npz.expanduser().resolve()
        if not chemin_npz.is_file():
            raise FileNotFoundError(f"Archive MNIST introuvable : {chemin_npz}")
        with np.load(chemin_npz, allow_pickle=False) as archive:
            manque = {"x_test", "y_test"} - set(archive.files)
            if manque:
                raise ValueError(
                    "Archive MNIST incomplète ; clés absentes : "
                    + ", ".join(sorted(manque))
                )
            images = np.asarray(archive["x_test"])
            etiquettes = np.asarray(archive["y_test"])
        origine = str(chemin_npz)
    else:
        try:
            import keras

            (_, _), (images, etiquettes) = keras.datasets.mnist.load_data()
        except Exception as exc:
            raise RuntimeError(
                "MNIST n'est pas disponible hors ligne. Placez mnist.npz dans "
                "~/.keras/datasets/ ou utilisez --mnist-path."
            ) from exc
        origine = "keras.datasets.mnist"

    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Images MNIST attendues sous forme (N, 28, 28), reçu {images.shape}")
    etiquettes_brutes = np.asarray(etiquettes).reshape(-1)
    if len(images) != len(etiquettes_brutes):
        raise ValueError("Le nombre d'images et d'étiquettes MNIST diffère.")
    if len(images) == 0:
        raise ValueError("Le jeu de test MNIST est vide.")
    if not np.issubdtype(etiquettes_brutes.dtype, np.number):
        raise ValueError("Les étiquettes MNIST doivent être numériques.")
    etiquettes_flottantes = etiquettes_brutes.astype(np.float64)
    if not np.isfinite(etiquettes_flottantes).all():
        raise ValueError("Les étiquettes MNIST contiennent NaN ou une valeur infinie.")
    if np.any(etiquettes_flottantes != np.floor(etiquettes_flottantes)):
        raise ValueError("Les étiquettes MNIST doivent être des entiers exacts.")
    if np.any((etiquettes_flottantes < 0) | (etiquettes_flottantes > 9)):
        raise ValueError("Les étiquettes MNIST doivent être comprises entre 0 et 9.")
    etiquettes = etiquettes_flottantes.astype(np.int64)
    indices_source = np.arange(len(images), dtype=np.int64)
    taille_source = len(images)
    sous_echantillonnage_applique = limite is not None and limite < taille_source

    if sous_echantillonnage_applique:
        generateur = np.random.default_rng(GRAINE_SOUS_ECHANTILLON)
        indices = np.sort(generateur.choice(len(images), size=limite, replace=False))
        images = images[indices]
        etiquettes = etiquettes[indices]
        indices_source = indices_source[indices]

    images = images.astype(np.float32)
    division_255_appliquee = bool(images.size and float(np.max(images)) > 1.0)
    if division_255_appliquee:
        images /= 255.0
    if not np.isfinite(images).all() or float(np.min(images)) < 0 or float(np.max(images)) > 1:
        raise ValueError("Les pixels MNIST doivent être des valeurs finies comprises entre 0 et 1.")
    images = np.expand_dims(images, axis=-1)
    empreinte_sous_ensemble = sha256_tableaux(images, etiquettes, indices_source)
    preparation = {
        "taille_source": taille_source,
        "limite_demandee": limite,
        "taille_effective": len(images),
        "sous_echantillonnage_applique": sous_echantillonnage_applique,
        "methode_selection": (
            "tirage sans remise, indices triés"
            if sous_echantillonnage_applique
            else "tous les exemples"
        ),
        "graine_selection": (
            GRAINE_SOUS_ECHANTILLON if sous_echantillonnage_applique else None
        ),
        "division_255_appliquee": division_255_appliquee,
    }
    return (
        images,
        etiquettes,
        indices_source,
        origine,
        empreinte_sous_ensemble,
        preparation,
    )


def intervalle_wilson(
    succes: int,
    total: int,
    z: float = Z_95,
) -> tuple[float, float]:
    """Intervalle de Wilson bilatéral pour une proportion binomiale."""
    if total <= 0:
        return math.nan, math.nan
    proportion = succes / total
    denominateur = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominateur
    rayon = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominateur
    )
    return max(0.0, centre - rayon), min(1.0, centre + rayon)


def division_sure(numerateur: float, denominateur: float) -> float:
    return numerateur / denominateur if denominateur > 0 else math.nan


def calculer_metriques_par_classe(matrice: np.ndarray) -> pd.DataFrame:
    """Calcule les métriques one-vs-rest de chaque ligne/classe."""
    matrice = np.asarray(matrice)
    if matrice.shape != (10, 10):
        raise ValueError(f"Une matrice 10×10 est attendue, reçu {matrice.shape}.")
    if not np.issubdtype(matrice.dtype, np.number) or not np.isfinite(matrice).all():
        raise ValueError("La matrice doit contenir uniquement des nombres finis.")
    if np.any(matrice < 0) or np.any(matrice != np.floor(matrice)):
        raise ValueError("Les effectifs de la matrice doivent être des entiers positifs.")
    matrice = matrice.astype(np.int64)
    total = int(matrice.sum())
    if total <= 0:
        raise ValueError("La matrice de confusion est vide.")

    lignes: list[dict[str, Any]] = []
    for classe in CLASSES:
        tp = int(matrice[classe, classe])
        fn = int(matrice[classe, :].sum() - tp)
        fp = int(matrice[:, classe].sum() - tp)
        tn = total - tp - fn - fp
        support = tp + fn
        predits = tp + fp
        negatifs = tn + fp

        sensibilite = division_sure(tp, support)
        specificite = division_sure(tn, negatifs)
        if predits > 0:
            precision = tp / predits
        elif support > 0:
            # Politique zero_division=0 : une classe présente mais jamais
            # prédite a une précision et un F1 nuls.
            precision = 0.0
        else:
            precision = math.nan
        denominateur_f1 = 2 * tp + fp + fn
        f1 = 2 * tp / denominateur_f1 if denominateur_f1 > 0 else math.nan
        accuracy_ovr = (tp + tn) / total
        balanced = (
            (sensibilite + specificite) / 2
            if not math.isnan(sensibilite) and not math.isnan(specificite)
            else math.nan
        )
        sens_bas, sens_haut = intervalle_wilson(tp, support)
        spec_bas, spec_haut = intervalle_wilson(tn, negatifs)

        if support == 0:
            warnings.warn(
                f"La classe {classe} est absente : sensibilité indéfinie.",
                RuntimeWarning,
                stacklevel=2,
            )
        lignes.append(
            {
                "classe": classe,
                "support": support,
                "predictions_classe": predits,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "sensibilite": sensibilite,
                "sensibilite_ic95_bas": sens_bas,
                "sensibilite_ic95_haut": sens_haut,
                "specificite": specificite,
                "specificite_ic95_bas": spec_bas,
                "specificite_ic95_haut": spec_haut,
                "precision": precision,
                "f1": f1,
                "accuracy_ovr": accuracy_ovr,
                "balanced_accuracy_ovr": balanced,
                "fnr": 1 - sensibilite if not math.isnan(sensibilite) else math.nan,
                "fpr": 1 - specificite if not math.isnan(specificite) else math.nan,
            }
        )
    return pd.DataFrame(lignes)


def normaliser_matrice(matrice: np.ndarray, axe: int) -> np.ndarray:
    matrice = np.asarray(matrice, dtype=float)
    sommes = matrice.sum(axis=axe, keepdims=True)
    resultat = np.full_like(matrice, np.nan, dtype=float)
    return np.divide(matrice, sommes, out=resultat, where=sommes != 0)


def calculer_resume(matrice: np.ndarray, metriques: pd.DataFrame) -> pd.DataFrame:
    total = int(np.asarray(matrice).sum())
    accuracy = float(np.trace(matrice) / total)
    support = metriques["support"].to_numpy(dtype=float)
    sensibilite = metriques["sensibilite"].to_numpy(dtype=float)
    rappel_pondere = float(np.nansum(sensibilite * support) / support.sum())
    return pd.DataFrame(
        [
            {
                "images_test": total,
                "accuracy_globale": accuracy,
                "rappel_pondere": rappel_pondere,
                "sensibilite_macro": float(metriques["sensibilite"].mean(skipna=True)),
                "specificite_macro": float(metriques["specificite"].mean(skipna=True)),
                "precision_macro": float(metriques["precision"].mean(skipna=True)),
                "f1_macro": float(metriques["f1"].mean(skipna=True)),
                "balanced_accuracy_ovr_macro": float(
                    metriques["balanced_accuracy_ovr"].mean(skipna=True)
                ),
            }
        ]
    )


def verifier_coherence(
    matrice: np.ndarray,
    metriques: pd.DataFrame,
    etiquettes_reelles: np.ndarray | None = None,
) -> None:
    total = int(matrice.sum())
    if total <= 0 or matrice.shape != (10, 10):
        raise AssertionError("Matrice de confusion invalide.")
    for ligne in metriques.itertuples(index=False):
        if ligne.tp + ligne.fn + ligne.fp + ligne.tn != total:
            raise AssertionError(f"Décomposition incohérente pour la classe {ligne.classe}.")
        for nom in (
            "sensibilite",
            "specificite",
            "precision",
            "f1",
            "accuracy_ovr",
            "balanced_accuracy_ovr",
            "fnr",
            "fpr",
        ):
            valeur = getattr(ligne, nom)
            if not math.isnan(valeur) and not 0 <= valeur <= 1:
                raise AssertionError(f"{nom} hors de [0,1] pour la classe {ligne.classe}.")
    if etiquettes_reelles is not None:
        effectifs = np.bincount(etiquettes_reelles, minlength=10)
        if not np.array_equal(matrice.sum(axis=1), effectifs):
            raise AssertionError("Les lignes de la matrice ne correspondent pas aux supports.")


def matrice_vers_dataframe(matrice: np.ndarray) -> pd.DataFrame:
    tableau = pd.DataFrame(matrice, columns=[f"predit_{i}" for i in CLASSES])
    tableau.insert(0, "classe_reelle", CLASSES)
    return tableau


def lire_matrice_csv(chemin: Path) -> np.ndarray:
    tableau = pd.read_csv(chemin)
    return tableau[[f"predit_{i}" for i in CLASSES]].to_numpy()


def tracer_matrice(
    matrice: np.ndarray,
    chemin: Path,
    titre: str,
    normalisee: bool,
    dpi: int,
    afficher: bool,
) -> None:
    figure, axe = plt.subplots(figsize=(9, 7.5))
    if normalisee:
        donnees = matrice * 100
        format_annotation = ".1f"
        etiquette_barre = "Pourcentage des vrais chiffres (%)"
    else:
        donnees = matrice
        format_annotation = "g"
        etiquette_barre = "Nombre d'images"
    sns.heatmap(
        donnees,
        annot=True,
        fmt=format_annotation,
        cmap="Blues",
        vmin=0,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar_kws={"label": etiquette_barre},
        ax=axe,
    )
    axe.set_title(titre)
    axe.set_xlabel("Classe prédite")
    axe.set_ylabel("Classe réelle")
    figure.tight_layout()
    figure.savefig(chemin, dpi=dpi, bbox_inches="tight")
    if afficher:
        plt.show()
    plt.close(figure)


def _annoter_barres(
    axe: Any,
    barres: Any,
    pourcentage: bool = True,
    couleur_interieure: str = "#25313c",
) -> None:
    plafond = float(axe.get_ylim()[1])
    for barre in barres:
        valeur = float(barre.get_height())
        if not math.isfinite(valeur) or valeur <= 0:
            continue
        texte = f"{valeur * 100:.1f}" if pourcentage else f"{valeur:.3f}"
        if valeur >= plafond * 0.75:
            position_y = valeur - plafond * 0.018
            alignement_vertical = "top"
            couleur = couleur_interieure
        else:
            position_y = valeur + max(plafond * 0.012, 0.001)
            alignement_vertical = "bottom"
            couleur = "#25313c"
        axe.text(
            barre.get_x() + barre.get_width() / 2,
            position_y,
            texte,
            ha="center",
            va=alignement_vertical,
            fontsize=7,
            rotation=90,
            color=couleur,
        )


def tracer_sensibilite_specificite(
    metriques: pd.DataFrame,
    chemin: Path,
    dpi: int,
    afficher: bool,
) -> None:
    x = np.arange(len(metriques))
    largeur = 0.38
    sens = metriques["sensibilite"].to_numpy(float)
    spec = metriques["specificite"].to_numpy(float)
    sens_err = np.vstack(
        (
            sens - metriques["sensibilite_ic95_bas"].to_numpy(float),
            metriques["sensibilite_ic95_haut"].to_numpy(float) - sens,
        )
    )
    spec_err = np.vstack(
        (
            spec - metriques["specificite_ic95_bas"].to_numpy(float),
            metriques["specificite_ic95_haut"].to_numpy(float) - spec,
        )
    )
    figure, axe = plt.subplots(figsize=(12, 6.8))
    barres_sens = axe.bar(
        x - largeur / 2,
        sens,
        largeur,
        yerr=sens_err,
        capsize=3,
        color="#3478b8",
        edgecolor="#173f62",
        label="Sensibilité (rappel)",
    )
    barres_spec = axe.bar(
        x + largeur / 2,
        spec,
        largeur,
        yerr=spec_err,
        capsize=3,
        color="#f3c969",
        edgecolor="#725719",
        hatch="//",
        label="Spécificité",
    )
    axe.set_title(
        "Sensibilité et spécificité par chiffre\n"
        "Barres d'erreur : intervalles de Wilson à 95 % sur les images de test"
    )
    axe.set_xlabel("Chiffre")
    axe.set_ylabel("Proportion")
    axe.set_xticks(x, metriques["classe"].astype(str))
    axe.set_ylim(0, 1.025)
    axe.yaxis.set_major_formatter(PercentFormatter(1.0))
    axe.grid(axis="y", color="#d9dee3", linewidth=0.7)
    axe.set_axisbelow(True)
    axe.legend(loc="lower right")
    _annoter_barres(axe, barres_sens, couleur_interieure="white")
    _annoter_barres(axe, barres_spec, couleur_interieure="#4d3b13")
    figure.tight_layout()
    figure.savefig(chemin, dpi=dpi, bbox_inches="tight")
    if afficher:
        plt.show()
    plt.close(figure)


def tracer_taux_erreurs(
    metriques: pd.DataFrame,
    chemin: Path,
    dpi: int,
    afficher: bool,
) -> None:
    x = np.arange(len(metriques))
    largeur = 0.38
    fnr = metriques["fnr"].to_numpy(float)
    fpr = metriques["fpr"].to_numpy(float)
    maximum = float(np.nanmax(np.concatenate((fnr, fpr))))
    plafond = min(1.0, max(0.02, maximum * 1.32))
    figure, axe = plt.subplots(figsize=(12, 6.8))
    barres_fnr = axe.bar(
        x - largeur / 2,
        fnr,
        largeur,
        color="#3478b8",
        edgecolor="#173f62",
        label="FNR : vrais chiffres manqués",
    )
    barres_fpr = axe.bar(
        x + largeur / 2,
        fpr,
        largeur,
        color="#f3c969",
        edgecolor="#725719",
        hatch="//",
        label="FPR : autres chiffres pris pour cette classe",
    )
    axe.set_title(
        "Taux d'erreurs par chiffre\n"
        "Axe démarrant à zéro ; FNR = 1 − sensibilité, FPR = 1 − spécificité"
    )
    axe.set_xlabel("Chiffre")
    axe.set_ylabel("Taux d'erreur")
    axe.set_xticks(x, metriques["classe"].astype(str))
    axe.set_ylim(0, plafond)
    axe.yaxis.set_major_formatter(PercentFormatter(1.0))
    axe.grid(axis="y", color="#d9dee3", linewidth=0.7)
    axe.set_axisbelow(True)
    axe.legend(loc="upper right")
    _annoter_barres(axe, barres_fnr)
    _annoter_barres(axe, barres_fpr)
    figure.tight_layout()
    figure.savefig(chemin, dpi=dpi, bbox_inches="tight")
    if afficher:
        plt.show()
    plt.close(figure)


def versions_logiciels() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "pandas", "scikit-learn", "keras", "tensorflow"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "non détecté"
    return versions


def ecrire_documentation(
    chemins: CheminsEvaluation,
    modele: Path,
    modele_sha256: str,
    origine_mnist: str,
    empreinte_sous_ensemble: str,
    indices_source: np.ndarray,
    preparation_dataset: dict[str, Any],
    type_sortie_modele: str,
    args: argparse.Namespace,
    resume: pd.DataFrame,
) -> None:
    protocole = {
        "schema_version": 2,
        "dataset": "MNIST test",
        "sous_ensemble_sha256": empreinte_sous_ensemble,
        "indices_source_sha256": sha256_tableaux(indices_source),
        "nombre_images": int(len(indices_source)),
        "selection": {
            "limite_demandee": preparation_dataset["limite_demandee"],
            "taille_source": preparation_dataset["taille_source"],
            "taille_effective": preparation_dataset["taille_effective"],
            "sous_echantillonnage_applique": preparation_dataset[
                "sous_echantillonnage_applique"
            ],
            "methode": preparation_dataset["methode_selection"],
            "graine": preparation_dataset["graine_selection"],
        },
        "normalisation": {
            "description": "conversion float32, division par 255 si nécessaire, canal ajouté",
            "division_255_appliquee": preparation_dataset[
                "division_255_appliquee"
            ],
        },
        "classes": list(CLASSES),
        "metriques": {
            "version": 2,
            "definition": "one-vs-rest par classe",
            "precision_zero_division": "0 si classe présente mais jamais prédite, NaN si classe absente et jamais prédite",
            "balanced_accuracy": "moyenne one-vs-rest de la sensibilité et de la spécificité",
        },
        "intervalle": "Wilson bilatéral 95 % pour sensibilité et spécificité",
    }
    protocole_hex = hashlib.sha256(
        json.dumps(protocole, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    protocole_id = f"sha256:{protocole_hex}"
    evaluation_id = "sha256:" + hashlib.sha256(
        f"{protocole_id}|{modele_sha256}".encode("utf-8")
    ).hexdigest()
    configuration = {
        "created_at_utc": maintenant_utc(),
        "protocol_id": protocole_id,
        "evaluation_id": evaluation_id,
        "protocole": protocole,
        "dataset": {
            "source": origine_mnist,
            "sous_ensemble_sha256": empreinte_sous_ensemble,
            "indices_csv": str(chemins.indices_csv.relative_to(chemins.racine)),
        },
        "modele": {
            "chemin": str(modele),
            "sha256": modele_sha256,
            "taille_octets": modele.stat().st_size,
            "type_sortie_detecte": type_sortie_modele,
        },
        "options": {
            "sauver_predictions": bool(args.sauver_predictions),
            "dpi": args.dpi,
        },
        "commande": [sys.executable, *sys.argv],
        "resultat_principal": resume.iloc[0].to_dict(),
        "versions": versions_logiciels(),
    }
    ecrire_json_atomique(chemins.configuration, configuration)

    chemins.guide.write_text(
        f"""ÉVALUATION PAR CLASSE MNIST
=============================

Modèle : {modele}
Empreinte SHA-256 : {modele_sha256}
Source MNIST : {origine_mnist}
Protocol ID : {protocole_id}
Evaluation ID : {evaluation_id}

OÙ TROUVER LES RÉSULTATS
------------------------
- donnees/matrice_confusion_effectifs.csv : erreurs exactes entre chaque paire de chiffres.
- donnees/metriques_par_classe.csv : TP, FN, FP, TN, sensibilité, spécificité,
  précision, F1, accuracy one-vs-rest et balanced accuracy one-vs-rest.
- donnees/indices_images_test.csv : correspondance avec les indices MNIST d'origine.
- donnees/resume_global.csv : métriques globales et moyennes macro.
- graphiques/ : figures construites à partir des CSV exportés.

COMMENT LIRE LES MÉTRIQUES
--------------------------
- Sensibilité : parmi les vrais chiffres c, proportion reconnue comme c.
- Spécificité : parmi les autres chiffres, proportion non confondue avec c.
- FNR : proportion des vrais c manqués. FPR : fausses alertes pour c.
- Accuracy one-vs-rest : souvent très haute parce qu'environ 90 % des images
  sont négatives pour une classe. Pour comparer les chiffres, privilégier la
  sensibilité, la spécificité, la balanced accuracy one-vs-rest, la précision et le F1.

Les intervalles de Wilson mesurent l'incertitude liée au nombre d'images du jeu
de test pour cette évaluation. Ils ne mesurent pas la variabilité entre plusieurs
entraînements : cette dernière doit être étudiée avec plusieurs graines.
""",
        encoding="utf-8",
    )


def ecrire_catalogue(chemins: CheminsEvaluation) -> None:
    descriptions = {
        chemins.configuration: "Paramètres, provenance, empreinte du modèle et protocole.",
        chemins.guide: "Guide de lecture de cette évaluation.",
        chemins.matrice_effectifs_csv: "Matrice de confusion 10×10 en effectifs.",
        chemins.matrice_reel_csv: "Matrice normalisée par classe réelle (sensibilité).",
        chemins.matrice_predit_csv: "Matrice normalisée par classe prédite (lecture de la précision).",
        chemins.metriques_csv: "Métriques one-vs-rest et IC 95 % pour chaque chiffre.",
        chemins.resume_csv: "Résumé global et moyennes macro.",
        chemins.indices_csv: "Indices des images dans le jeu de test MNIST source.",
        chemins.predictions_csv: "Prédictions et scores du modèle image par image.",
        chemins.matrice_effectifs_png: "Matrice de confusion en effectifs.",
        chemins.matrice_reel_png: "Matrice de confusion normalisée par ligne.",
        chemins.metriques_png: "Sensibilité et spécificité avec IC Wilson 95 %.",
        chemins.erreurs_png: "Taux de faux négatifs et faux positifs.",
    }
    lignes = []
    for chemin, description in descriptions.items():
        if chemin.exists():
            lignes.append(
                {
                    "fichier": str(chemin.relative_to(chemins.racine)),
                    "type": chemin.suffix.lstrip(".") or "dossier",
                    "description": description,
                }
            )
    ecrire_csv_atomique(pd.DataFrame(lignes), chemins.catalogue)


def evaluer(args: argparse.Namespace) -> CheminsEvaluation:
    modele = resoudre_modele(args.modele)
    chemins = creer_chemins(args, modele)
    modele_sha256 = sha256_fichier(modele)
    (
        images,
        etiquettes,
        indices_source,
        origine_mnist,
        empreinte_sous_ensemble,
        preparation_dataset,
    ) = charger_mnist_test(
        args.mnist_path,
        args.test_limit,
    )

    try:
        import keras
    except ImportError as exc:
        raise RuntimeError("Keras est nécessaire pour charger le modèle.") from exc
    reseau = keras.models.load_model(modele)
    scores = np.asarray(reseau.predict(images, verbose=0))
    if scores.shape != (len(images), 10):
        raise ValueError(
            "Le modèle doit produire dix scores par image ; "
            f"forme obtenue : {scores.shape}."
        )
    if not np.isfinite(scores).all():
        raise ValueError("Les prédictions contiennent NaN ou une valeur infinie.")
    sommes_scores = scores.sum(axis=1)
    sont_probabilites = bool(
        np.all(scores >= -1e-7)
        and np.all(scores <= 1 + 1e-7)
        and np.allclose(sommes_scores, 1.0, rtol=1e-5, atol=1e-5)
    )
    type_sortie_modele = (
        "probabilites_normalisees" if sont_probabilites else "scores_non_normalises"
    )
    predictions = np.argmax(scores, axis=1)
    matrice = confusion_matrix(etiquettes, predictions, labels=CLASSES)
    metriques = calculer_metriques_par_classe(matrice)
    verifier_coherence(matrice, metriques, etiquettes)
    resume = calculer_resume(matrice, metriques)

    chemins.creer()
    ecrire_csv_atomique(matrice_vers_dataframe(matrice), chemins.matrice_effectifs_csv)
    ecrire_csv_atomique(
        matrice_vers_dataframe(normaliser_matrice(matrice, axe=1)),
        chemins.matrice_reel_csv,
        float_format="%.10f",
    )
    ecrire_csv_atomique(
        matrice_vers_dataframe(normaliser_matrice(matrice, axe=0)),
        chemins.matrice_predit_csv,
        float_format="%.10f",
    )
    ecrire_csv_atomique(metriques, chemins.metriques_csv, float_format="%.10f")
    ecrire_csv_atomique(resume, chemins.resume_csv, float_format="%.10f")
    indices_df = pd.DataFrame(
        {
            "index_evaluation": np.arange(len(etiquettes)),
            "index_source_mnist_test": indices_source,
            "classe_reelle": etiquettes,
        }
    )
    ecrire_csv_atomique(indices_df, chemins.indices_csv)

    if args.sauver_predictions:
        predictions_df = pd.DataFrame(
            {
                "index_evaluation": np.arange(len(etiquettes)),
                "index_source_mnist_test": indices_source,
                "classe_reelle": etiquettes,
                "classe_predite": predictions,
                "correct": predictions == etiquettes,
                "score_max": scores.max(axis=1),
            }
        )
        for classe in CLASSES:
            predictions_df[f"score_{classe}"] = scores[:, classe]
        ecrire_csv_atomique(
            predictions_df,
            chemins.predictions_csv,
            float_format="%.10f",
        )

    # Les figures relisent volontairement les CSV : les données sources restent auditables.
    matrice_exportee = lire_matrice_csv(chemins.matrice_effectifs_csv)
    matrice_reel_exportee = lire_matrice_csv(chemins.matrice_reel_csv)
    metriques_exportees = pd.read_csv(chemins.metriques_csv)
    tracer_matrice(
        matrice_exportee,
        chemins.matrice_effectifs_png,
        "Matrice de confusion MNIST — effectifs",
        False,
        args.dpi,
        args.afficher,
    )
    tracer_matrice(
        matrice_reel_exportee,
        chemins.matrice_reel_png,
        "Matrice de confusion MNIST — normalisée par classe réelle",
        True,
        args.dpi,
        args.afficher,
    )
    tracer_sensibilite_specificite(
        metriques_exportees,
        chemins.metriques_png,
        args.dpi,
        args.afficher,
    )
    tracer_taux_erreurs(
        metriques_exportees,
        chemins.erreurs_png,
        args.dpi,
        args.afficher,
    )
    ecrire_documentation(
        chemins,
        modele,
        modele_sha256,
        origine_mnist,
        empreinte_sous_ensemble,
        indices_source,
        preparation_dataset,
        type_sortie_modele,
        args,
        resume,
    )
    ecrire_catalogue(chemins)
    return chemins


def main(argv: Sequence[str] | None = None) -> int:
    parser = construire_parser()
    args = parser.parse_args(argv)
    valider_arguments(parser, args)
    try:
        chemins = evaluer(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print("Évaluation terminée.")
    print(f"Dossier : {chemins.racine}")
    print(f"Métriques par classe : {chemins.metriques_csv}")
    print(f"Résumé global : {chemins.resume_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
