#!/usr/bin/env python3
"""Explorer les hyperparamètres d'un CNN MNIST et produire des cartes lisibles.

Le script sépare volontairement les résultats bruts, les résultats agrégés et les
graphiques. Une exécution interrompue peut être reprise : les essais déjà réussis
dans ``resultats_bruts.csv`` sont ignorés, sauf si ``--force`` est demandé.

Exemple d'essai court :
    python scripts/experiences/generer_color_map.py --nom-experience essai_rapide \
        --filters-1 4,8 --filters-2 8,16 --dropouts 0.2,0.4 \
        --repetitions 1 --epochs 1 --train-limit 2000 --test-limit 500

Régénération des graphiques sans réentraîner :
    python scripts/experiences/generer_color_map.py --nom-experience essai_rapide --plot-only

Surface dense autour du meilleur dropout observé :
    python scripts/experiences/generer_color_map.py \
        --nom-experience surface_dense_dropout_04 \
        --preset-surface-dropout-04 --confirmer-grande-grille

Criblage des limites jusqu'à 128 filtres, ReLU et Softmax convolutifs :
    python scripts/experiences/generer_color_map.py \
        --nom-experience limite_128_criblage \
        --preset-limite-128 --confirmer-grande-grille
"""

from __future__ import annotations

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401


import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from reconnaissance_chiffres import config as env_config


os.environ["KERAS_BACKEND"] = "tensorflow"

OUTPUT_ROOT = env_config.SORTIES_HYPERPARAMETRES
SURFACE_DENSE_FILTERS_1 = (4, 8, 12, 16, 20, 24, 32)
SURFACE_DENSE_FILTERS_2 = (8, 16, 24, 32, 40, 48, 64)
SURFACE_DENSE_DROPOUT = 0.4
LIMIT_128_FILTERS_1 = (4, 8, 16, 32, 64, 128)
LIMIT_128_FILTERS_2 = (8, 16, 32, 64, 128)
LIMIT_128_DROPOUTS = (0.0, 0.2, 0.4, 0.6, 0.8)
SUPPORTED_CONV_ACTIVATIONS = ("relu", "softmax")
OUTPUT_ACTIVATION = "softmax"
MAX_RANDOM_SEED = 2**32 - 1
NO_SUCCESS_EXIT_CODE = 2
DATASET_HASH_UNAVAILABLE = "indisponible_avant_chargement_keras"
RAW_COLUMNS = [
    "experiment_name",
    "protocol_id",
    "run_id",
    "conv_activation",
    "output_activation",
    "filter_1",
    "filter_2",
    "dropout",
    "repetition",
    "seed",
    "repetitions_requested",
    "epochs_requested",
    "epochs_completed",
    "best_epoch",
    "batch_size",
    "validation_split",
    "learning_rate",
    "train_samples",
    "validation_samples",
    "test_samples",
    "model_parameters",
    "best_val_accuracy",
    "best_val_loss",
    "test_accuracy",
    "test_loss",
    "duration_seconds",
    "status",
    "error_message",
    "completed_at_utc",
]


class ExperimentPaths:
    """Chemins centralisés d'une expérience."""

    def __init__(self, experiment_slug: str) -> None:
        self.root = OUTPUT_ROOT / experiment_slug
        self.data = self.root / "donnees"
        self.figures = self.root / "graphiques"
        self.analysis_data = self.data / "analyse"
        self.correlation_figures = self.figures / "correlations"
        self.distribution_figures = self.figures / "distributions"
        self.raw_csv = self.data / "resultats_bruts.csv"
        self.aggregated_csv = self.data / "resultats_agreges.csv"
        self.configuration = self.root / "configuration.json"
        self.catalog = self.root / "catalogue_sorties.csv"
        self.guide = self.root / "LISEZ_MOI.txt"
        self.heatmaps = self.figures / "heatmaps_2d_accuracy_moyenne.png"
        self.scatter_3d = self.figures / "nuage_3d_accuracy_moyenne.png"
        self.surface_details = self.figures / "surfaces_3d_detaillees"
        self.surface_relu = self.surface_details / "activation_relu.png"
        self.surface_softmax = self.surface_details / "activation_softmax.png"
        self.analytical_csv = (
            self.analysis_data / "configurations_analytiques.csv"
        )
        self.spearman_csv = self.analysis_data / "correlations_spearman.csv"
        self.pearson_csv = self.analysis_data / "correlations_pearson.csv"
        self.analysis_catalog = self.analysis_data / "catalogue_analyse.csv"
        self.scatter_relationships = (
            self.correlation_figures / "scatter_accuracy_hyperparametres.png"
        )
        self.scatter_cost = (
            self.correlation_figures / "scatter_cout_performance.png"
        )
        self.correlation_matrix = (
            self.correlation_figures / "matrice_correlation_spearman.png"
        )
        self.accuracy_histograms = (
            self.distribution_figures / "histogrammes_accuracy.png"
        )
        self.stability_histograms = (
            self.distribution_figures / "histogrammes_stabilite_duree.png"
        )

    def create(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.analysis_data.mkdir(parents=True, exist_ok=True)
        self.correlation_figures.mkdir(parents=True, exist_ok=True)
        self.distribution_figures.mkdir(parents=True, exist_ok=True)
        self.surface_details.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """Calculer une empreinte de contenu sans charger tout le fichier en mémoire."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Identifier exactement la version du générateur ayant défini le protocole."""
    return sha256_file(Path(__file__).resolve())


def slugify_experiment_name(value: str) -> str:
    """Créer un nom de dossier sûr tout en acceptant les accents et les espaces."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value).strip("._-").lower()
    if not slug:
        raise argparse.ArgumentTypeError(
            "le nom de l'expérience doit contenir au moins une lettre ou un chiffre"
        )
    return slug[:80]


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "utiliser des entiers séparés par des virgules, par exemple 4,8,16"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("tous les nombres de filtres doivent être > 0")
    return values


def comma_separated_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "utiliser des nombres séparés par des virgules, par exemple 0.1,0.2,0.3"
        ) from exc
    if not values or any(
        not math.isfinite(item) or item < 0 or item >= 1 for item in values
    ):
        raise argparse.ArgumentTypeError("chaque dropout doit vérifier 0 <= dropout < 1")
    return values


def comma_separated_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "utiliser des graines entières séparées par des virgules"
        ) from exc
    if not seeds or any(seed < 0 or seed > MAX_RANDOM_SEED for seed in seeds):
        raise argparse.ArgumentTypeError(
            f"les graines doivent être comprises entre 0 et {MAX_RANDOM_SEED}"
        )
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("les graines doivent être uniques")
    return seeds


def comma_separated_activations(value: str) -> tuple[str, ...]:
    activations = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    if not activations:
        raise argparse.ArgumentTypeError("fournir au moins une activation")
    unsupported = [
        activation
        for activation in activations
        if activation not in SUPPORTED_CONV_ACTIVATIONS
    ]
    if unsupported:
        supported = ", ".join(SUPPORTED_CONV_ACTIVATIONS)
        raise argparse.ArgumentTypeError(
            f"activation(s) inconnue(s) : {', '.join(unsupported)} ; "
            f"valeurs acceptées : {supported}"
        )
    return activations


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("une valeur entière est attendue") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("la valeur doit être strictement positive")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("une valeur entière est attendue") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("la valeur doit être positive ou nulle")
    return parsed


def probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("une valeur décimale est attendue") from exc
    if not math.isfinite(parsed) or not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("la valeur doit être strictement comprise entre 0 et 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne une grille de CNN MNIST avec répétitions, sauvegarde chaque "
            "essai dans un CSV reprenable, puis génère des heatmaps 2D, un nuage "
            "3D et des surfaces 3D par dropout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nom-experience",
        "--experiment-name",
        default="grille_dense",
        help="nom du sous-dossier d'expérience",
    )
    parser.add_argument(
        "--filters-1",
        "--filtres-1",
        type=comma_separated_ints,
        default=(4, 8, 16),
        metavar="LISTE",
        help="valeurs de filter_1 séparées par des virgules",
    )
    parser.add_argument(
        "--filters-2",
        "--filtres-2",
        type=comma_separated_ints,
        default=(8, 16, 32),
        metavar="LISTE",
        help="valeurs de filter_2 séparées par des virgules",
    )
    parser.add_argument(
        "--dropouts",
        type=comma_separated_floats,
        default=(0.1, 0.2, 0.3, 0.4, 0.5),
        metavar="LISTE",
        help="taux de dropout séparés par des virgules",
    )
    parser.add_argument(
        "--activations",
        "--activation-conv",
        type=comma_separated_activations,
        default=("relu",),
        metavar="LISTE",
        help=(
            "activation commune aux deux Conv2D, séparée par des virgules ; "
            "la sortie reste toujours softmax"
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=3,
        help="nombre d'entraînements indépendants par combinaison",
    )
    parser.add_argument(
        "--seeds",
        type=comma_separated_seeds,
        default=None,
        metavar="LISTE",
        help="graines explicites ; remplace --repetitions et --seed-base",
    )
    parser.add_argument(
        "--seed-base",
        type=non_negative_int,
        default=42,
        help="première graine lorsque --seeds n'est pas fourni",
    )
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--patience", type=non_negative_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument(
        "--validation-split",
        type=probability,
        default=0.15,
        help="fraction des données d'entraînement réservée à la validation",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--train-limit",
        type=positive_int,
        default=None,
        help="sous-échantillon fixe pour un essai rapide ; vide = tout MNIST",
    )
    parser.add_argument(
        "--test-limit",
        type=positive_int,
        default=None,
        help="limite fixe du jeu de test ; vide = tout MNIST",
    )
    parser.add_argument(
        "--dataset-seed",
        type=non_negative_int,
        default=2026,
        help="graine du sous-échantillonnage, identique pour tous les essais",
    )
    parser.add_argument(
        "--mnist-path",
        type=Path,
        default=None,
        help=(
            "fichier mnist.npz local ; sinon recherche dans les caches du projet "
            "et de l'utilisateur avant le téléchargement Keras"
        ),
    )
    parser.add_argument(
        "--cascade-dir",
        type=Path,
        default=None,
        help=(
            "dossier dataset_numpy contenant x_train_cascade.npy, "
            "y_train_cascade.npy et ids_train_cascade.npy ; les exemples "
            "Cascade sont ajoutés uniquement à l'entraînement"
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="niveau d'affichage de Keras",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="relire le CSV et recréer les graphiques sans charger Keras ni entraîner",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="réentraîner et remplacer les essais déjà réussis pour la grille demandée",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="ouvrir aussi les figures à l'écran après leur sauvegarde",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="afficher la grille et les chemins sans créer de fichiers ni entraîner",
    )
    parser.add_argument(
        "--preset-rapide",
        action="store_true",
        help=(
            "smoke test : grille 2 x 2 x 2, 1 répétition, 1 époque, "
            "2 000 images d'entraînement et 500 de test"
        ),
    )
    parser.add_argument(
        "--preset-surface-dropout-04",
        action="store_true",
        help=(
            "grille dense 7 x 7 pour une surface plus détaillée : "
            "filter_1=4,8,12,16,20,24,32 ; "
            "filter_2=8,16,24,32,40,48,64 ; dropout=0.4 uniquement"
        ),
    )
    parser.add_argument(
        "--preset-limite-128",
        action="store_true",
        help=(
            "criblage des limites : filter_1=4,8,16,32,64,128 ; "
            "filter_2=8,16,32,64,128 ; dropout=0,0.2,0.4,0.6,0.8 ; "
            "activations=relu,softmax ; une répétition (300 entraînements)"
        ),
    )
    parser.add_argument(
        "--arreter-sur-erreur",
        "--stop-on-error",
        action="store_true",
        help=(
            "arrêter la grille au premier échec ; par défaut les erreurs et OOM "
            "sont cataloguées puis la grille continue"
        ),
    )
    parser.add_argument(
        "--confirmer-grande-grille",
        action="store_true",
        help="autoriser explicitement une invocation de 50 entraînements ou plus",
    )
    return parser


def resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.seeds is not None:
        seeds = tuple(args.seeds)
    else:
        seeds = tuple(args.seed_base + offset for offset in range(args.repetitions))
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(seed < 0 or seed > MAX_RANDOM_SEED for seed in seeds)
    ):
        raise ValueError(
            f"les graines résolues doivent être uniques et comprises entre 0 et "
            f"{MAX_RANDOM_SEED}"
        )
    return seeds


def dataset_reference(args: argparse.Namespace) -> str:
    return (
        str(args.mnist_path_resolved)
        if args.mnist_path_resolved is not None
        else "cache Keras ou téléchargement automatique"
    )


def cascade_dataset_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Décrire précisément le complément Cascade utilisé par le protocole."""
    if getattr(args, "cascade_dir_resolved", None) is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "directory": str(args.cascade_dir_resolved),
        "x_file": str(args.cascade_x_path),
        "x_sha256": args.cascade_x_sha256,
        "y_file": str(args.cascade_y_path),
        "y_sha256": args.cascade_y_sha256,
        "ids_file": str(args.cascade_ids_path),
        "ids_sha256": args.cascade_ids_sha256,
        "manifest_file": str(args.cascade_manifest_path),
        "manifest_sha256": args.cascade_manifest_sha256,
        "samples": args.cascade_samples,
        "class_distribution": args.cascade_class_distribution,
        "split_policy": "Cascade exclusivement entraînement",
    }


def dataset_display_name(args: argparse.Namespace) -> str:
    if getattr(args, "cascade_dir_resolved", None) is not None:
        return "MNIST Keras + Cascade Top-N validé"
    return "MNIST Keras"


def validation_policy(args: argparse.Namespace) -> str:
    if getattr(args, "cascade_dir_resolved", None) is not None:
        return (
            "validation MNIST historique prélevée avant ajout de Cascade ; "
            "Cascade exclusivement entraînement"
        )
    return "validation_split Keras sur MNIST"


def build_protocol_payload(
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Décrire tout ce qui doit rester identique pour reprendre sans mélange."""
    return {
        "schema_version": 2,
        "input": {
            "dataset": dataset_display_name(args),
            "dataset_file": dataset_reference(args),
            "dataset_sha256": getattr(args, "mnist_sha256", None)
            or DATASET_HASH_UNAVAILABLE,
            "cascade": cascade_dataset_payload(args),
            "image_shape": [28, 28, 1],
            "normalization": "float32 divisé par 255",
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "dataset_seed": args.dataset_seed,
        },
        "grid": {
            "preset": active_preset_name(args),
            "filter_1": list(args.filters_1),
            "filter_2": list(args.filters_2),
            "dropout": list(args.dropouts),
            "conv_activation": list(args.activations),
            "output_activation": OUTPUT_ACTIVATION,
            "seeds": list(seeds),
        },
        "training": {
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "validation_policy": validation_policy(args),
            "learning_rate": args.learning_rate,
            "kernel_sizes": [[5, 5], [5, 5]],
            "activation_policy": "même activation pour les deux Conv2D",
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy",
        },
        "implementation": {
            "generator_sha256": implementation_sha256(),
            "protocol_schema": "color-map-v2-content-addressed",
        },
    }


def make_protocol_id(args: argparse.Namespace, seeds: Sequence[int]) -> str:
    encoded = json.dumps(
        build_protocol_payload(args, seeds),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate doit être strictement positif")
    if args.seed_base < 0 or args.seed_base > MAX_RANDOM_SEED:
        parser.error(f"--seed-base doit être compris entre 0 et {MAX_RANDOM_SEED}")
    if args.dataset_seed < 0 or args.dataset_seed >= MAX_RANDOM_SEED:
        parser.error(
            f"--dataset-seed doit être compris entre 0 et {MAX_RANDOM_SEED - 1} "
            "(la graine suivante est réservée au jeu de test)"
        )
    if args.plot_only and args.force:
        parser.error("--plot-only et --force ne peuvent pas être utilisés ensemble")
    selected_presets = sum(
        bool(value)
        for value in (
            args.preset_rapide,
            args.preset_surface_dropout_04,
            args.preset_limite_128,
        )
    )
    if selected_presets > 1:
        parser.error(
            "--preset-rapide, --preset-surface-dropout-04 et "
            "--preset-limite-128 sont incompatibles entre eux"
        )


def apply_quick_preset(args: argparse.Namespace) -> None:
    """Appliquer un petit preset déterministe, pratique pour valider le pipeline."""
    if not args.preset_rapide:
        return
    args.filters_1 = (4, 8)
    args.filters_2 = (8, 16)
    args.dropouts = (0.2, 0.4)
    args.activations = ("relu",)
    args.repetitions = 1
    args.seeds = None
    args.epochs = 1
    args.patience = 0
    args.train_limit = 2_000
    args.test_limit = 500


def apply_dense_surface_preset(args: argparse.Namespace) -> None:
    """Cibler dropout 0,4 avec assez de points pour une vraie surface 7 x 7."""
    if not args.preset_surface_dropout_04:
        return
    args.filters_1 = SURFACE_DENSE_FILTERS_1
    args.filters_2 = SURFACE_DENSE_FILTERS_2
    args.dropouts = (SURFACE_DENSE_DROPOUT,)
    args.activations = ("relu",)


def apply_limit_128_preset(args: argparse.Namespace) -> None:
    """Cribler largement filtres, dropout et activations avant raffinement."""
    if not args.preset_limite_128:
        return
    args.filters_1 = LIMIT_128_FILTERS_1
    args.filters_2 = LIMIT_128_FILTERS_2
    args.dropouts = LIMIT_128_DROPOUTS
    args.activations = SUPPORTED_CONV_ACTIVATIONS
    args.repetitions = 1
    args.seeds = None


def active_preset_name(args: argparse.Namespace) -> str | None:
    if args.preset_limite_128:
        return "limite_128_criblage_6x5x5x2"
    if args.preset_surface_dropout_04:
        return "surface_dense_dropout_0.4_7x7"
    if args.preset_rapide:
        return "quick_smoke_test"
    return None


def resolve_mnist_path(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Répertorier un cache MNIST existant afin d'éviter un téléchargement inutile."""
    if args.mnist_path is not None:
        candidate = args.mnist_path.expanduser()
        if not candidate.is_absolute():
            candidate = env_config.PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            parser.error(f"fichier MNIST introuvable : {candidate}")
        args.mnist_path_resolved = candidate
        args.mnist_sha256 = sha256_file(candidate)
        return

    candidates = (
        env_config.PROJECT_CACHE / "keras" / "datasets" / "mnist.npz",
        Path.home() / ".keras" / "datasets" / "mnist.npz",
    )
    args.mnist_path_resolved = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        None,
    )
    args.mnist_sha256 = (
        sha256_file(args.mnist_path_resolved)
        if args.mnist_path_resolved is not None
        else None
    )


def resolve_cascade_dataset(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Valider le dataset Cascade avant de calculer l'identité du protocole."""
    args.cascade_dir_resolved = None
    args.cascade_x_path = None
    args.cascade_y_path = None
    args.cascade_ids_path = None
    args.cascade_manifest_path = None
    args.cascade_x_sha256 = None
    args.cascade_y_sha256 = None
    args.cascade_ids_sha256 = None
    args.cascade_manifest_sha256 = None
    args.cascade_samples = 0
    args.cascade_class_distribution = {}
    if args.cascade_dir is None:
        return

    import numpy as np

    directory = args.cascade_dir.expanduser()
    if not directory.is_absolute():
        directory = env_config.PROJECT_ROOT / directory
    directory = directory.resolve()
    paths = {
        "x": directory / "x_train_cascade.npy",
        "y": directory / "y_train_cascade.npy",
        "ids": directory / "ids_train_cascade.npy",
        "manifest": directory.parent / "manifest.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(
            "dataset Cascade incomplet ; fichier(s) introuvable(s) : "
            + ", ".join(missing)
        )

    try:
        x = np.load(paths["x"], allow_pickle=False)
        y = np.load(paths["y"], allow_pickle=False)
        ids = np.load(paths["ids"], allow_pickle=False)
    except (OSError, ValueError) as exc:
        parser.error(f"dataset Cascade illisible : {exc}")
    if x.ndim != 4 or x.shape[1:] != (28, 28, 1):
        parser.error(f"forme Cascade x invalide : {x.shape}, attendu (N, 28, 28, 1)")
    if y.shape != (len(x),):
        parser.error(f"forme Cascade y invalide : {y.shape}, attendu ({len(x)},)")
    if ids.shape != (len(x),):
        parser.error(f"forme Cascade ids invalide : {ids.shape}, attendu ({len(x)},)")
    if len(x) == 0:
        parser.error("le dataset Cascade est vide")
    if not np.issubdtype(x.dtype, np.floating):
        parser.error(f"dtype Cascade x invalide : {x.dtype}, attendu flottant")
    if not np.isfinite(x).all() or float(x.min()) < 0 or float(x.max()) > 1:
        parser.error("les pixels Cascade doivent être finis et compris entre 0 et 1")
    if not np.issubdtype(y.dtype, np.integer):
        parser.error(f"dtype Cascade y invalide : {y.dtype}, attendu entier")
    if int(y.min()) < 0 or int(y.max()) > 9:
        parser.error("les étiquettes Cascade doivent être comprises entre 0 et 9")
    id_values = [str(value) for value in ids.tolist()]
    if len(set(id_values)) != len(id_values):
        parser.error("les identifiants Cascade doivent être uniques")

    labels, counts = np.unique(y.astype(int), return_counts=True)
    args.cascade_dir_resolved = directory
    args.cascade_x_path = paths["x"]
    args.cascade_y_path = paths["y"]
    args.cascade_ids_path = paths["ids"]
    args.cascade_manifest_path = paths["manifest"]
    args.cascade_x_sha256 = sha256_file(paths["x"])
    args.cascade_y_sha256 = sha256_file(paths["y"])
    args.cascade_ids_sha256 = sha256_file(paths["ids"])
    args.cascade_manifest_sha256 = sha256_file(paths["manifest"])
    args.cascade_samples = len(x)
    args.cascade_class_distribution = {
        str(int(label)): int(count) for label, count in zip(labels, counts)
    }


def dropout_identifier_token(dropout: float) -> str:
    """Créer un token unique tout en préservant les anciens IDs usuels.

    Les anciens scripts limitaient le texte à huit chiffres significatifs. On
    conserve exactement ce format lorsqu'il représente le float sans perte.
    Une valeur plus fine reçoit un token v2 à 17 chiffres, suffisant pour un
    aller-retour unique de tout float Python fini.
    """
    normalized = 0.0 if dropout == 0 else float(dropout)
    if not math.isfinite(normalized) or not 0 <= normalized < 1:
        raise ValueError("dropout doit être fini et vérifier 0 <= dropout < 1")
    legacy_text = f"{normalized:.8g}"
    if float(legacy_text) == normalized:
        return legacy_text.replace(".", "p")
    return "v2_" + format(normalized, ".17g").replace(".", "p")


def make_run_id(
    filter_1: int,
    filter_2: int,
    dropout: float,
    seed: int,
    conv_activation: str = "relu",
) -> str:
    if seed < 0 or seed > MAX_RANDOM_SEED:
        raise ValueError(f"seed doit être compris entre 0 et {MAX_RANDOM_SEED}")
    dropout_token = dropout_identifier_token(dropout)
    legacy_id = (
        f"f1_{filter_1}__f2_{filter_2}__dropout_{dropout_token}__seed_{seed}"
    )
    if conv_activation == "relu":
        # Conserver les identifiants historiques pour reprendre les anciennes grilles.
        return legacy_id
    return f"activation_{conv_activation}__{legacy_id}"


def planned_runs(
    args: argparse.Namespace,
    seeds: Sequence[int],
    protocol_id: str,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for filter_1 in args.filters_1:
        for filter_2 in args.filters_2:
            for dropout in args.dropouts:
                for conv_activation in args.activations:
                    for repetition, seed in enumerate(seeds, start=1):
                        runs.append(
                            {
                                "protocol_id": protocol_id,
                                "run_id": make_run_id(
                                    filter_1,
                                    filter_2,
                                    dropout,
                                    seed,
                                    conv_activation,
                                ),
                                "conv_activation": conv_activation,
                                "output_activation": OUTPUT_ACTIVATION,
                                "filter_1": filter_1,
                                "filter_2": filter_2,
                                "dropout": dropout,
                                "repetition": repetition,
                                "seed": seed,
                            }
                        )
    run_ids = [str(run["run_id"]) for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(
            "collision interne de run_id : vérifiez les dropouts, activations et graines"
        )
    return runs


def relative_to_experiment(path: Path, paths: ExperimentPaths) -> str:
    return str(path.relative_to(paths.root))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_dataframe_atomic(dataframe: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_experiment_configuration(
    parser: argparse.ArgumentParser,
    paths: ExperimentPaths,
) -> dict[str, Any]:
    try:
        configuration = json.loads(paths.configuration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(
            "configuration.json est illisible alors que des résultats existent : "
            f"{exc}. Utilisez un nouveau --nom-experience."
        )
    if not isinstance(configuration, dict):
        parser.error("configuration.json doit contenir un objet JSON")
    return configuration


def protocol_id_from_configuration(configuration: dict[str, Any]) -> str:
    """Lire l'identité enregistrée, sans la recalculer depuis les options CLI."""
    value = configuration.get("protocol_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    try:
        schema_version = int(configuration.get("schema_version", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("schema_version invalide dans configuration.json") from exc
    if schema_version >= 2:
        raise ValueError(
            "configuration v2 sans protocol_id : le manifeste est incomplet"
        )
    return "legacy-v1"


def validate_raw_protocol_identity(
    parser: argparse.ArgumentParser,
    paths: ExperimentPaths,
    expected_protocol_id: str,
) -> None:
    """Refuser un CSV contenant plusieurs protocoles ou un protocole inattendu."""
    raw_results = load_raw_results(
        paths,
        legacy_protocol_id=expected_protocol_id,
    )
    identifiers = {
        str(value).strip()
        for value in raw_results["protocol_id"].dropna().tolist()
        if str(value).strip()
    }
    if len(identifiers) > 1:
        parser.error(
            "resultats_bruts.csv mélange plusieurs protocol_id : "
            + ", ".join(sorted(identifiers))
        )
    if identifiers and identifiers != {expected_protocol_id}:
        parser.error(
            "le protocol_id du CSV brut ne correspond pas à configuration.json : "
            f"CSV={sorted(identifiers)!r}, configuration={expected_protocol_id!r}"
        )


def validate_existing_experiment(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    paths: ExperimentPaths,
    seeds: Sequence[int],
    protocol_id: str,
) -> dict[str, Any] | None:
    """Empêcher le mélange silencieux de protocoles dans un même CSV brut."""
    raw_exists = paths.raw_csv.exists()
    configuration_exists = paths.configuration.exists()
    if raw_exists and not configuration_exists:
        parser.error(
            "refus de lire ou reprendre resultats_bruts.csv sans configuration.json : "
            "l'origine scientifique des lignes ne peut pas être vérifiée. "
            "Utilisez un nouveau --nom-experience ou effectuez une migration auditée."
        )
    if args.plot_only and not raw_exists:
        parser.error(f"--plot-only nécessite le fichier existant : {paths.raw_csv}")
    if not configuration_exists:
        return None

    previous = read_experiment_configuration(parser, paths)
    try:
        stored_protocol_id = protocol_id_from_configuration(previous)
    except ValueError as exc:
        parser.error(str(exc))

    if args.plot_only:
        validate_raw_protocol_identity(parser, paths, stored_protocol_id)
        return previous
    if not raw_exists:
        return previous

    previous_input = previous.get("input", {})
    previous_grid = previous.get("grid", {})
    previous_training = previous.get("training", {})
    previous_protocol_payload = previous.get("protocol", {})
    previous_protocol_input = previous_protocol_payload.get("input", {})
    previous_implementation = previous.get(
        "implementation",
        previous_protocol_payload.get("implementation", {}),
    )
    previous_protocol_id = previous.get("protocol_id")
    try:
        previous_schema_version = int(previous.get("schema_version", 1))
    except (TypeError, ValueError):
        parser.error("schema_version invalide dans configuration.json")
    expected = {
        "protocol_id": (
            previous_protocol_id if previous_protocol_id is not None else protocol_id,
            protocol_id,
        ),
        "input.train_limit": (previous_input.get("train_limit"), args.train_limit),
        "input.test_limit": (previous_input.get("test_limit"), args.test_limit),
        "input.dataset_seed": (previous_input.get("dataset_seed"), args.dataset_seed),
        "input.dataset_file": (
            previous_input.get("dataset_file"),
            dataset_reference(args),
        ),
        "input.cascade": (
            previous_input.get(
                "cascade",
                previous_protocol_input.get("cascade", {"enabled": False}),
            ),
            cascade_dataset_payload(args),
        ),
        "grid.preset": (
            previous_grid.get("preset"),
            active_preset_name(args),
        ),
        "grid.filter_1": (
            previous_grid.get("filter_1"),
            list(args.filters_1),
        ),
        "grid.filter_2": (
            previous_grid.get("filter_2"),
            list(args.filters_2),
        ),
        "grid.dropout": (
            previous_grid.get("dropout"),
            list(args.dropouts),
        ),
        "grid.conv_activation": (
            previous_grid.get(
                "conv_activation",
                previous_grid.get("activations", ["relu"]),
            ),
            list(args.activations),
        ),
        "grid.output_activation": (
            previous_grid.get("output_activation", OUTPUT_ACTIVATION),
            OUTPUT_ACTIVATION,
        ),
        "grid.seeds": (
            previous_grid.get("seeds"),
            list(seeds),
        ),
        "training.epochs_max": (previous_training.get("epochs_max"), args.epochs),
        "training.early_stopping_patience": (
            previous_training.get("early_stopping_patience"),
            args.patience,
        ),
        "training.batch_size": (previous_training.get("batch_size"), args.batch_size),
        "training.validation_split": (
            previous_training.get("validation_split"),
            args.validation_split,
        ),
        "training.validation_policy": (
            previous_training.get(
                "validation_policy",
                "validation_split Keras sur MNIST",
            ),
            validation_policy(args),
        ),
        "training.learning_rate": (
            previous_training.get("learning_rate"),
            args.learning_rate,
        ),
    }
    if previous_schema_version >= 2 or previous_protocol_id is not None:
        expected.update(
            {
                "input.dataset_sha256": (
                    previous_input.get(
                        "dataset_sha256",
                        previous_protocol_input.get("dataset_sha256"),
                    ),
                    getattr(args, "mnist_sha256", None)
                    or DATASET_HASH_UNAVAILABLE,
                ),
                "implementation.generator_sha256": (
                    previous_implementation.get("generator_sha256"),
                    implementation_sha256(),
                ),
            }
        )
    differences = [
        f"{field}: ancien={old!r}, nouveau={new!r}"
        for field, (old, new) in expected.items()
        if old != new
    ]
    if differences:
        parser.error(
            "le protocole diffère de celui déjà enregistré dans cette expérience. "
            "Utilisez un nouveau --nom-experience pour ne pas mélanger les résultats. "
            + " ; ".join(differences)
        )
    if previous_schema_version < 2 and previous_protocol_id is None:
        print(
            "Avertissement : reprise legacy v1 auditée. Le manifeste historique "
            "ne contenait ni protocol_id ni empreintes SHA-256 ; les autres champs "
            "ont été comparés avant attribution du protocole v2 courant.",
            file=sys.stderr,
        )
    validate_raw_protocol_identity(parser, paths, protocol_id)
    return previous


def write_experiment_documentation(
    paths: ExperimentPaths,
    args: argparse.Namespace,
    display_name: str,
    slug: str,
    seeds: Sequence[int],
    number_of_runs: int,
    protocol_id: str,
) -> None:
    created_at = utc_now()
    previous: dict[str, Any] = {}
    if paths.configuration.exists():
        try:
            previous = json.loads(paths.configuration.read_text(encoding="utf-8"))
            created_at = previous.get("experiment", {}).get(
                "created_at_utc", created_at
            )
        except (OSError, json.JSONDecodeError):
            previous = {}

    current_configuration = {
        "schema_version": 2,
        "protocol_id": protocol_id,
        "protocol": build_protocol_payload(args, seeds),
        "experiment": {
            "display_name": display_name,
            "folder_name": slug,
            "created_at_utc": created_at,
            "last_invocation_at_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
            "mode": "plot_only" if args.plot_only else "training",
        },
        "input": {
            "dataset": dataset_display_name(args),
            "dataset_file": dataset_reference(args),
            "dataset_sha256": getattr(args, "mnist_sha256", None)
            or DATASET_HASH_UNAVAILABLE,
            "cascade": cascade_dataset_payload(args),
            "image_shape": [28, 28, 1],
            "normalization": "float32 divisé par 255",
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "dataset_seed": args.dataset_seed,
        },
        "grid": {
            "preset": active_preset_name(args),
            "filter_1": list(args.filters_1),
            "filter_2": list(args.filters_2),
            "dropout": list(args.dropouts),
            "conv_activation": list(args.activations),
            "output_activation": OUTPUT_ACTIVATION,
            "seeds": list(seeds),
            "runs_planned_for_this_invocation": number_of_runs,
        },
        "training": {
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "validation_policy": validation_policy(args),
            "learning_rate": args.learning_rate,
            "kernel_sizes": [[5, 5], [5, 5]],
            "conv_activation_policy": "même activation pour les deux Conv2D",
            "output_activation": OUTPUT_ACTIVATION,
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy",
        },
        "implementation": {
            "generator_sha256": implementation_sha256(),
            "script": "scripts/experiences/generer_color_map.py",
        },
        "outputs": {
            "raw_results": relative_to_experiment(paths.raw_csv, paths),
            "aggregated_results": relative_to_experiment(paths.aggregated_csv, paths),
            "heatmaps_2d": relative_to_experiment(paths.heatmaps, paths),
            "scatter_3d": relative_to_experiment(paths.scatter_3d, paths),
            "surfaces_3d_detaillees": relative_to_experiment(
                paths.surface_details,
                paths,
            ),
            "analysis_data": relative_to_experiment(paths.analysis_data, paths),
            "correlation_figures": relative_to_experiment(
                paths.correlation_figures,
                paths,
            ),
            "distribution_figures": relative_to_experiment(
                paths.distribution_figures,
                paths,
            ),
            "catalog": relative_to_experiment(paths.catalog, paths),
        },
    }
    if args.plot_only and previous:
        configuration = previous
        # Ne pas prétendre migrer un ancien protocole en v2 : --plot-only ne
        # connaît pas nécessairement la grille d'origine passée sur la ligne
        # de commande. Les CSV sont seulement normalisés en mémoire.
        configuration["schema_version"] = int(
            configuration.get("schema_version", 1)
        )
        configuration.setdefault("experiment", {})
        configuration["experiment"].update(
            {
                "display_name": display_name,
                "folder_name": slug,
                "created_at_utc": created_at,
                "last_invocation_at_utc": utc_now(),
                "command": [sys.executable, *sys.argv],
                "mode": "plot_only",
            }
        )
        configuration["last_plot_only_invocation"] = {
            "at_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
        }
        configuration["outputs"] = current_configuration["outputs"]
    else:
        configuration = current_configuration
    write_json_atomic(paths.configuration, configuration)

    configured_cascade = configuration.get("input", {}).get(
        "cascade",
        configuration.get("protocol", {}).get("input", {}).get(
            "cascade", {"enabled": False}
        ),
    )
    cascade_enabled = bool(configured_cascade.get("enabled", False))
    if int(configuration.get("schema_version", 1)) >= 2:
        if cascade_enabled:
            protocol_notice = (
                "- Le protocole v2 contient les SHA-256 de MNIST, des tableaux "
                "Cascade, du manifeste et du générateur."
            )
        else:
            protocol_notice = (
                "- Le protocole v2 contient les SHA-256 du fichier MNIST local "
                "et du générateur."
            )
    else:
        protocol_notice = (
            "- Cette expérience historique utilise le schéma legacy v1 ; "
            "consulter configuration.json avant de la comparer à une étude récente."
        )

    if cascade_enabled:
        cascade_samples = int(configured_cascade.get("samples", 0))
        cascade_notice = (
            f"- {cascade_samples} chiffres Cascade validés sont ajoutés "
            "uniquement à l'entraînement ; la validation et le test restent MNIST."
        )
    else:
        cascade_notice = "- Aucun complément Cascade n'est utilisé."

    guide = f"""EXPÉRIENCE : {display_name}
DOSSIER : {slug}

ENTRÉE
- MNIST est chargé par Keras, normalisé entre 0 et 1, puis remodelé en 28 x 28 x 1.
{cascade_notice}
- Les paramètres exacts de la dernière invocation sont dans configuration.json.
{protocol_notice}

À OUVRIR EN PREMIER
1. graphiques/surfaces_3d_detaillees/ : résultat principal, lisible et séparé
   par activation. Chaque panneau correspond à une valeur de dropout.
2. graphiques/heatmaps_2d_accuracy_moyenne.png : vue 2D plus simple pour
   comparer précisément les nombres de filtres.
3. Les autres graphiques sont des analyses complémentaires ; ils ne remplacent
   pas les surfaces détaillées.

SORTIES
- donnees/resultats_bruts.csv : une ligne par combinaison et par graine.
- donnees/resultats_agreges.csv : moyenne, écart-type et étendue par combinaison.
- donnees/analyse/ : tables de corrélations et catalogue de l'analyse descriptive.
- graphiques/heatmaps_2d_accuracy_moyenne.png : une heatmap filter_1 x filter_2 par dropout.
- graphiques/surfaces_3d_detaillees/ : l'unique présentation active sous forme
  de surface, avec accuracy en hauteur et en couleur.
- graphiques/nuage_3d_accuracy_moyenne.png : analyse complémentaire où filter_1,
  filter_2 et dropout sont les trois axes ; l'accuracy est uniquement la couleur.
- graphiques/correlations/ : scatter plots, coût/performance et matrice de corrélation.
- graphiques/distributions/ : histogrammes des scores, dispersions et durées.
- catalogue_sorties.csv : inventaire des fichiers produits et de leur rôle.

REPRISE
Relancer la même commande reprend automatiquement le CSV. Seules les lignes uniques dont
status=success et dont les métadonnées/métriques sont finies et cohérentes sont ignorées.
--force les réentraîne ; --plot-only lit le protocol_id enregistré et ne fait que recalculer
l'agrégation et les figures à partir du CSV brut.
"""
    paths.guide.write_text(guide, encoding="utf-8")


def load_raw_results(
    paths: ExperimentPaths,
    legacy_protocol_id: str | None = None,
) -> Any:
    import pandas as pd

    if not paths.raw_csv.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    dataframe = pd.read_csv(paths.raw_csv)
    if "conv_activation" not in dataframe.columns:
        dataframe["conv_activation"] = "relu"
    else:
        dataframe["conv_activation"] = dataframe["conv_activation"].fillna("relu")
    if "output_activation" not in dataframe.columns:
        dataframe["output_activation"] = OUTPUT_ACTIVATION
    else:
        dataframe["output_activation"] = dataframe["output_activation"].fillna(
            OUTPUT_ACTIVATION
        )
    if "protocol_id" not in dataframe.columns:
        dataframe["protocol_id"] = legacy_protocol_id or "legacy-v1"
    elif legacy_protocol_id is not None:
        dataframe["protocol_id"] = dataframe["protocol_id"].fillna(
            legacy_protocol_id
        )
    else:
        dataframe["protocol_id"] = dataframe["protocol_id"].fillna("legacy-v1")
    for column in RAW_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None
    return dataframe[RAW_COLUMNS]


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def integer_value(value: Any) -> int | None:
    parsed = finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def is_coherent_success_row(
    row: Any,
    expected_run: dict[str, Any] | None = None,
) -> bool:
    """Vérifier qu'un succès est suffisamment complet pour être repris."""
    if str(row.get("status", "")).strip() != "success":
        return False
    if not str(row.get("run_id", "")).strip():
        return False

    filter_1 = integer_value(row.get("filter_1"))
    filter_2 = integer_value(row.get("filter_2"))
    seed = integer_value(row.get("seed"))
    dropout = finite_float(row.get("dropout"))
    epochs_completed = integer_value(row.get("epochs_completed"))
    best_epoch = integer_value(row.get("best_epoch"))
    model_parameters = integer_value(row.get("model_parameters"))
    if (
        filter_1 is None
        or filter_1 <= 0
        or filter_2 is None
        or filter_2 <= 0
        or seed is None
        or not 0 <= seed <= MAX_RANDOM_SEED
        or dropout is None
        or not 0 <= dropout < 1
        or epochs_completed is None
        or epochs_completed < 1
        or best_epoch is None
        or not 1 <= best_epoch <= epochs_completed
        or model_parameters is None
        or model_parameters <= 0
    ):
        return False

    epochs_requested = integer_value(row.get("epochs_requested"))
    if epochs_requested is not None and epochs_completed > epochs_requested:
        return False

    metrics = {
        name: finite_float(row.get(name))
        for name in (
            "best_val_accuracy",
            "best_val_loss",
            "test_accuracy",
            "test_loss",
            "duration_seconds",
        )
    }
    if any(value is None for value in metrics.values()):
        return False
    if not (
        0 <= metrics["best_val_accuracy"] <= 1
        and 0 <= metrics["test_accuracy"] <= 1
        and metrics["best_val_loss"] >= 0
        and metrics["test_loss"] >= 0
        and metrics["duration_seconds"] >= 0
    ):
        return False

    if expected_run is None:
        return True
    if str(row.get("run_id")) != str(expected_run["run_id"]):
        return False
    if str(row.get("protocol_id")) != str(expected_run["protocol_id"]):
        return False
    if str(row.get("conv_activation", "relu")) != str(
        expected_run["conv_activation"]
    ):
        return False
    if str(row.get("output_activation", OUTPUT_ACTIVATION)) != str(
        expected_run["output_activation"]
    ):
        return False
    return (
        filter_1 == int(expected_run["filter_1"])
        and filter_2 == int(expected_run["filter_2"])
        and seed == int(expected_run["seed"])
        and dropout == float(expected_run["dropout"])
    )


def resumable_success_ids(
    raw_results: Any,
    runs: Sequence[dict[str, Any]],
    protocol_id: str,
) -> set[str]:
    """Ne reprendre que les succès uniques, finis et cohérents avec le plan."""
    expected_by_id = {str(run["run_id"]): run for run in runs}
    protocol_rows = raw_results[
        raw_results["protocol_id"].astype(str) == str(protocol_id)
    ]
    valid_ids: set[str] = set()
    for run_id, rows in protocol_rows.groupby("run_id", dropna=False):
        token = str(run_id)
        expected = expected_by_id.get(token)
        if expected is None or len(rows) != 1:
            continue
        if is_coherent_success_row(rows.iloc[0], expected):
            valid_ids.add(token)
    return valid_ids


def upsert_raw_result(dataframe: Any, row: dict[str, Any], paths: ExperimentPaths) -> Any:
    import pandas as pd

    if not dataframe.empty:
        same_result = (
            (dataframe["run_id"].astype(str) == str(row["run_id"]))
            & (
                dataframe["protocol_id"].astype(str)
                == str(row["protocol_id"])
            )
        )
        dataframe = dataframe[~same_result]
    dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)
    dataframe = dataframe.sort_values(
        ["protocol_id", "conv_activation", "filter_1", "filter_2", "dropout", "seed"],
        kind="stable",
    ).reset_index(drop=True)
    write_dataframe_atomic(dataframe[RAW_COLUMNS], paths.raw_csv)
    return dataframe[RAW_COLUMNS]


def fixed_subset(images: Any, labels: Any, limit: int | None, seed: int) -> tuple[Any, Any]:
    if limit is None or limit >= len(images):
        return images, labels
    import numpy as np

    generator = np.random.default_rng(seed)
    indices = generator.choice(len(images), size=limit, replace=False)
    return images[indices], labels[indices]


def load_mnist(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    import keras
    import numpy as np

    if args.mnist_path_resolved is not None:
        with np.load(args.mnist_path_resolved) as dataset:
            x_train = dataset["x_train"]
            y_train = dataset["y_train"]
            x_test = dataset["x_test"]
            y_test = dataset["y_test"]
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = fixed_subset(
        x_train, y_train, args.train_limit, args.dataset_seed
    )
    x_test, y_test = fixed_subset(
        x_test, y_test, args.test_limit, args.dataset_seed + 1
    )
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, axis=-1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, axis=-1)
    return x_train, y_train, x_test, y_test


def load_training_data(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any | None, Any | None, Any, Any]:
    """Charger MNIST et, si demandé, ajouter Cascade au train uniquement."""
    import numpy as np

    x_mnist, y_mnist, x_test, y_test = load_mnist(args)
    if args.cascade_dir_resolved is None:
        return x_mnist, y_mnist, None, None, x_test, y_test

    # Reproduire exactement la règle historique de Keras : la validation est
    # la fin du tableau MNIST, prélevée avant tout mélange d'entraînement.
    split_at = int(len(x_mnist) * (1.0 - args.validation_split))
    if split_at <= 0 or split_at >= len(x_mnist):
        raise ValueError("la séparation MNIST entraînement/validation est vide")
    x_train = x_mnist[:split_at]
    y_train = y_mnist[:split_at]
    x_validation = x_mnist[split_at:]
    y_validation = y_mnist[split_at:]

    x_cascade = np.load(args.cascade_x_path, allow_pickle=False).astype(
        "float32", copy=False
    )
    y_cascade = np.load(args.cascade_y_path, allow_pickle=False)
    x_train = np.concatenate((x_train, x_cascade), axis=0)
    y_train = np.concatenate((y_train, y_cascade), axis=0)

    # Mélange fixe des deux sources dans le train ; validation et test ne sont
    # jamais mélangés avec Cascade.
    permutation = np.random.default_rng(args.dataset_seed).permutation(len(x_train))
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    print(
        f"Données : {len(x_train)} entraînement "
        f"({split_at} MNIST + {len(x_cascade)} Cascade), "
        f"{len(x_validation)} validation MNIST, {len(x_test)} test MNIST."
    )
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def build_model(
    keras: Any,
    filter_1: int,
    filter_2: int,
    dropout: float,
    conv_activation: str,
    args: argparse.Namespace,
) -> Any:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(
                filter_1,
                kernel_size=(5, 5),
                activation=conv_activation,
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
                filter_2,
                kernel_size=(5, 5),
                activation=conv_activation,
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(10, activation=OUTPUT_ACTIVATION),
        ]
    )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def result_row_base(
    args: argparse.Namespace,
    display_name: str,
    run: dict[str, Any],
    training_samples: int,
    validation_samples: int,
    test_samples: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    return {
        "experiment_name": display_name,
        **run,
        "repetitions_requested": len(seeds),
        "epochs_requested": args.epochs,
        "epochs_completed": None,
        "best_epoch": None,
        "batch_size": args.batch_size,
        "validation_split": args.validation_split,
        "learning_rate": args.learning_rate,
        "train_samples": training_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "model_parameters": None,
        "best_val_accuracy": None,
        "best_val_loss": None,
        "test_accuracy": None,
        "test_loss": None,
        "duration_seconds": None,
        "status": "error",
        "error_message": None,
        "completed_at_utc": None,
    }


def train_grid(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    display_name: str,
    runs: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    protocol_id: str,
) -> Any:
    raw_results = load_raw_results(paths, legacy_protocol_id=protocol_id)
    successful_ids = resumable_success_ids(raw_results, runs, protocol_id)
    declared_successes = int(
        (
            (raw_results["status"] == "success")
            & (raw_results["protocol_id"].astype(str) == protocol_id)
            & (raw_results["run_id"].astype(str).isin([run["run_id"] for run in runs]))
        ).sum()
    )
    invalid_successes = max(0, declared_successes - len(successful_ids))
    if invalid_successes:
        print(
            f"Reprise : {invalid_successes} succès incomplet(s), non fini(s), "
            "dupliqué(s) ou incohérent(s) seront recalculés.",
            file=sys.stderr,
        )
    pending_runs = [
        run for run in runs if args.force or run["run_id"] not in successful_ids
    ]

    print(
        f"Expérience '{display_name}' : {len(runs)} essais demandés, "
        f"{len(runs) - len(pending_runs)} déjà terminés, {len(pending_runs)} à exécuter."
    )
    if not pending_runs:
        return raw_results

    import keras
    import numpy as np

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = load_training_data(args)
    if x_validation is None:
        training_samples = int(len(x_train) * (1.0 - args.validation_split))
        validation_samples = len(x_train) - training_samples
        validation_arguments = {"validation_split": args.validation_split}
    else:
        training_samples = len(x_train)
        validation_samples = len(x_validation)
        validation_arguments = {
            "validation_data": (x_validation, y_validation),
        }
    total_pending = len(pending_runs)

    for index, run in enumerate(pending_runs, start=1):
        print(
            f"\n[{index}/{total_pending}] filter_1={run['filter_1']}, "
            f"filter_2={run['filter_2']}, dropout={run['dropout']}, "
            f"activation={run['conv_activation']}, seed={run['seed']}"
        )
        row = result_row_base(
            args,
            display_name,
            run,
            training_samples,
            validation_samples,
            len(x_test),
            seeds,
        )
        started = time.perf_counter()
        try:
            keras.backend.clear_session()
            keras.utils.set_random_seed(run["seed"])
            model = build_model(
                keras,
                run["filter_1"],
                run["filter_2"],
                run["dropout"],
                run["conv_activation"],
                args,
            )
            # La taille reste informative même si model.fit échoue ensuite
            # (par exemple par manque de mémoire).
            row["model_parameters"] = model.count_params()
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=True,
                )
            ]
            history = model.fit(
                x_train,
                y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=args.verbose,
                shuffle=True,
                **validation_arguments,
            )
            evaluation = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
            val_losses = np.asarray(history.history["val_loss"], dtype=float)
            val_accuracies = np.asarray(history.history["val_accuracy"], dtype=float)
            best_index = int(np.argmin(val_losses))
            row.update(
                {
                    "epochs_completed": len(history.history["loss"]),
                    "best_epoch": best_index + 1,
                    "model_parameters": model.count_params(),
                    "best_val_accuracy": float(val_accuracies[best_index]),
                    "best_val_loss": float(val_losses[best_index]),
                    "test_accuracy": float(evaluation["accuracy"]),
                    "test_loss": float(evaluation["loss"]),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "status": "success",
                    "error_message": None,
                    "completed_at_utc": utc_now(),
                }
            )
            if not is_coherent_success_row(row, run):
                raise ValueError(
                    "l'entraînement a renvoyé des métriques non finies ou incohérentes"
                )
            print(f"Accuracy test : {row['test_accuracy']:.4f}")
        except Exception as exc:
            row.update(
                {
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "completed_at_utc": utc_now(),
                }
            )
            raw_results = upsert_raw_result(raw_results, row, paths)
            print(
                f"Échec enregistré dans {paths.raw_csv}: {row['error_message']}",
                file=sys.stderr,
            )
            keras.backend.clear_session()
            if args.arreter_sur_erreur:
                raise
            continue

        raw_results = upsert_raw_result(raw_results, row, paths)

    return raw_results


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_critical_95(sample_size: int) -> float:
    """Quantile bilatéral 95 % de Student, sans dépendance SciPy obligatoire."""
    if sample_size < 2:
        return math.nan
    degrees_of_freedom = sample_size - 1
    if degrees_of_freedom <= 30:
        return T_CRITICAL_95[degrees_of_freedom]
    if degrees_of_freedom <= 40:
        return 2.042
    if degrees_of_freedom <= 60:
        return 2.021
    if degrees_of_freedom <= 120:
        return 2.000
    return 1.980


def aggregate_results(raw_results: Any, paths: ExperimentPaths) -> Any:
    import pandas as pd

    if raw_results.empty:
        raise ValueError("le fichier de résultats bruts est vide")

    numeric_columns = [
        "filter_1",
        "filter_2",
        "dropout",
        "repetitions_requested",
        "epochs_completed",
        "model_parameters",
        "best_val_accuracy",
        "best_val_loss",
        "test_accuracy",
        "test_loss",
        "duration_seconds",
    ]
    working = raw_results.copy()
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working["conv_activation"] = working["conv_activation"].fillna("relu")
    working["output_activation"] = working["output_activation"].fillna(
        OUTPUT_ACTIVATION
    )
    working["protocol_id"] = working["protocol_id"].fillna("legacy-v1")
    protocol_ids = {
        str(value).strip()
        for value in working["protocol_id"].tolist()
        if str(value).strip()
    }
    has_blank_protocol = bool(
        (working["protocol_id"].astype(str).str.strip() == "").any()
    )
    if len(protocol_ids) != 1 or has_blank_protocol:
        raise ValueError(
            "une expérience doit contenir exactement un protocol_id non vide ; "
            f"valeurs trouvées : {sorted(protocol_ids)!r}"
        )
    working["_coherent_success"] = working.apply(
        is_coherent_success_row,
        axis=1,
    )
    group_columns = [
        "protocol_id",
        "conv_activation",
        "output_activation",
        "filter_1",
        "filter_2",
        "dropout",
    ]
    attempts = (
        working.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            runs_attempted=("run_id", "nunique"),
            runs_failed=(
                "_coherent_success",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            runs_expected=("repetitions_requested", "max"),
            model_parameters_attempted=("model_parameters", "max"),
            attempt_duration_seconds_mean=("duration_seconds", "mean"),
            attempt_duration_seconds_std=("duration_seconds", "std"),
        )
    )
    successful = working[working["_coherent_success"]].dropna(
        subset=["filter_1", "filter_2", "dropout", "test_accuracy"]
    )
    metric_columns = [
        "runs_completed",
        "model_parameters",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_accuracy_min",
        "test_accuracy_max",
        "test_loss_mean",
        "test_loss_std",
        "best_val_accuracy_mean",
        "best_val_accuracy_std",
        "best_val_loss_mean",
        "best_val_loss_std",
        "duration_seconds_mean",
        "duration_seconds_std",
        "epochs_completed_mean",
        "epochs_completed_std",
    ]
    if successful.empty:
        successful_summary = pd.DataFrame(columns=[*group_columns, *metric_columns])
    else:
        successful_summary = successful.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        ).agg(
            runs_completed=("run_id", "nunique"),
            model_parameters=("model_parameters", "max"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_accuracy_min=("test_accuracy", "min"),
            test_accuracy_max=("test_accuracy", "max"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            best_val_accuracy_mean=("best_val_accuracy", "mean"),
            best_val_accuracy_std=("best_val_accuracy", "std"),
            best_val_loss_mean=("best_val_loss", "mean"),
            best_val_loss_std=("best_val_loss", "std"),
            duration_seconds_mean=("duration_seconds", "mean"),
            duration_seconds_std=("duration_seconds", "std"),
            epochs_completed_mean=("epochs_completed", "mean"),
            epochs_completed_std=("epochs_completed", "std"),
        )

    # La table des tentatives est la table de gauche : une configuration ayant
    # uniquement échoué (par exemple OOM à 128 filtres) reste donc visible.
    aggregated = attempts.merge(
        successful_summary,
        on=group_columns,
        how="left",
    )
    aggregated["model_parameters"] = aggregated["model_parameters"].fillna(
        aggregated["model_parameters_attempted"]
    )
    aggregated = aggregated.drop(columns=["model_parameters_attempted"])
    aggregated["runs_completed"] = (
        pd.to_numeric(aggregated["runs_completed"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    aggregated["runs_failed"] = (
        pd.to_numeric(aggregated["runs_failed"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    aggregated["runs_attempted"] = (
        pd.to_numeric(aggregated["runs_attempted"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    aggregated["runs_expected"] = (
        pd.to_numeric(aggregated["runs_expected"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    aggregated["test_accuracy_sem"] = aggregated["test_accuracy_std"] / (
        aggregated["runs_completed"].where(aggregated["runs_completed"] > 0) ** 0.5
    )
    critical_values = aggregated["runs_completed"].map(t_critical_95)
    half_width = critical_values * aggregated["test_accuracy_sem"]
    aggregated["test_accuracy_ci95_low"] = (
        aggregated["test_accuracy_mean"] - half_width
    ).clip(lower=0.0, upper=1.0)
    aggregated["test_accuracy_ci95_high"] = (
        aggregated["test_accuracy_mean"] + half_width
    ).clip(lower=0.0, upper=1.0)
    for metric_name, lower_bound, upper_bound in (
        ("test_loss", 0.0, None),
        ("duration_seconds", 0.0, None),
        ("epochs_completed", 0.0, None),
    ):
        standard_deviation_column = f"{metric_name}_std"
        mean_column = f"{metric_name}_mean"
        sem_column = f"{metric_name}_sem"
        low_column = f"{metric_name}_ci95_low"
        high_column = f"{metric_name}_ci95_high"
        aggregated[sem_column] = aggregated[standard_deviation_column] / (
            aggregated["runs_completed"].where(
                aggregated["runs_completed"] > 0
            )
            ** 0.5
        )
        metric_half_width = critical_values * aggregated[sem_column]
        aggregated[low_column] = (
            aggregated[mean_column] - metric_half_width
        ).clip(lower=lower_bound, upper=upper_bound)
        aggregated[high_column] = (
            aggregated[mean_column] + metric_half_width
        ).clip(lower=lower_bound, upper=upper_bound)
    aggregated["attempts_complete"] = (
        aggregated["runs_attempted"] >= aggregated["runs_expected"]
    )
    aggregated["is_complete"] = (
        aggregated["runs_completed"] >= aggregated["runs_expected"]
    )
    aggregated = aggregated.sort_values(group_columns).reset_index(drop=True)
    write_dataframe_atomic(aggregated, paths.aggregated_csv)
    return aggregated


def finite_accuracy_range(aggregated: Any) -> tuple[float, float]:
    import numpy as np

    values = aggregated["test_accuracy_mean"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("aucune accuracy numérique à représenter")
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        padding = max(0.001, abs(minimum) * 0.002)
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.04
    return minimum - padding, maximum + padding


def completion_exit_code(aggregated: Any) -> int:
    """Retourner une erreur explicite lorsque toutes les tentatives ont échoué."""
    completed = 0
    if "runs_completed" in aggregated.columns:
        for value in aggregated["runs_completed"].tolist():
            parsed = finite_float(value)
            if parsed is not None and parsed > 0:
                completed += int(parsed)
    return 0 if completed > 0 else NO_SUCCESS_EXIT_CODE


def activation_dropout_panels(aggregated: Any) -> list[tuple[str, float]]:
    panels = {
        (str(row.conv_activation), float(row.dropout))
        for row in aggregated[["conv_activation", "dropout"]]
        .dropna(subset=["dropout"])
        .itertuples(index=False)
    }
    return sorted(panels, key=lambda item: (item[0], item[1]))


def plot_heatmaps(aggregated: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    panels = activation_dropout_panels(aggregated)
    if not panels:
        raise ValueError("aucune activation ni valeur de dropout à représenter")

    column_count = min(3, len(panels))
    row_count = math.ceil(len(panels) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 4.3 * row_count),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes_flat = list(axes.flat)
    vmin, vmax = finite_accuracy_range(aggregated)
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("#eceff1")
    last_image = None

    for axis, (activation, dropout) in zip(axes_flat, panels):
        subset = aggregated[
            (aggregated["conv_activation"].astype(str) == activation)
            & (aggregated["dropout"] == dropout)
        ]
        grid = subset.pivot_table(
            index="filter_1",
            columns="filter_2",
            values="test_accuracy_mean",
            aggfunc="mean",
            dropna=False,
        ).sort_index(ascending=True).sort_index(axis=1, ascending=True)
        failures = subset.pivot_table(
            index="filter_1",
            columns="filter_2",
            values="runs_failed",
            aggfunc="sum",
            dropna=False,
        ).reindex(index=grid.index, columns=grid.columns)
        matrix = grid.to_numpy(dtype=float)
        last_image = axis.imshow(
            np.ma.masked_invalid(matrix),
            origin="lower",
            aspect="auto",
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_xticks(range(len(grid.columns)), labels=[int(v) for v in grid.columns])
        axis.set_yticks(range(len(grid.index)), labels=[int(v) for v in grid.index])
        axis.set_xlabel("Filtres de la 2e convolution (filter_2)")
        axis.set_ylabel("Filtres de la 1re convolution (filter_1)")
        axis.set_title(f"Activation = {activation} | dropout = {dropout:g}")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    normalized = (value - vmin) / (vmax - vmin)
                    text_color = "white" if normalized < 0.43 else "#17212b"
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2%}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                    )
                elif failures.iloc[row_index, column_index] > 0:
                    axis.text(
                        column_index,
                        row_index,
                        "échec",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#8b1e2d",
                    )

    for axis in axes_flat[len(panels) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Accuracy moyenne selon les filtres, le dropout et l'activation",
        fontsize=15,
        y=1.01,
    )
    if last_image is not None:
        color_axis = figure.add_axes([0.915, 0.20, 0.014, 0.62])
        color_bar = figure.colorbar(last_image, cax=color_axis)
        color_bar.set_label("Accuracy moyenne sur le jeu de test")
    figure.subplots_adjust(top=0.90, right=0.875, hspace=0.35, wspace=0.30)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_scatter_3d(aggregated: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    # Les trois hyperparametres sont des niveaux discrets. Les placer sur des
    # positions regulieres rend la grille beaucoup plus lisible que leurs
    # valeurs numeriques brutes (4, 8, 16 ou 8, 16, 32).
    filter_1_levels = sorted(aggregated["filter_1"].dropna().unique())
    filter_2_levels = sorted(aggregated["filter_2"].dropna().unique())
    dropout_levels = sorted(aggregated["dropout"].dropna().unique())
    if not filter_1_levels or not filter_2_levels or not dropout_levels:
        raise ValueError("les trois axes 3D doivent contenir au moins une valeur")

    filter_1_positions = {
        value: index for index, value in enumerate(filter_1_levels)
    }
    filter_2_positions = {
        value: index for index, value in enumerate(filter_2_levels)
    }
    dropout_positions = {
        value: index for index, value in enumerate(dropout_levels)
    }
    x_values = np.asarray(
        [filter_1_positions[value] for value in aggregated["filter_1"]],
        dtype=float,
    )
    y_values = np.asarray(
        [filter_2_positions[value] for value in aggregated["filter_2"]],
        dtype=float,
    )
    z_values = np.asarray(
        [dropout_positions[value] for value in aggregated["dropout"]],
        dtype=float,
    )
    accuracies = aggregated["test_accuracy_mean"].to_numpy(dtype=float)
    accuracies_valides = accuracies[np.isfinite(accuracies)]
    if accuracies_valides.size == 0:
        raise ValueError("aucune accuracy moyenne valide n'est disponible")
    accuracy_min_observee = float(np.min(accuracies_valides))
    accuracy_max_observee = float(np.max(accuracies_valides))
    vmin, vmax = finite_accuracy_range(aggregated)
    normalisation = Normalize(vmin=vmin, vmax=vmax)
    color_map = plt.get_cmap("cividis")
    activations = sorted(aggregated["conv_activation"].fillna("relu").unique())
    available_markers = ("o", "^", "s", "D", "P", "X")
    activation_markers = {
        activation: available_markers[index % len(available_markers)]
        for index, activation in enumerate(activations)
    }
    nombre_meilleurs = 5 if len(aggregated) >= 20 else min(3, len(aggregated))
    meilleurs_indices = list(
        aggregated["test_accuracy_mean"]
        .nlargest(nombre_meilleurs)
        .index
    )

    figure = plt.figure(figsize=(16, 8.8), facecolor="white")
    axes = [
        figure.add_subplot(121, projection="3d"),
        figure.add_subplot(122, projection="3d"),
    ]
    vues = (
        (27, -55, "Vue A — filtres 1 au premier plan"),
        (27, 125, "Vue B — angle opposé"),
    )
    scatter = None

    for axis, (elevation, azimut, titre_vue) in zip(axes, vues):
        for activation in activations:
            activation_mask = (
                aggregated["conv_activation"].fillna("relu").astype(str).to_numpy()
                == activation
            )
            scatter = axis.scatter(
                x_values[activation_mask],
                y_values[activation_mask],
                z_values[activation_mask],
                c=accuracies[activation_mask],
                cmap=color_map,
                norm=normalisation,
                s=105,
                marker=activation_markers[activation],
                edgecolors="#1f2933",
                linewidths=0.65,
                alpha=0.88,
                depthshade=False,
                label=f"activation {activation}",
            )
        axis.scatter(
            x_values[meilleurs_indices],
            y_values[meilleurs_indices],
            z_values[meilleurs_indices],
            c=accuracies[meilleurs_indices],
            cmap=color_map,
            norm=normalisation,
            s=260,
            marker="*",
            edgecolors="#111820",
            linewidths=1.05,
            depthshade=False,
        )

        for rang, index in enumerate(meilleurs_indices, start=1):
            axis.text(
                x_values[index],
                y_values[index],
                z_values[index] + 0.13,
                f"#{rang}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#111820",
            )

        axis.set_xlabel("Filtres convolution 1", labelpad=9)
        axis.set_ylabel("Filtres convolution 2", labelpad=9)
        axis.set_zlabel("Dropout", labelpad=7)
        axis.set_xticks(
            range(len(filter_1_levels)),
            labels=[str(int(value)) for value in filter_1_levels],
        )
        axis.set_yticks(
            range(len(filter_2_levels)),
            labels=[str(int(value)) for value in filter_2_levels],
        )
        axis.set_zticks(
            range(len(dropout_levels)),
            labels=[f"{value:g}" for value in dropout_levels],
        )
        axis.tick_params(labelsize=9, pad=1)
        axis.set_title(titre_vue, fontsize=12, pad=12)
        activation_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=activation_markers[activation],
                markerfacecolor="#7a8793",
                markeredgecolor="#1f2933",
                markersize=7,
                label=activation,
            )
            for activation in activations
        ]
        axis.legend(
            handles=activation_handles,
            title="Activation Conv2D",
            loc="upper left",
            fontsize=8,
            title_fontsize=8,
            frameon=True,
        )
        axis.set_proj_type("ortho")
        axis.view_init(elev=elevation, azim=azimut)
        axis.set_box_aspect((1.0, 1.0, 1.25))
        axis.grid(True, color="#d9dee3", linewidth=0.65)
        for axe_cartesien in (axis.xaxis, axis.yaxis, axis.zaxis):
            axe_cartesien.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axe_cartesien.pane.set_edgecolor("#d9dee3")

    figure.suptitle(
        "Cartographie 3D des hyperparamètres et de l'accuracy moyenne",
        fontsize=17,
        y=0.98,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"{len(aggregated)} configurations | deux angles complémentaires | "
            f"accuracy observée : {accuracy_min_observee:.2%} à "
            f"{accuracy_max_observee:.2%} | "
            f"étoiles = top {nombre_meilleurs}"
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#45525e",
    )

    # Un axe explicite empêche la barre de couleur de recouvrir la vue de droite.
    axe_barre_couleur = figure.add_axes([0.91, 0.27, 0.016, 0.50])
    color_bar = figure.colorbar(scatter, cax=axe_barre_couleur)
    color_bar.set_label("Accuracy moyenne sur le jeu de test", labelpad=10)
    color_bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

    lignes_classement = []
    for rang, index in enumerate(meilleurs_indices, start=1):
        ligne = aggregated.loc[index]
        ecart_type = ligne.get("test_accuracy_std")
        dispersion = (
            "écart-type n.d."
            if ecart_type is None or not np.isfinite(float(ecart_type))
            else f"± {float(ecart_type):.2%}"
        )
        lignes_classement.append(
            f"#{rang}  f1={int(ligne['filter_1'])}, "
            f"f2={int(ligne['filter_2'])}, dropout={ligne['dropout']:g}, "
            f"activation={ligne['conv_activation']}  "
            f"→ {ligne['test_accuracy_mean']:.2%} {dispersion}"
        )
    classement_multiligne = "\n".join(
        "   |   ".join(lignes_classement[index : index + 2])
        for index in range(0, len(lignes_classement), 2)
    )
    figure.text(
        0.5,
        0.025,
        "Meilleures configurations\n" + classement_multiligne,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#1f2933",
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.035,
        right=0.865,
        top=0.88,
        bottom=0.23,
        wspace=0.03,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def interpolate_response_surface(
    filter_1_levels: Any,
    filter_2_levels: Any,
    matrix: Any,
) -> tuple[Any, Any, Any]:
    """Relier une grille complète par interpolation bilinéaire sans extrapoler."""
    import numpy as np

    x_levels = np.asarray(filter_1_levels, dtype=float)
    y_levels = np.asarray(filter_2_levels, dtype=float)
    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(x_levels), len(y_levels)):
        raise ValueError("la matrice de surface ne correspond pas aux deux axes")
    valid = np.isfinite(values)
    if len(x_levels) < 2 or len(y_levels) < 2 or not bool(valid.all()):
        raise ValueError(
            "une surface exige une grille complète et au moins deux valeurs par axe"
        )

    x_dense = np.linspace(x_levels.min(), x_levels.max(), 61)
    y_dense = np.linspace(y_levels.min(), y_levels.max(), 61)
    interpolated_y = np.vstack(
        [np.interp(y_dense, y_levels, row) for row in values]
    )
    z_dense = np.empty((len(x_dense), len(y_dense)), dtype=float)
    for column_index in range(len(y_dense)):
        z_dense[:, column_index] = np.interp(
            x_dense,
            x_levels,
            interpolated_y[:, column_index],
        )
    x_mesh, y_mesh = np.meshgrid(x_dense, y_dense, indexing="ij")
    return x_mesh, y_mesh, z_dense


def _render_surfaces_3d(
    aggregated: Any,
    output_path: Path,
    show: bool,
    panels: list[tuple[str, float]],
    *,
    detail_activation: str | None = None,
) -> None:
    """Rendre un atlas 3D sans modifier les données ni l'interpolation."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    if not panels:
        raise ValueError("aucune activation ni valeur de dropout à représenter")

    column_count = min(3, len(panels))
    row_count = math.ceil(len(panels) / column_count)
    single_row = row_count == 1
    detail_mode = detail_activation is not None
    figure = plt.figure(
        figsize=(
            (6.3 if detail_mode else 5.9) * column_count + 1.0,
            (6.2 if detail_mode else 5.5) * row_count
            + (2.8 if single_row else 1.8),
        ),
        facecolor="white",
    )
    axes = []
    for index in range(row_count * column_count):
        if index < len(panels):
            axis = figure.add_subplot(
                row_count,
                column_count,
                index + 1,
                projection="3d",
            )
        else:
            axis = figure.add_subplot(row_count, column_count, index + 1)
        axes.append(axis)

    vmin, vmax = finite_accuracy_range(aggregated)
    normalisation = Normalize(vmin=vmin, vmax=vmax)
    color_map = plt.get_cmap("viridis")
    mean_values = aggregated["test_accuracy_mean"].to_numpy(dtype=float)
    standard_deviations = aggregated["test_accuracy_std"].to_numpy(dtype=float)
    has_uncertainty = bool(np.isfinite(standard_deviations).any())
    finite_means = np.isfinite(mean_values)
    safe_deviations = np.where(
        np.isfinite(standard_deviations),
        standard_deviations,
        0.0,
    )
    lower_values = mean_values[finite_means] - safe_deviations[finite_means]
    upper_values = mean_values[finite_means] + safe_deviations[finite_means]
    uncertainty_range = float(upper_values.max() - lower_values.min())
    uncertainty_padding = max(uncertainty_range * 0.04, 1e-4)
    z_axis_min = float(lower_values.min() - uncertainty_padding)
    z_axis_max = float(upper_values.max() + uncertainty_padding)
    z_offset = max((z_axis_max - z_axis_min) * 0.008, 1e-5)
    observed_count = 0

    for axis, (activation, dropout) in zip(axes, panels):
        subset = aggregated[
            (aggregated["conv_activation"].astype(str) == activation)
            & (aggregated["dropout"] == dropout)
        ]
        grid = subset.pivot_table(
            index="filter_1",
            columns="filter_2",
            values="test_accuracy_mean",
            aggfunc="mean",
            dropna=False,
        ).sort_index(ascending=True).sort_index(axis=1, ascending=True)
        filter_1_levels = grid.index.to_numpy(dtype=float)
        filter_2_levels = grid.columns.to_numpy(dtype=float)
        matrix = grid.to_numpy(dtype=float)
        standard_deviation_grid = subset.pivot_table(
            index="filter_1",
            columns="filter_2",
            values="test_accuracy_std",
            aggfunc="mean",
            dropna=False,
        ).reindex(index=grid.index, columns=grid.columns)
        standard_deviation_matrix = standard_deviation_grid.to_numpy(dtype=float)
        valid = np.isfinite(matrix)
        observed_count += int(valid.sum())
        filter_1_positions = np.arange(len(filter_1_levels), dtype=float)
        filter_2_positions = np.arange(len(filter_2_levels), dtype=float)
        observed_x, observed_y = np.meshgrid(
            filter_1_positions,
            filter_2_positions,
            indexing="ij",
        )

        surface_available = True
        try:
            surface_x, surface_y, surface_z = interpolate_response_surface(
                filter_1_positions,
                filter_2_positions,
                matrix,
            )
        except ValueError:
            surface_available = False
        if surface_available:
            axis.plot_surface(
                surface_x,
                surface_y,
                surface_z,
                facecolors=color_map(normalisation(surface_z)),
                rcount=surface_z.shape[0],
                ccount=surface_z.shape[1],
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=0.92,
            )

        for row_index, column_index in zip(*np.where(valid)):
            standard_deviation = standard_deviation_matrix[row_index, column_index]
            if not np.isfinite(standard_deviation):
                continue
            mean_accuracy = matrix[row_index, column_index]
            axis.plot(
                [filter_1_positions[row_index]] * 2,
                [filter_2_positions[column_index]] * 2,
                [
                    mean_accuracy - standard_deviation,
                    mean_accuracy + standard_deviation,
                ],
                color="#26323c",
                linewidth=0.8,
                alpha=0.72,
                zorder=4,
            )

        axis.scatter(
            observed_x[valid],
            observed_y[valid],
            matrix[valid] + z_offset,
            s=34,
            marker="o",
            facecolors="white",
            edgecolors="#17212b",
            linewidths=0.85,
            depthshade=False,
            zorder=5,
        )

        if bool(valid.any()):
            best_flat_index = int(np.nanargmax(matrix))
            best_row, best_column = np.unravel_index(best_flat_index, matrix.shape)
            best_filter_1 = filter_1_levels[best_row]
            best_filter_2 = filter_2_levels[best_column]
            best_accuracy = float(matrix[best_row, best_column])
            axis.scatter(
                [filter_1_positions[best_row]],
                [filter_2_positions[best_column]],
                [best_accuracy + 2.0 * z_offset],
                s=155,
                marker="*",
                facecolors="#f4c542",
                edgecolors="#111820",
                linewidths=1.0,
                depthshade=False,
                zorder=6,
            )
            panel_result = (
                f"max : {best_accuracy:.2%} "
                f"(f1={int(best_filter_1)}, f2={int(best_filter_2)})"
            )
        else:
            panel_result = "aucun entraînement réussi"

        activation_label = (
            "ReLU" if activation == "relu" else activation.capitalize()
        )
        panel_heading = (
            f"Dropout = {dropout:g}"
            if detail_mode
            else f"{activation_label} — dropout = {dropout:g}"
        )
        axis.set_title(
            panel_heading + "\n" + panel_result,
            fontsize=12.2 if detail_mode else 11.5,
            fontweight="semibold",
            color="#17212b",
            pad=10,
        )
        axis.set_xlabel("Filtres conv. 1", fontsize=10.5, labelpad=6)
        axis.set_ylabel("Filtres conv. 2", fontsize=10.5, labelpad=6)
        axis.set_zlabel("Accuracy", fontsize=10.5, labelpad=7)
        axis.set_xticks(
            filter_1_positions,
            labels=[int(v) for v in filter_1_levels],
        )
        axis.set_yticks(
            filter_2_positions,
            labels=[int(v) for v in filter_2_levels],
        )
        axis.set_zlim(z_axis_min, z_axis_max + 2.5 * z_offset)
        axis.zaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.zaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        axis.tick_params(labelsize=9.5, pad=1)
        axis.view_init(elev=28, azim=-55)
        axis.set_box_aspect((1.0, 1.0, 0.85))
        axis.grid(True, color="#d9dee3", linewidth=0.6)
        for cartesian_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            cartesian_axis.pane.set_facecolor((0.98, 0.98, 0.98, 0.75))
            cartesian_axis.pane.set_edgecolor("#d9dee3")

    unused_axes = axes[len(panels) :]
    for axis in unused_axes:
        axis.set_axis_off()
    if unused_axes:
        uncertainty_explanation = (
            "• Les traits verticaux montrent ±1 écart-type.\n"
            if has_uncertainty
            else "• Écart-type indisponible avec une seule répétition.\n"
        )
        relevant_rows = aggregated
        best_scope = "global"
        if detail_mode:
            relevant_rows = aggregated[
                aggregated["conv_activation"].astype(str) == detail_activation
            ]
            best_scope = f"pour l'activation {detail_activation}"
        global_best_index = int(relevant_rows["test_accuracy_mean"].idxmax())
        global_best = relevant_rows.loc[global_best_index]
        unused_axes[0].text(
            0.08,
            0.86,
            "Comment lire la figure",
            transform=unused_axes[0].transAxes,
            fontsize=14,
            fontweight="bold",
            color="#1f2933",
            va="top",
        )
        unused_axes[0].text(
            0.08,
            0.68,
            (
                "• La surface relie linéairement les points.\n"
                "• Les pastilles sont les essais réels.\n"
                + uncertainty_explanation
                + "• L'étoile marque le meilleur point local."
            ),
            transform=unused_axes[0].transAxes,
            fontsize=11.5,
            color="#34414c",
            linespacing=1.55,
            va="top",
        )
        unused_axes[0].text(
            0.08,
            0.31,
            (
                f"Meilleur résultat {best_scope}\n"
                f"f1={int(global_best['filter_1'])}, "
                f"f2={int(global_best['filter_2'])}, "
                f"dropout={global_best['dropout']:g}, "
                f"activation={global_best['conv_activation']}\n"
                f"accuracy = {global_best['test_accuracy_mean']:.2%}"
            ),
            transform=unused_axes[0].transAxes,
            fontsize=12,
            fontweight="bold",
            color="#1f2933",
            linespacing=1.4,
            va="top",
        )
        if len(unused_axes) > 1:
            unused_axes[1].text(
                0.08,
                0.78,
                "Vues agrandies",
                transform=unused_axes[1].transAxes,
                fontsize=14,
                fontweight="bold",
                color="#1f2933",
                va="top",
            )
            unused_axes[1].text(
                0.08,
                0.60,
                (
                    "Une image séparée par activation est enregistrée dans\n"
                    "graphiques/surfaces_3d_detaillees/.\n\n"
                    "Ces vues 3 × 2 sont prévues pour la lecture à l'écran."
                ),
                transform=unused_axes[1].transAxes,
                fontsize=11.5,
                color="#34414c",
                linespacing=1.55,
                va="top",
            )

    activation_title = ""
    if detail_mode:
        activation_label = (
            "ReLU"
            if detail_activation == "relu"
            else str(detail_activation).capitalize()
        )
        activation_title = f" — activation {activation_label}"
    figure.suptitle(
        "Surfaces 3D de l'accuracy moyenne" + activation_title,
        fontsize=19,
        fontweight="semibold",
        color="#17212b",
        y=0.985,
    )
    figure.text(
        0.47,
        0.925 if single_row else 0.947,
        (
            "filter_1 et filter_2 au sol | accuracy en hauteur et en couleur | "
            f"axe Z commun et focalisé : {vmin:.2%} à {vmax:.2%}"
        ),
        ha="center",
        va="center",
        fontsize=11.5,
        color="#34414c",
    )
    figure.text(
        0.47,
        0.875 if single_row else 0.918,
        (
            f"{observed_count} configurations mesurées | surface = interpolation "
            "linéaire sans extrapolation | axes filtres espacés par niveaux testés"
        ),
        ha="center",
        va="center",
        fontsize=10.8,
        color="#45525e",
    )

    color_reference = cm.ScalarMappable(norm=normalisation, cmap=color_map)
    color_reference.set_array([])
    color_axis = figure.add_axes([0.915, 0.25, 0.02, 0.52])
    color_bar = figure.colorbar(color_reference, cax=color_axis)
    color_bar.set_label(
        "Accuracy moyenne sur le jeu de test",
        fontsize=11,
        labelpad=11,
    )
    color_bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    color_bar.ax.tick_params(labelsize=9.5)

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#17212b",
            markersize=6,
            label="configuration réellement mesurée",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#f4c542",
            markeredgecolor="#111820",
            markersize=11,
            label="meilleure configuration du dropout",
        ),
    ]
    if has_uncertainty:
        legend_items.append(
            Line2D(
                [0],
                [0],
                color="#26323c",
                linewidth=1.0,
                label="±1 écart-type",
            )
        )
    figure.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.47, 0.012),
        ncol=len(legend_items),
        frameon=False,
        fontsize=10.5,
    )
    figure.subplots_adjust(
        left=0.03,
        right=0.885,
        top=0.77 if single_row else 0.86,
        bottom=0.08,
        wspace=0.06,
        hspace=0.40,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    if show:
        plt.show()
    plt.close(figure)


def surface_detail_path(output_directory: Path, activation: str) -> Path:
    """Ranger les vues agrandies dans un sous-dossier explicite."""
    activation_slug = slugify_experiment_name(str(activation))
    return output_directory / f"activation_{activation_slug}.png"


def plot_surfaces_3d(aggregated: Any, output_directory: Path, show: bool) -> None:
    """Créer uniquement la présentation 3D lisible, séparée par activation."""
    panels = activation_dropout_panels(aggregated)
    if not panels:
        raise ValueError("aucune activation ni valeur de dropout à représenter")

    activations = sorted({activation for activation, _ in panels})
    for activation in activations:
        activation_panels = [
            panel for panel in panels if panel[0] == activation
        ]
        _render_surfaces_3d(
            aggregated,
            surface_detail_path(output_directory, activation),
            show,
            activation_panels,
            detail_activation=activation,
        )


def write_output_catalog(paths: ExperimentPaths) -> None:
    import pandas as pd

    generated_at = utc_now()
    activation_labels = {"relu": "ReLU", "softmax": "Softmax"}

    def surface_catalog_entry(surface_path: Path) -> dict[str, str]:
        activation_key = surface_path.stem.removeprefix("activation_")
        activation_label = activation_labels.get(activation_key, activation_key)
        return {
            "path": relative_to_experiment(surface_path, paths),
            "type": "figure PNG détaillée",
            "description": (
                "Vue 3D agrandie des dropouts pour l'activation "
                f"{activation_label}, optimisée "
                "pour la lecture à l'écran."
            ),
        }

    surface_entries = [
        surface_catalog_entry(surface_path)
        for surface_path in sorted(paths.surface_details.glob("activation_*.png"))
    ]
    entries = [
        {
            "path": relative_to_experiment(paths.configuration, paths),
            "type": "configuration",
            "description": "Paramètres, provenance des entrées et chemins de la dernière invocation.",
        },
        {
            "path": relative_to_experiment(paths.guide, paths),
            "type": "documentation",
            "description": "Guide rapide de l'arborescence et de la reprise.",
        },
        {
            "path": relative_to_experiment(paths.raw_csv, paths),
            "type": "données brutes",
            "description": "Une ligne par entraînement, combinaison et graine.",
        },
        {
            "path": relative_to_experiment(paths.aggregated_csv, paths),
            "type": "données agrégées",
            "description": "Statistiques par combinaison de filter_1, filter_2 et dropout.",
        },
        {
            "path": relative_to_experiment(paths.heatmaps, paths),
            "type": "figure PNG",
            "description": "Heatmaps 2D de l'accuracy moyenne, une facette par dropout.",
        },
        {
            "path": relative_to_experiment(paths.scatter_3d, paths),
            "type": "figure PNG",
            "description": "Nuage 3D des trois hyperparamètres, coloré par accuracy moyenne.",
        },
        *surface_entries,
        {
            "path": relative_to_experiment(paths.analytical_csv, paths),
            "type": "données d'analyse",
            "description": (
                "Une ligne par configuration, transformations logarithmiques et "
                "incertitude nécessaires aux graphiques."
            ),
        },
        {
            "path": relative_to_experiment(paths.spearman_csv, paths),
            "type": "données d'analyse",
            "description": "Corrélations descriptives de Spearman au grain configuration.",
        },
        {
            "path": relative_to_experiment(paths.pearson_csv, paths),
            "type": "données d'analyse",
            "description": "Corrélations descriptives de Pearson au grain configuration.",
        },
        {
            "path": relative_to_experiment(paths.scatter_relationships, paths),
            "type": "figure PNG",
            "description": (
                "Scatter plots accuracy-hyperparamètres avec activation distinguée "
                "par couleur et marque."
            ),
        },
        {
            "path": relative_to_experiment(paths.scatter_cost, paths),
            "type": "figure PNG",
            "description": "Relations coût, taille du modèle et accuracy moyenne.",
        },
        {
            "path": relative_to_experiment(paths.correlation_matrix, paths),
            "type": "figure PNG",
            "description": "Matrice des corrélations descriptives de Spearman.",
        },
        {
            "path": relative_to_experiment(paths.accuracy_histograms, paths),
            "type": "figure PNG",
            "description": "Histogrammes des accuracies brutes par activation.",
        },
        {
            "path": relative_to_experiment(paths.stability_histograms, paths),
            "type": "figure PNG",
            "description": "Histogrammes de dispersion et de durée des configurations.",
        },
        {
            "path": relative_to_experiment(paths.analysis_catalog, paths),
            "type": "catalogue",
            "description": "Inventaire détaillé des sorties de l'analyse descriptive.",
        },
    ]
    catalog = pd.DataFrame(entries)
    catalog["exists"] = [
        (paths.root / relative_path).exists() for relative_path in catalog["path"]
    ]
    catalog["catalogued_at_utc"] = generated_at
    write_dataframe_atomic(catalog, paths.catalog)


def print_plan(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    slug: str,
    seeds: Sequence[int],
    runs: Iterable[dict[str, Any]],
    protocol_id: str,
) -> None:
    run_count = len(list(runs))
    print(f"Nom de dossier : {slug}")
    print(f"Dossier de sortie : {paths.root}")
    print(f"Protocol ID : {protocol_id}")
    if args.preset_surface_dropout_04:
        print("Preset : surface dense 7 x 7, dropout=0.4 uniquement")
    if args.preset_limite_128:
        print(
            "Preset : criblage jusqu'à 128 filtres, cinq dropouts et "
            "deux activations"
        )
    if args.plot_only:
        print(
            "Mode --plot-only : aucune grille ne sera entraînée ; "
            "le CSV brut existant sera relu."
        )
        return
    print(f"filter_1 : {list(args.filters_1)}")
    print(f"filter_2 : {list(args.filters_2)}")
    print(f"dropout : {list(args.dropouts)}")
    print(f"activations Conv2D : {list(args.activations)}")
    print(f"activation de sortie : {OUTPUT_ACTIVATION} (fixe)")
    print(f"graines : {list(seeds)}")
    print(
        "Source MNIST : "
        + (
            str(args.mnist_path_resolved)
            if args.mnist_path_resolved is not None
            else "cache Keras / téléchargement automatique"
        )
    )
    if args.cascade_dir_resolved is not None:
        print(
            f"Complément Cascade : {args.cascade_dir_resolved} "
            f"({args.cascade_samples} exemples, entraînement uniquement)"
        )
        print("Validation : MNIST uniquement ; test : MNIST uniquement")
    else:
        print("Complément Cascade : aucun")
    print(f"Nombre total d'entraînements planifiés : {run_count}")
    if run_count >= 100:
        print(
            "Attention : cette grille est robuste mais longue. Pour un essai, utilisez "
            "--repetitions 1 --epochs 1 --train-limit 2000 --test-limit 500."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        # Les rendus doivent aussi fonctionner depuis un terminal sans interface
        # graphique (CI, SSH ou exécution automatisée).
        os.environ.setdefault("MPLBACKEND", "Agg")
    validate_args(parser, args)
    apply_quick_preset(args)
    apply_dense_surface_preset(args)
    apply_limit_128_preset(args)
    resolve_mnist_path(parser, args)
    resolve_cascade_dataset(parser, args)

    display_name = args.nom_experience.strip()
    try:
        slug = slugify_experiment_name(display_name)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    paths = ExperimentPaths(slug)
    try:
        seeds = resolve_seeds(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plot_only:
        # Ne pas recalculer un protocole trompeur depuis les valeurs CLI par
        # défaut : le manifeste existant est l'unique source d'identité.
        previous_configuration = validate_existing_experiment(
            parser,
            args,
            paths,
            seeds,
            "",
        )
        if previous_configuration is None:  # pragma: no cover - protégé plus haut
            parser.error("configuration.json est requis en mode --plot-only")
        try:
            protocol_id = protocol_id_from_configuration(previous_configuration)
        except ValueError as exc:
            parser.error(str(exc))
        runs: list[dict[str, Any]] = []
    else:
        protocol_id = make_protocol_id(args, seeds)
        try:
            runs = planned_runs(args, seeds, protocol_id)
        except ValueError as exc:
            parser.error(str(exc))
        validate_existing_experiment(
            parser,
            args,
            paths,
            seeds,
            protocol_id,
        )

    print_plan(args, paths, slug, seeds, runs, protocol_id)
    if args.dry_run:
        return 0
    if (
        len(runs) >= 50
        and not args.confirmer_grande_grille
        and not args.plot_only
    ):
        parser.error(
            f"cette grille demande {len(runs)} entraînements. Vérifiez-la avec "
            "--dry-run, utilisez --preset-rapide, ou ajoutez "
            "--confirmer-grande-grille pour la lancer."
        )
    paths.create()
    write_experiment_documentation(
        paths,
        args,
        display_name,
        slug,
        seeds,
        len(runs),
        protocol_id,
    )

    if args.plot_only:
        raw_results = load_raw_results(paths, legacy_protocol_id=protocol_id)
    else:
        raw_results = train_grid(
            args,
            paths,
            display_name,
            runs,
            seeds,
            protocol_id,
        )

    aggregated = aggregate_results(raw_results, paths)
    exit_code = completion_exit_code(aggregated)
    no_success = exit_code == NO_SUCCESS_EXIT_CODE
    if no_success:
        print(
            "ERREUR : aucun entraînement réussi avec des métriques finies et "
            "cohérentes. Les échecs restent catalogués ; le programme terminera "
            f"avec le code {NO_SUCCESS_EXIT_CODE}.",
            file=sys.stderr,
        )
    plot_errors: list[str] = []
    for label, plotting_function, output_path in (
        ("heatmaps 2D", plot_heatmaps, paths.heatmaps),
        ("nuage 3D", plot_scatter_3d, paths.scatter_3d),
        ("surfaces 3D", plot_surfaces_3d, paths.surface_details),
    ):
        try:
            plotting_function(aggregated, output_path, args.show)
        except Exception as exc:
            plot_errors.append(f"{label}: {exc}")
            print(f"Graphique ignoré ({label}) : {exc}", file=sys.stderr)

    try:
        from scripts.experiences.analyser_hyperparametres import generate_analysis_outputs

        generate_analysis_outputs(
            raw_results,
            aggregated,
            paths.root,
            show=args.show,
        )
    except Exception as exc:
        plot_errors.append(f"analyse descriptive: {exc}")
        print(f"Analyse descriptive ignorée : {exc}", file=sys.stderr)

    write_output_catalog(paths)
    print(f"\nRésultats bruts : {paths.raw_csv}")
    print(f"Résultats agrégés : {paths.aggregated_csv}")
    print(f"Heatmaps 2D : {paths.heatmaps}")
    print(f"Nuage 3D : {paths.scatter_3d}")
    print(f"Surfaces 3D détaillées : {paths.surface_details}")
    print(f"Analyse des corrélations : {paths.correlation_figures}")
    print(f"Histogrammes : {paths.distribution_figures}")
    print(f"Catalogue : {paths.catalog}")
    if plot_errors:
        print("Avertissements de rendu : " + " ; ".join(plot_errors))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
