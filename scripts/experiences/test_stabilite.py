#!/usr/bin/env python3
"""Mesurer la stabilité statistique d'un CNN sur MNIST.

Sans option, le protocole historique est conservé : deux convolutions de 4 et
8 filtres, noyaux 5x5 puis 4x4, activation ``softmax`` dans les convolutions et
dropout à 0,3. La commande historique représente 100 entraînements ; elle doit
désormais être confirmée explicitement afin d'éviter un lancement accidentel.

Exemples, depuis la racine du projet :

    .venv/bin/python scripts/experiences/test_stabilite.py --dry-run

    .venv/bin/python scripts/experiences/test_stabilite.py \
        --nom-experience stabilite_historique \
        --confirmer-grande-etude

    .venv/bin/python scripts/experiences/test_stabilite.py \
        --nom-experience stabilite_locale \
        --filters-1 16,20,24 --filters-2 40,48,56 \
        --dropouts 0.35,0.375,0.4,0.425,0.45 \
        --activation-conv relu --repetitions 5 \
        --confirmer-grande-etude

Les résultats sont écrits après chaque essai. Relancer exactement le même
protocole reprend les graines qui ne sont pas encore marquées ``success``.
"""

from __future__ import annotations

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401


import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from reconnaissance_chiffres import config as env_config


os.environ["KERAS_BACKEND"] = "tensorflow"

DEFAULT_FILTERS_1 = (4,)
DEFAULT_FILTERS_2 = (8,)
DEFAULT_DROPOUTS = (0.3,)
DEFAULT_ACTIVATIONS = ("softmax",)
DEFAULT_KERNEL_1 = 5
DEFAULT_KERNEL_2 = 4
DEFAULT_OUTPUT_ROOT = env_config.SORTIES_STABILITE
CONFIRMATION_THRESHOLD = 50
KERAS_MNIST_SHA256 = (
    "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
)
SPLIT_SCHEMA = "stabilite_split_v1"
SUPPORTED_ACTIVATIONS = {
    "elu",
    "gelu",
    "linear",
    "relu",
    "selu",
    "sigmoid",
    "softmax",
    "swish",
    "tanh",
}

RAW_COLUMNS = [
    "protocol_id",
    "experiment_name",
    "run_id",
    "config_id",
    "model_mode",
    "filter_1",
    "filter_2",
    "kernel_1",
    "kernel_2",
    "activation_conv",
    "dropout",
    "seed",
    "repetition",
    "runs_requested_for_config",
    "split_seed",
    "epochs_requested",
    "epochs_completed",
    "best_epoch",
    "batch_size",
    "learning_rate",
    "train_samples",
    "validation_samples",
    "test_samples",
    "model_parameters",
    "best_train_accuracy",
    "best_val_accuracy",
    "best_val_loss",
    "test_accuracy",
    "test_loss",
    "duration_seconds",
    "status",
    "error_type",
    "error_message",
    "completed_at_utc",
]

HISTORY_COLUMNS = [
    "protocol_id",
    "run_id",
    "config_id",
    "seed",
    "epoch",
    "reached",
    "train_accuracy",
    "train_loss",
    "val_accuracy",
    "val_loss",
]

COMPARISON_COLUMNS = [
    "protocol_id",
    "reference_config_id",
    "candidate_config_id",
    "paired_seeds",
    "confidence_level",
    "delta_test_accuracy_mean",
    "delta_test_accuracy_std",
    "delta_test_accuracy_sem",
    "delta_test_accuracy_ci_low",
    "delta_test_accuracy_ci_high",
    "delta_test_accuracy_median",
    "candidate_wins",
    "ties",
    "reference_wins",
]

AGGREGATED_COLUMNS = [
    "protocol_id",
    "config_id",
    "model_mode",
    "filter_1",
    "filter_2",
    "kernel_1",
    "kernel_2",
    "activation_conv",
    "dropout",
    "runs_expected",
    "runs_attempted",
    "runs_successful",
    "runs_failed",
    "is_complete",
    "confidence_level",
    "test_accuracy_mean",
    "test_accuracy_std",
    "test_accuracy_sem",
    "test_accuracy_ci_low",
    "test_accuracy_ci_high",
    "test_accuracy_median",
    "test_accuracy_q1",
    "test_accuracy_q3",
    "test_accuracy_min",
    "test_accuracy_max",
    "test_loss_mean",
    "test_loss_std",
    "best_val_accuracy_mean",
    "best_val_accuracy_std",
    "duration_seconds_mean",
    "duration_seconds_median",
    "epochs_completed_mean",
    "epochs_completed_std",
]


@dataclass(frozen=True)
class StabilityPaths:
    """Tous les chemins d'une expérience de stabilité."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "donnees"

    @property
    def figures(self) -> Path:
        return self.root / "graphiques"

    @property
    def configuration(self) -> Path:
        return self.root / "configuration.json"

    @property
    def readme(self) -> Path:
        return self.root / "LISEZ_MOI.txt"

    @property
    def catalog(self) -> Path:
        return self.root / "catalogue_sorties.csv"

    @property
    def raw_csv(self) -> Path:
        return self.data / "resultats_bruts.csv"

    @property
    def history_csv(self) -> Path:
        return self.data / "courbes_par_epoque.csv"

    @property
    def aggregated_csv(self) -> Path:
        return self.data / "resultats_agreges.csv"

    @property
    def comparisons_csv(self) -> Path:
        return self.data / "comparaisons_pairees.csv"

    @property
    def errors_csv(self) -> Path:
        return self.data / "erreurs.csv"

    @property
    def split_indices(self) -> Path:
        return self.data / "indices_split.npz"

    @property
    def mean_ci_figure(self) -> Path:
        return self.figures / "accuracy_moyenne_ic95.png"

    @property
    def distributions_figure(self) -> Path:
        return self.figures / "distribution_accuracy_runs.png"

    @property
    def paired_figure(self) -> Path:
        return self.figures / "trajectoires_appariees_par_seed.png"

    @property
    def learning_curves_figure(self) -> Path:
        return self.figures / "courbes_moyennes_ecart_type.png"

    def create(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value).strip("._-").lower()
    if not slug:
        raise argparse.ArgumentTypeError(
            "le nom doit contenir au moins une lettre ou un chiffre"
        )
    return slug[:100]


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
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            "la valeur doit être strictement comprise entre 0 et 1"
        )
    return parsed


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


def comma_separated_dropouts(value: str) -> tuple[float, ...]:
    try:
        values = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "utiliser des décimaux séparés par des virgules"
        ) from exc
    if not values or any(
        not math.isfinite(item) or item < 0.0 or item >= 1.0 for item in values
    ):
        raise argparse.ArgumentTypeError("chaque dropout doit vérifier 0 <= d < 1")
    return values


def comma_separated_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "utiliser des graines entières séparées par des virgules"
        ) from exc
    if not seeds or any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "les graines doivent être comprises entre 0 et 2**32-1"
        )
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("les graines doivent être uniques")
    return seeds


def comma_separated_activations(value: str) -> tuple[str, ...]:
    activations = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    unknown = sorted(set(activations) - SUPPORTED_ACTIVATIONS)
    if not activations or unknown:
        valid = ", ".join(sorted(SUPPORTED_ACTIVATIONS))
        suffix = f" Inconnues : {', '.join(unknown)}." if unknown else ""
        raise argparse.ArgumentTypeError(f"activations acceptées : {valid}.{suffix}")
    return activations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Répète l'entraînement d'un CNN, sauvegarde chaque essai et produit "
            "des statistiques avec intervalles de confiance de Student."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nom-experience",
        "--experiment-name",
        default="stabilite_architecture_historique",
        help="nom du sous-dossier de sortie",
    )
    parser.add_argument(
        "--sortie-racine",
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="dossier qui contient les expériences de stabilité",
    )
    parser.add_argument(
        "--filters-1",
        "--filtres-1",
        type=comma_separated_ints,
        default=DEFAULT_FILTERS_1,
        metavar="LISTE",
        help="nombres de filtres de la première convolution",
    )
    parser.add_argument(
        "--filters-2",
        "--filtres-2",
        type=comma_separated_ints,
        default=DEFAULT_FILTERS_2,
        metavar="LISTE",
        help="nombres de filtres de la deuxième convolution",
    )
    parser.add_argument(
        "--dropouts",
        "--dropout",
        type=comma_separated_dropouts,
        default=DEFAULT_DROPOUTS,
        metavar="LISTE",
        help="taux de dropout, séparés par des virgules",
    )
    parser.add_argument(
        "--activation-conv",
        type=comma_separated_activations,
        default=DEFAULT_ACTIVATIONS,
        metavar="LISTE",
        help=(
            "activation(s) appliquée(s) aux deux Conv2D ; une configuration est "
            "créée par activation"
        ),
    )
    parser.add_argument("--kernel-1", type=positive_int, default=DEFAULT_KERNEL_1)
    parser.add_argument("--kernel-2", type=positive_int, default=DEFAULT_KERNEL_2)
    parser.add_argument(
        "--modele",
        "--model",
        type=Path,
        default=None,
        help=(
            "modèle Keras complet utilisé comme gabarit d'architecture ; ses poids "
            "ne sont pas réutilisés et les options filtres/dropout sont alors fixes"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        help=(
            "'mnist' ou chemin d'un NPZ contenant x_train, y_train, x_test, y_test"
        ),
    )
    parser.add_argument(
        "--mnist-path",
        type=Path,
        default=None,
        help="fichier mnist.npz local prioritaire lorsque --dataset=mnist",
    )
    parser.add_argument(
        "--train-limit",
        type=positive_int,
        default=None,
        help="sous-échantillon stratifié du train complet",
    )
    parser.add_argument(
        "--test-limit",
        type=positive_int,
        default=None,
        help="sous-échantillon stratifié du test complet",
    )
    parser.add_argument(
        "--validation-split",
        type=probability,
        default=0.15,
        help="fraction fixe et stratifiée réservée à la validation",
    )
    parser.add_argument(
        "--split-seed",
        "--dataset-seed",
        type=non_negative_int,
        default=2026,
        help="graine des sous-échantillons et du split ; distincte des entraînements",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=100,
        help="nombre de graines d'entraînement par configuration",
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
        help="première graine si --seeds n'est pas fourni",
    )
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--patience", type=non_negative_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--restore-best-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restaurer les poids de la meilleure val_loss avant le test",
    )
    parser.add_argument(
        "--deterministe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="demander des opérations TensorFlow déterministes lorsque disponibles",
    )
    parser.add_argument(
        "--niveau-confiance",
        type=probability,
        default=None,
        help=(
            "niveau des intervalles de Student ; défaut 0.95, ou valeur enregistrée "
            "en mode --plot-only"
        ),
    )
    parser.add_argument(
        "--reference-config",
        default=None,
        help=(
            "config_id de référence ; par défaut la première configuration lorsque "
            "la grille en contient plusieurs"
        ),
    )
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="recalculer les agrégats et figures depuis les CSV existants",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="afficher le protocole sans créer de fichier ni charger TensorFlow",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="réexécuter et remplacer les runs demandés, même déjà réussis",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="afficher les figures après leur sauvegarde",
    )
    parser.add_argument(
        "--confirmer-grande-etude",
        action="store_true",
        help=f"autoriser une invocation d'au moins {CONFIRMATION_THRESHOLD} runs",
    )
    parser.add_argument(
        "--preset-rapide",
        action="store_true",
        help=(
            "validation technique : 2 graines, 1 époque, 2 000 images de train "
            "et 500 de test"
        ),
    )
    return parser


def apply_quick_preset(args: argparse.Namespace) -> None:
    if not args.preset_rapide:
        return
    args.repetitions = 2
    args.seeds = None
    args.epochs = 1
    args.patience = 0
    args.train_limit = 2_000
    args.test_limit = 500


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate doit être strictement positif")
    if args.plot_only and args.force:
        parser.error("--plot-only et --force sont incompatibles")
    if args.plot_only and args.dry_run:
        parser.error("--plot-only et --dry-run sont incompatibles")
    if args.dataset != "mnist" and args.mnist_path is not None:
        parser.error("--mnist-path ne s'utilise qu'avec --dataset=mnist")
    if args.modele is not None:
        grid_is_default = (
            args.filters_1 == DEFAULT_FILTERS_1
            and args.filters_2 == DEFAULT_FILTERS_2
            and args.dropouts == DEFAULT_DROPOUTS
            and args.activation_conv == DEFAULT_ACTIVATIONS
            and args.kernel_1 == DEFAULT_KERNEL_1
            and args.kernel_2 == DEFAULT_KERNEL_2
        )
        if not grid_is_default:
            parser.error(
                "--modele ne peut pas être combiné avec une grille de filtres, "
                "dropout, activation ou noyaux"
            )


def resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.seeds is not None:
        return args.seeds
    seeds = tuple(args.seed_base + offset for offset in range(args.repetitions))
    if seeds[-1] > 2**32 - 1:
        raise argparse.ArgumentTypeError(
            "--seed-base + --repetitions dépasse la graine maximale 2**32-1"
        )
    return seeds


def resolve_path(value: Path) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        expanded = env_config.PROJECT_ROOT / expanded
    return expanded.resolve()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_npz_normalization(path: Path) -> dict[str, Any]:
    """Déterminer sans ambiguïté la normalisation d'un dataset NPZ.

    Les deux partitions d'images doivent utiliser la même échelle. Les valeurs
    dans [0, 1] sont conservées ; celles dans ]1, 255] sont divisées par 255.
    La décision et les plages observées font partie du protocole.
    """
    import numpy as np

    required = {"x_train", "y_train", "x_test", "y_test"}
    try:
        with np.load(path, allow_pickle=False) as dataset:
            missing = sorted(required - set(dataset.files))
            if missing:
                raise ValueError(
                    "le dataset NPZ ne contient pas : " + ", ".join(missing)
                )
            image_arrays = {
                "x_train": np.asarray(dataset["x_train"]),
                "x_test": np.asarray(dataset["x_test"]),
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"dataset NPZ illisible : {exc}") from exc

    observations: dict[str, dict[str, Any]] = {}
    modes: set[str] = set()
    for name, images in image_arrays.items():
        if images.size == 0:
            raise ValueError(f"{name} est vide")
        if not np.issubdtype(images.dtype, np.number) or np.issubdtype(
            images.dtype, np.complexfloating
        ):
            raise ValueError(f"{name} doit contenir des pixels numériques")
        minimum = float(np.min(images))
        maximum = float(np.max(images))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(f"{name} contient des pixels NaN ou infinis")
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(
                f"{name} doit être dans [0, 1] ou [0, 255], plage reçue "
                f"[{minimum:g}, {maximum:g}]"
            )
        mode = "deja_0_1" if maximum <= 1.0 else "division_255"
        modes.add(mode)
        observations[name] = {
            "dtype": str(images.dtype),
            "minimum": minimum,
            "maximum": maximum,
        }
    if len(modes) != 1:
        raise ValueError(
            "x_train et x_test n'utilisent pas la même échelle de pixels"
        )
    mode = modes.pop()
    return {
        "policy": "inspection_plage_npz_v1",
        "mode": mode,
        "divisor": 1.0 if mode == "deja_0_1" else 255.0,
        "observations": observations,
    }


def resolve_dataset_descriptor(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> dict[str, Any]:
    if args.dataset == "mnist":
        explicit_path = args.mnist_path
        if explicit_path is not None:
            resolved = resolve_path(explicit_path)
            if not resolved.is_file():
                parser.error(f"dataset introuvable : {resolved}")
            digest = hash_file(resolved)
            try:
                normalization = inspect_npz_normalization(resolved)
            except ValueError as exc:
                parser.error(str(exc))
            return {
                "kind": "mnist_npz_explicit",
                "display_name": "MNIST",
                "resolved_path": str(resolved),
                "sha256": digest,
                "identity": f"mnist-explicite:{digest}",
                "normalization": normalization,
            }

        candidates = (
            env_config.PROJECT_CACHE / "keras" / "datasets" / "mnist.npz",
            Path.home() / ".keras" / "datasets" / "mnist.npz",
        )
        resolved = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()), None
        )
        if resolved is not None:
            digest = hash_file(resolved)
            if digest != KERAS_MNIST_SHA256:
                parser.error(
                    "le cache MNIST local ne correspond pas au fichier officiel Keras : "
                    f"{resolved} (SHA-256 {digest})"
                )
        return {
            "kind": "mnist_keras",
            "display_name": "MNIST Keras",
            "resolved_path": str(resolved) if resolved is not None else None,
            "sha256": KERAS_MNIST_SHA256,
            "sha256_verified_locally": resolved is not None,
            "identity": f"mnist-keras:{KERAS_MNIST_SHA256}",
            "normalization": {
                "policy": "mnist_keras_officiel_v1",
                "mode": "division_255",
                "divisor": 255.0,
            },
        }

    resolved = resolve_path(Path(args.dataset))
    if not resolved.is_file():
        parser.error(f"dataset NPZ introuvable : {resolved}")
    digest = hash_file(resolved)
    try:
        normalization = inspect_npz_normalization(resolved)
    except ValueError as exc:
        parser.error(str(exc))
    return {
        "kind": "custom_npz",
        "display_name": resolved.stem,
        "resolved_path": str(resolved),
        "sha256": digest,
        "identity": f"npz:{digest}",
        "normalization": normalization,
    }


def resolve_model_descriptor(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> dict[str, Any]:
    if args.modele is None:
        return {
            "mode": "cnn_2conv_parametrique",
            "resolved_path": None,
            "sha256": None,
        }
    resolved = resolve_path(args.modele)
    if not resolved.is_file():
        parser.error(f"modèle Keras introuvable : {resolved}")
    digest = hash_file(resolved)
    return {
        "mode": "gabarit_keras",
        "resolved_path": str(resolved),
        "sha256": digest,
    }


def float_token(value: float) -> str:
    """Représentation lisible et injective d'un float Python fini.

    ``repr`` est une représentation décimale à aller-retour exact : deux
    floats distincts ne partagent donc pas le même texte.
    """
    if not math.isfinite(value):
        raise ValueError("un float fini est requis pour construire un identifiant")
    return (
        repr(float(value))
        .replace(".", "p")
        .replace("+", "plus")
        .replace("-", "moins")
    )


def build_configurations(
    args: argparse.Namespace, model_descriptor: dict[str, Any]
) -> list[dict[str, Any]]:
    if model_descriptor["mode"] == "gabarit_keras":
        model_stem = slugify(Path(model_descriptor["resolved_path"]).stem)
        return [
            {
                "config_id": f"modele_{model_stem}",
                "model_mode": "gabarit_keras",
                "filter_1": None,
                "filter_2": None,
                "kernel_1": None,
                "kernel_2": None,
                "activation_conv": None,
                "dropout": None,
            }
        ]

    configurations: list[dict[str, Any]] = []
    for filter_1 in args.filters_1:
        for filter_2 in args.filters_2:
            for dropout in args.dropouts:
                for activation in args.activation_conv:
                    config_id = (
                        f"f1_{filter_1}__f2_{filter_2}__dropout_{float_token(dropout)}"
                        f"__activation_{slugify(activation)}"
                    )
                    configurations.append(
                        {
                            "config_id": config_id,
                            "model_mode": "cnn_2conv_parametrique",
                            "filter_1": filter_1,
                            "filter_2": filter_2,
                            "kernel_1": args.kernel_1,
                            "kernel_2": args.kernel_2,
                            "activation_conv": activation,
                            "dropout": dropout,
                        }
                    )
    config_ids = [configuration["config_id"] for configuration in configurations]
    if len(config_ids) != len(set(config_ids)):
        raise ValueError("la grille produit des config_id dupliqués")
    return configurations


def resolve_reference_config(
    parser: argparse.ArgumentParser,
    requested: str | None,
    configurations: Sequence[dict[str, Any]],
) -> str | None:
    config_ids = [configuration["config_id"] for configuration in configurations]
    if requested is not None and requested not in config_ids:
        parser.error(
            "--reference-config doit être un config_id de la grille. Valeurs : "
            + ", ".join(config_ids)
        )
    if requested is not None:
        return requested
    if len(config_ids) > 1:
        return config_ids[0]
    return None


def implementation_descriptor() -> dict[str, Any]:
    """Empreinte des éléments susceptibles de modifier les résultats."""
    return {
        "script_sha256": hash_file(Path(__file__).resolve()),
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "libraries": {
            distribution: package_version(distribution)
            for distribution in (
                "keras",
                "tensorflow",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
            )
        },
    }


def protocol_payload(
    args: argparse.Namespace,
    dataset_descriptor: dict[str, Any],
    model_descriptor: dict[str, Any],
    configurations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Partie immuable : les graines et options d'affichage en sont exclues."""
    model_protocol: dict[str, Any] = {
        "mode": model_descriptor["mode"],
        "sha256": model_descriptor["sha256"],
        "configurations": list(configurations),
    }
    if model_descriptor["mode"] == "cnn_2conv_parametrique":
        model_protocol["layers"] = [
            "Input(28,28,1)",
            "Conv2D",
            "BatchNormalization",
            "MaxPooling2D(2,2)",
            "Conv2D",
            "BatchNormalization",
            "MaxPooling2D(2,2)",
            "Flatten",
            "Dropout",
            "Dense(10, softmax)",
        ]

    return {
        "protocol_schema": "stabilite_mnist_v3",
        "implementation": implementation_descriptor(),
        "dataset": {
            "identity": dataset_descriptor["identity"],
            "sha256": dataset_descriptor["sha256"],
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "validation_split": args.validation_split,
            "split_seed": args.split_seed,
            "normalization": dataset_descriptor["normalization"],
        },
        "model": model_protocol,
        "training": {
            "epochs_max": args.epochs,
            "patience": args.patience,
            "restore_best_weights": args.restore_best_weights,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy",
            "deterministic_ops_requested": args.deterministe,
        },
    }


def make_protocol_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def planned_runs(
    protocol_id: str,
    configurations: Sequence[dict[str, Any]],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for configuration in configurations:
        for repetition, seed in enumerate(seeds, start=1):
            runs.append(
                {
                    **configuration,
                    "run_id": (
                        f"{protocol_id[:12]}__{configuration['config_id']}__seed_{seed}"
                    ),
                    "seed": seed,
                    "repetition": repetition,
                }
            )
    run_ids = [run["run_id"] for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("le protocole produit des run_id dupliqués")
    return runs


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_dataframe_atomic(dataframe: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        dataframe.to_csv(temporary, index=False, na_rep="NaN")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_dataframe(path: Path, columns: Sequence[str]) -> Any:
    import pandas as pd

    if not path.exists():
        return pd.DataFrame(columns=columns)
    dataframe = pd.read_csv(path)
    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = None
    return dataframe[list(columns)]


def validated_integer_series(series: Any, label: str, maximum: int | None = None) -> Any:
    import numpy as np
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values < 0.0) | (values != np.floor(values))
    if maximum is not None:
        invalid |= values > maximum
    if bool(np.any(invalid)):
        raise ValueError(f"{label} contient une valeur entière invalide")
    return numeric.astype("int64")


def validate_result_frames(
    raw_results: Any,
    histories: Any | None,
    protocol_id: str,
) -> None:
    """Refuser tout mélange de protocole ou toute ambiguïté de clé."""
    import numpy as np
    import pandas as pd

    def validate_protocol(dataframe: Any, label: str) -> None:
        if dataframe.empty:
            return
        values = dataframe["protocol_id"].astype("string")
        if bool(values.isna().any()) or set(values.astype(str)) != {protocol_id}:
            received = sorted(set(values.dropna().astype(str)))
            raise ValueError(
                f"{label} contient un protocol_id absent ou étranger : {received}"
            )

    validate_protocol(raw_results, "resultats_bruts.csv")
    if not raw_results.empty:
        if bool(raw_results["run_id"].isna().any()) or bool(
            raw_results["config_id"].isna().any()
        ):
            raise ValueError("resultats_bruts.csv contient une clé manquante")
        seeds = validated_integer_series(
            raw_results["seed"], "resultats_bruts.csv.seed", 2**32 - 1
        )
        keys = raw_results[["run_id", "config_id"]].copy()
        keys["seed"] = seeds
        if bool(keys["run_id"].astype(str).duplicated(keep=False).any()):
            raise ValueError("resultats_bruts.csv contient des run_id dupliqués")
        if bool(keys[["config_id", "seed"]].duplicated(keep=False).any()):
            raise ValueError(
                "resultats_bruts.csv contient plusieurs lignes pour une même "
                "configuration et une même seed"
            )
        statuses = set(raw_results["status"].dropna().astype(str))
        if statuses - {"success", "error"} or bool(raw_results["status"].isna().any()):
            raise ValueError("resultats_bruts.csv contient un status inconnu")
        successful = raw_results[raw_results["status"] == "success"]
        metric_ranges = {
            "test_accuracy": (0.0, 1.0),
            "test_loss": (0.0, None),
            "best_train_accuracy": (0.0, 1.0),
            "best_val_accuracy": (0.0, 1.0),
            "best_val_loss": (0.0, None),
            "duration_seconds": (0.0, None),
        }
        for column, (minimum, maximum) in metric_ranges.items():
            values = pd.to_numeric(successful[column], errors="coerce").to_numpy(
                dtype=float
            )
            invalid = ~np.isfinite(values) | (values < minimum)
            if maximum is not None:
                invalid |= values > maximum
            if bool(np.any(invalid)):
                raise ValueError(
                    f"resultats_bruts.csv marque success avec {column} invalide"
                )

    if histories is None:
        return
    validate_protocol(histories, "courbes_par_epoque.csv")
    if histories.empty:
        return
    if bool(histories["run_id"].isna().any()) or bool(
        histories["config_id"].isna().any()
    ):
        raise ValueError("courbes_par_epoque.csv contient une clé manquante")
    seeds = validated_integer_series(
        histories["seed"], "courbes_par_epoque.csv.seed", 2**32 - 1
    )
    epochs = validated_integer_series(
        histories["epoch"], "courbes_par_epoque.csv.epoch"
    )
    if bool((epochs < 1).any()):
        raise ValueError("courbes_par_epoque.csv contient une époque < 1")
    keys = histories[["run_id", "config_id"]].copy()
    keys["seed"] = seeds
    keys["epoch"] = epochs
    if bool(keys[["run_id", "epoch"]].duplicated(keep=False).any()):
        raise ValueError(
            "courbes_par_epoque.csv contient plusieurs lignes pour un même "
            "run_id et une même époque"
        )
    if bool(
        keys[["config_id", "seed", "epoch"]].duplicated(keep=False).any()
    ):
        raise ValueError(
            "courbes_par_epoque.csv contient une clé "
            "(config_id, seed, epoch) dupliquée"
        )
    mapping_counts = keys.groupby("run_id")[["config_id", "seed"]].nunique()
    if bool((mapping_counts > 1).to_numpy().any()):
        raise ValueError(
            "courbes_par_epoque.csv associe un run_id à plusieurs configurations "
            "ou seeds"
        )


def upsert_raw_result(dataframe: Any, row: dict[str, Any], path: Path) -> Any:
    import pandas as pd

    if not dataframe.empty:
        dataframe = dataframe[dataframe["run_id"].astype(str) != str(row["run_id"])]
    dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)
    dataframe = dataframe.sort_values(
        ["config_id", "seed"], kind="stable"
    ).reset_index(drop=True)
    dataframe = dataframe[RAW_COLUMNS]
    write_dataframe_atomic(dataframe, path)
    return dataframe


def upsert_history(dataframe: Any, rows: Sequence[dict[str, Any]], path: Path) -> Any:
    import pandas as pd

    if not rows:
        return dataframe
    run_id = str(rows[0]["run_id"])
    if not dataframe.empty:
        dataframe = dataframe[dataframe["run_id"].astype(str) != run_id]
    dataframe = pd.concat([dataframe, pd.DataFrame(rows)], ignore_index=True)
    dataframe = dataframe.sort_values(
        ["config_id", "seed", "epoch"], kind="stable"
    ).reset_index(drop=True)
    dataframe = dataframe[HISTORY_COLUMNS]
    write_dataframe_atomic(dataframe, path)
    return dataframe


def write_split_indices_atomic(
    path: Path,
    protocol_id: str,
    dataset_identity: str,
    train_source_size: int,
    test_source_size: int,
    train_indices: Any,
    validation_indices: Any,
    test_indices: Any,
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(SPLIT_SCHEMA),
                protocol_id=np.asarray(protocol_id),
                dataset_identity=np.asarray(dataset_identity),
                train_source_size=np.asarray(train_source_size, dtype=np.int64),
                test_source_size=np.asarray(test_source_size, dtype=np.int64),
                train_indices=np.asarray(train_indices, dtype=np.int64),
                validation_indices=np.asarray(validation_indices, dtype=np.int64),
                test_indices=np.asarray(test_indices, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_split_indices(
    path: Path,
    expected_protocol_id: str,
    expected_dataset_identity: str | None = None,
    expected_train_source_size: int | None = None,
    expected_test_source_size: int | None = None,
) -> tuple[Any, Any, Any]:
    import numpy as np

    required = {
        "schema",
        "protocol_id",
        "dataset_identity",
        "train_source_size",
        "test_source_size",
        "train_indices",
        "validation_indices",
        "test_indices",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("champs absents : " + ", ".join(missing))
            schema = str(np.asarray(archive["schema"]).item())
            stored_protocol_id = str(np.asarray(archive["protocol_id"]).item())
            dataset_identity = str(np.asarray(archive["dataset_identity"]).item())
            train_source_size = int(
                np.asarray(archive["train_source_size"]).item()
            )
            test_source_size = int(np.asarray(archive["test_source_size"]).item())
            arrays = tuple(
                np.asarray(archive[name])
                for name in (
                    "train_indices",
                    "validation_indices",
                    "test_indices",
                )
            )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"indices_split.npz illisible : {exc}") from exc

    if schema != SPLIT_SCHEMA:
        raise ValueError(f"schéma de split inconnu : {schema}")
    if stored_protocol_id != expected_protocol_id:
        raise ValueError(
            "indices_split.npz appartient à un autre protocol_id "
            f"({stored_protocol_id[:12]} au lieu de {expected_protocol_id[:12]})"
        )
    if expected_dataset_identity is not None and dataset_identity != expected_dataset_identity:
        raise ValueError("indices_split.npz appartient à un autre dataset")
    if train_source_size <= 0 or test_source_size <= 0:
        raise ValueError("indices_split.npz contient une taille source invalide")
    if (
        expected_train_source_size is not None
        and train_source_size != expected_train_source_size
    ):
        raise ValueError("la taille du train ne correspond plus au split enregistré")
    if (
        expected_test_source_size is not None
        and test_source_size != expected_test_source_size
    ):
        raise ValueError("la taille du test ne correspond plus au split enregistré")

    names = ("train", "validation", "test")
    limits = (train_source_size, train_source_size, test_source_size)
    normalized: list[Any] = []
    for name, values, limit in zip(names, arrays, limits):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"les indices {name} doivent être un vecteur entier")
        values = values.astype(np.int64, copy=False)
        if values.size == 0:
            raise ValueError(f"les indices {name} sont vides")
        if bool(np.any(values < 0)) or bool(np.any(values >= limit)):
            raise ValueError(f"les indices {name} sont hors limites")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"les indices {name} contiennent des doublons")
        normalized.append(values)
    if np.intersect1d(normalized[0], normalized[1]).size:
        raise ValueError("les partitions train et validation se chevauchent")
    return normalized[0], normalized[1], normalized[2]


def validate_existing_experiment(
    parser: argparse.ArgumentParser,
    paths: StabilityPaths,
    protocol_id: str,
) -> dict[str, Any] | None:
    result_files = (
        paths.raw_csv,
        paths.history_csv,
        paths.aggregated_csv,
        paths.comparisons_csv,
        paths.errors_csv,
    )
    existing_results = [path for path in result_files if path.exists()]
    if (existing_results or paths.split_indices.exists()) and not paths.configuration.exists():
        parser.error(
            "le dossier contient des données sans configuration.json ; utilisez un "
            "autre --nom-experience pour ne pas mélanger les protocoles"
        )
    if not paths.configuration.exists():
        return None
    try:
        previous = json.loads(paths.configuration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"configuration existante illisible : {exc}")
    previous_id = previous.get("protocol_id")
    previous_protocol = previous.get("protocol")
    if not isinstance(previous_protocol, dict) or make_protocol_id(previous_protocol) != previous_id:
        parser.error(
            "configuration.json est incohérent : son protocole ne correspond pas "
            "à son protocol_id"
        )
    if previous_id != protocol_id:
        parser.error(
            "le dossier contient un autre protocole "
            f"({str(previous_id)[:12]} au lieu de {protocol_id[:12]}). "
            "Utilisez un nouveau --nom-experience."
        )
    if existing_results and not paths.split_indices.is_file():
        parser.error(
            "des résultats existent sans donnees/indices_split.npz ; la reprise est "
            "refusée car leur partition de données n'est pas vérifiable"
        )
    if paths.split_indices.exists():
        try:
            load_split_indices(paths.split_indices, protocol_id)
        except ValueError as exc:
            parser.error(str(exc))
    raw_results = load_dataframe(paths.raw_csv, RAW_COLUMNS)
    histories = load_dataframe(paths.history_csv, HISTORY_COLUMNS)
    try:
        validate_result_frames(raw_results, histories, protocol_id)
    except ValueError as exc:
        parser.error(str(exc))
    return previous


def write_configuration(
    paths: StabilityPaths,
    args: argparse.Namespace,
    display_name: str,
    protocol_id: str,
    protocol: dict[str, Any],
    dataset_descriptor: dict[str, Any],
    model_descriptor: dict[str, Any],
    seeds: Sequence[int],
    reference_config: str | None,
    previous: dict[str, Any] | None,
    argv: Sequence[str] | None,
) -> dict[str, Any]:
    previous_seeds = (
        previous.get("execution", {}).get("all_requested_seeds", [])
        if previous is not None
        else []
    )
    all_seeds = sorted(set(int(seed) for seed in previous_seeds) | set(seeds))
    now = utc_now()
    command_args = list(argv) if argv is not None else sys.argv[1:]
    payload = {
        "schema_version": 3,
        "protocol_id": protocol_id,
        "experiment": {
            "display_name": display_name,
            "folder_name": paths.root.name,
            "created_at_utc": (
                previous.get("experiment", {}).get("created_at_utc", now)
                if previous is not None
                else now
            ),
            "last_invocation_at_utc": now,
            "command": [sys.executable, str(Path(__file__).resolve()), *command_args],
        },
        "protocol": protocol,
        "sources": {
            "dataset": dataset_descriptor,
            "model": model_descriptor,
        },
        "execution": {
            "seeds_requested_this_invocation": list(seeds),
            "all_requested_seeds": all_seeds,
            "runs_planned_this_invocation": len(protocol["model"]["configurations"])
            * len(seeds),
            "resume_policy": "ignorer les run_id avec status=success",
        },
        "analysis": {
            "reference_config_id": reference_config,
            "confidence_level": args.niveau_confiance,
            "standard_deviation": "écart-type échantillonnal, ddof=1",
            "confidence_interval": "Student bilatéral sur la moyenne",
            "paired_comparison": "différence candidate-référence pour une même seed",
        },
        "runtime": {
            "protocol_implementation": protocol["implementation"],
            "platform": platform.platform(),
            "deterministic_ops_requested": args.deterministe,
            "python_hash_seed_note": (
                "PYTHONHASHSEED doit être défini avant le lancement du processus ; "
                "keras.utils.set_random_seed contrôle les graines de chaque run."
            ),
        },
        "outputs": {
            "raw_results": "donnees/resultats_bruts.csv",
            "epoch_history": "donnees/courbes_par_epoque.csv",
            "aggregated_results": "donnees/resultats_agreges.csv",
            "paired_comparisons": "donnees/comparaisons_pairees.csv",
            "errors": "donnees/erreurs.csv",
            "split_indices": "donnees/indices_split.npz",
            "figures": "graphiques/",
            "catalog": "catalogue_sorties.csv",
        },
    }
    write_json_atomic(paths.configuration, payload)
    return payload


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def stratified_subset_indices(
    labels: Any, limit: int | None, seed: int
) -> Any:
    import numpy as np

    indices = np.arange(len(labels), dtype=np.int64)
    if limit is None or limit >= len(indices):
        return indices
    from sklearn.model_selection import train_test_split

    selected, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=labels,
    )
    return np.asarray(selected, dtype=np.int64)


def load_and_split_dataset(
    args: argparse.Namespace,
    dataset_descriptor: dict[str, Any],
    paths: StabilityPaths,
    protocol_id: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import numpy as np

    resolved_path = dataset_descriptor.get("resolved_path")
    if resolved_path is not None:
        with np.load(resolved_path, allow_pickle=False) as dataset:
            required = {"x_train", "y_train", "x_test", "y_test"}
            missing = sorted(required - set(dataset.files))
            if missing:
                raise ValueError(
                    "le dataset NPZ ne contient pas : " + ", ".join(missing)
                )
            x_train_full = dataset["x_train"]
            y_train_full = dataset["y_train"]
            x_test_full = dataset["x_test"]
            y_test_full = dataset["y_test"]
    else:
        import keras

        (x_train_full, y_train_full), (x_test_full, y_test_full) = (
            keras.datasets.mnist.load_data()
        )

    for images, labels, name in (
        (x_train_full, y_train_full, "train"),
        (x_test_full, y_test_full, "test"),
    ):
        if len(images) != len(labels) or len(images) == 0:
            raise ValueError(f"les images et labels {name} sont vides ou désalignés")
        numeric_labels = np.asarray(labels, dtype=float)
        if (
            not np.isfinite(numeric_labels).all()
            or not np.equal(numeric_labels, np.floor(numeric_labels)).all()
            or bool(np.any(numeric_labels < 0))
            or bool(np.any(numeric_labels > 9))
        ):
            raise ValueError(f"les labels {name} doivent être des entiers entre 0 et 9")

    if paths.split_indices.exists():
        train_indices, validation_indices, test_indices = load_split_indices(
            paths.split_indices,
            protocol_id,
            expected_dataset_identity=dataset_descriptor["identity"],
            expected_train_source_size=len(x_train_full),
            expected_test_source_size=len(x_test_full),
        )
    else:
        from sklearn.model_selection import train_test_split

        selected_train = stratified_subset_indices(
            y_train_full, args.train_limit, args.split_seed
        )
        train_indices, validation_indices = train_test_split(
            selected_train,
            test_size=args.validation_split,
            random_state=args.split_seed + 2,
            stratify=y_train_full[selected_train],
        )
        test_indices = stratified_subset_indices(
            y_test_full, args.test_limit, args.split_seed + 1
        )
        write_split_indices_atomic(
            paths.split_indices,
            protocol_id,
            dataset_descriptor["identity"],
            len(x_train_full),
            len(x_test_full),
            train_indices,
            validation_indices,
            test_indices,
        )

    expected_train_total = min(args.train_limit or len(x_train_full), len(x_train_full))
    expected_test_total = min(args.test_limit or len(x_test_full), len(x_test_full))
    if len(train_indices) + len(validation_indices) != expected_train_total:
        raise ValueError("le split enregistré ne respecte plus --train-limit")
    if len(test_indices) != expected_test_total:
        raise ValueError("le split enregistré ne respecte plus --test-limit")

    normalization = dataset_descriptor.get("normalization", {})
    divisor = float(normalization.get("divisor", math.nan))
    if divisor not in (1.0, 255.0):
        raise ValueError("politique de normalisation absente ou inconnue")

    def prepare(images: Any, indices: Any) -> Any:
        selected = images[indices].astype("float32") / divisor
        if selected.ndim == 3:
            selected = np.expand_dims(selected, axis=-1)
        if selected.shape[1:] != (28, 28, 1):
            raise ValueError(
                "les images doivent avoir la forme (N, 28, 28) ou (N, 28, 28, 1), "
                f"reçu {selected.shape}"
            )
        if (
            not np.isfinite(selected).all()
            or bool(np.any(selected < 0.0))
            or bool(np.any(selected > 1.0))
        ):
            raise ValueError("les pixels normalisés doivent être dans [0, 1]")
        return selected

    x_train = prepare(x_train_full, train_indices)
    x_validation = prepare(x_train_full, validation_indices)
    x_test = prepare(x_test_full, test_indices)
    y_train = np.asarray(y_train_full[train_indices])
    y_validation = np.asarray(y_train_full[validation_indices])
    y_test = np.asarray(y_test_full[test_indices])
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def configure_runtime_environment(deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def set_training_seed(seed: int, deterministic: bool, keras: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            import tensorflow as tf

            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass


def validate_template_model(template_model: Any) -> None:
    """Vérifier une fois le contrat MNIST du gabarit avant tout run."""

    input_shape = getattr(template_model, "input_shape", None)
    output_shape = getattr(template_model, "output_shape", None)
    if isinstance(input_shape, list) or tuple(input_shape or ()) != (None, 28, 28, 1):
        raise ValueError(
            "le modèle-gabarit doit avoir une unique entrée (None, 28, 28, 1), "
            f"reçu {input_shape}"
        )
    if isinstance(output_shape, list) or tuple(output_shape or ()) != (None, 10):
        raise ValueError(
            "le modèle-gabarit doit avoir une unique sortie (None, 10), "
            f"reçu {output_shape}"
        )
    layers = list(getattr(template_model, "layers", []))
    if not layers:
        raise ValueError("le modèle-gabarit ne contient aucune couche")
    last_layer = layers[-1]
    last_config = last_layer.get_config()
    activation = last_config.get("activation")
    is_softmax_layer = type(last_layer).__name__ == "Softmax"
    if activation != "softmax" and not is_softmax_layer:
        raise ValueError(
            "la sortie du modèle-gabarit doit utiliser softmax ; "
            f"activation reçue : {activation or type(last_layer).__name__}"
        )


def build_model(
    keras: Any,
    configuration: dict[str, Any],
    args: argparse.Namespace,
    template_model: Any | None,
) -> Any:
    if template_model is not None:
        model = keras.models.clone_model(template_model)
    else:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(28, 28, 1)),
                keras.layers.Conv2D(
                    configuration["filter_1"],
                    kernel_size=(configuration["kernel_1"], configuration["kernel_1"]),
                    activation=configuration["activation_conv"],
                ),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(
                    configuration["filter_2"],
                    kernel_size=(configuration["kernel_2"], configuration["kernel_2"]),
                    activation=configuration["activation_conv"],
                ),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dropout(configuration["dropout"]),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def validate_success_metrics(
    history: dict[str, Sequence[float]],
    evaluation: dict[str, Any],
    epochs_requested: int,
) -> dict[str, Any]:
    """Retourner des métriques numériques seulement si le run est exploitable."""
    import numpy as np

    specifications = {
        "loss": (0.0, None),
        "accuracy": (0.0, 1.0),
        "val_loss": (0.0, None),
        "val_accuracy": (0.0, 1.0),
    }
    arrays: dict[str, Any] = {}
    lengths: set[int] = set()
    for name, (minimum, maximum) in specifications.items():
        if name not in history:
            raise ValueError(f"historique incomplet : métrique {name} absente")
        values = np.asarray(history[name], dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"historique {name} vide ou non vectoriel")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"historique {name} contient NaN ou inf")
        if bool(np.any(values < minimum)) or (
            maximum is not None and bool(np.any(values > maximum))
        ):
            raise ValueError(f"historique {name} hors plage attendue")
        arrays[name] = values
        lengths.add(int(values.size))
    if len(lengths) != 1:
        raise ValueError("les séries de l'historique n'ont pas la même longueur")
    epochs_completed = lengths.pop()
    if epochs_completed > epochs_requested:
        raise ValueError("l'historique dépasse le nombre d'époques demandé")

    for name, minimum, maximum in (
        ("loss", 0.0, None),
        ("accuracy", 0.0, 1.0),
    ):
        if name not in evaluation:
            raise ValueError(f"évaluation incomplète : métrique {name} absente")
        value = float(evaluation[name])
        if not math.isfinite(value):
            raise FloatingPointError(f"évaluation {name} contient NaN ou inf")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"évaluation {name} hors plage attendue")
    return {
        "arrays": arrays,
        "epochs_completed": epochs_completed,
        "test_accuracy": float(evaluation["accuracy"]),
        "test_loss": float(evaluation["loss"]),
    }


def raw_result_base(
    args: argparse.Namespace,
    protocol_id: str,
    display_name: str,
    run: dict[str, Any],
    seeds: Sequence[int],
    sample_counts: tuple[int, int, int],
) -> dict[str, Any]:
    train_count, validation_count, test_count = sample_counts
    return {
        "protocol_id": protocol_id,
        "experiment_name": display_name,
        "run_id": run["run_id"],
        "config_id": run["config_id"],
        "model_mode": run["model_mode"],
        "filter_1": run["filter_1"],
        "filter_2": run["filter_2"],
        "kernel_1": run["kernel_1"],
        "kernel_2": run["kernel_2"],
        "activation_conv": run["activation_conv"],
        "dropout": run["dropout"],
        "seed": run["seed"],
        "repetition": run["repetition"],
        "runs_requested_for_config": len(seeds),
        "split_seed": args.split_seed,
        "epochs_requested": args.epochs,
        "epochs_completed": None,
        "best_epoch": None,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": train_count,
        "validation_samples": validation_count,
        "test_samples": test_count,
        "model_parameters": None,
        "best_train_accuracy": None,
        "best_val_accuracy": None,
        "best_val_loss": None,
        "test_accuracy": None,
        "test_loss": None,
        "duration_seconds": None,
        "status": "error",
        "error_type": None,
        "error_message": None,
        "completed_at_utc": None,
    }


def history_rows(
    protocol_id: str,
    run: dict[str, Any],
    epochs_requested: int,
    history: dict[str, Sequence[float]] | None,
) -> list[dict[str, Any]]:
    history = history or {}
    completed = len(history.get("loss", []))

    def value_at(key: str, index: int) -> float:
        values = history.get(key, [])
        return float(values[index]) if index < len(values) else math.nan

    rows: list[dict[str, Any]] = []
    for epoch_index in range(epochs_requested):
        rows.append(
            {
                "protocol_id": protocol_id,
                "run_id": run["run_id"],
                "config_id": run["config_id"],
                "seed": run["seed"],
                "epoch": epoch_index + 1,
                "reached": epoch_index < completed,
                "train_accuracy": value_at("accuracy", epoch_index),
                "train_loss": value_at("loss", epoch_index),
                "val_accuracy": value_at("val_accuracy", epoch_index),
                "val_loss": value_at("val_loss", epoch_index),
            }
        )
    return rows


def resumable_success_ids(
    raw_results: Any,
    histories: Any,
    protocol_id: str | None = None,
) -> set[str]:
    """Ne reprendre comme succès que les runs complets et cohérents sur disque."""
    import numpy as np
    import pandas as pd

    if protocol_id is not None:
        validate_result_frames(raw_results, histories, protocol_id)
    if raw_results.empty or histories.empty:
        return set()
    reached = histories["reached"].astype(str).str.lower().isin({"true", "1"})
    reached_counts = (
        histories.loc[reached].groupby("run_id")["epoch"].count().to_dict()
    )
    valid: set[str] = set()
    for _, row in raw_results[raw_results["status"] == "success"].iterrows():
        run_id = str(row["run_id"])
        epochs_completed = pd.to_numeric(
            pd.Series([row["epochs_completed"]]), errors="coerce"
        ).iloc[0]
        accuracy = pd.to_numeric(
            pd.Series([row["test_accuracy"]]), errors="coerce"
        ).iloc[0]
        loss = pd.to_numeric(pd.Series([row["test_loss"]]), errors="coerce").iloc[0]
        epochs_requested = pd.to_numeric(
            pd.Series([row["epochs_requested"]]), errors="coerce"
        ).iloc[0]
        run_history = histories[histories["run_id"].astype(str) == run_id]
        reached_history = run_history[
            run_history["reached"].astype(str).str.lower().isin({"true", "1"})
        ]
        history_accuracy = pd.to_numeric(
            reached_history["train_accuracy"], errors="coerce"
        ).to_numpy(dtype=float)
        history_val_accuracy = pd.to_numeric(
            reached_history["val_accuracy"], errors="coerce"
        ).to_numpy(dtype=float)
        history_losses = pd.to_numeric(
            reached_history[["train_loss", "val_loss"]].stack(), errors="coerce"
        ).to_numpy(dtype=float)
        metrics_valid = (
            np.isfinite(accuracy)
            and 0.0 <= float(accuracy) <= 1.0
            and np.isfinite(loss)
            and float(loss) >= 0.0
        )
        history_valid = (
            np.isfinite(epochs_completed)
            and int(epochs_completed) >= 1
            and int(reached_counts.get(run_id, 0)) == int(epochs_completed)
            and np.isfinite(epochs_requested)
            and int(epochs_requested) >= int(epochs_completed)
            and len(run_history) == int(epochs_requested)
            and len(history_accuracy) == int(epochs_completed)
            and np.isfinite(history_accuracy).all()
            and np.isfinite(history_val_accuracy).all()
            and np.isfinite(history_losses).all()
            and bool(np.all((history_accuracy >= 0.0) & (history_accuracy <= 1.0)))
            and bool(
                np.all(
                    (history_val_accuracy >= 0.0)
                    & (history_val_accuracy <= 1.0)
                )
            )
            and bool(np.all(history_losses >= 0.0))
        )
        if metrics_valid and history_valid:
            valid.add(run_id)
    return valid


def train_runs(
    args: argparse.Namespace,
    paths: StabilityPaths,
    display_name: str,
    protocol_id: str,
    runs: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    dataset_descriptor: dict[str, Any],
    model_descriptor: dict[str, Any],
) -> tuple[Any, Any]:
    raw_results = load_dataframe(paths.raw_csv, RAW_COLUMNS)
    histories = load_dataframe(paths.history_csv, HISTORY_COLUMNS)
    validate_result_frames(raw_results, histories, protocol_id)
    successful_ids = resumable_success_ids(raw_results, histories, protocol_id)
    pending = [
        run for run in runs if args.force or run["run_id"] not in successful_ids
    ]
    print(
        f"{len(runs)} runs demandés ; {len(runs) - len(pending)} déjà réussis ; "
        f"{len(pending)} à exécuter."
    )
    if not pending:
        return raw_results, histories

    import keras
    import numpy as np

    configure_runtime_environment(args.deterministe)
    x_train, y_train, x_validation, y_validation, x_test, y_test = (
        load_and_split_dataset(args, dataset_descriptor, paths, protocol_id)
    )
    sample_counts = (len(x_train), len(x_validation), len(x_test))
    template_model = None
    if model_descriptor["mode"] == "gabarit_keras":
        keras.backend.clear_session()
        try:
            template_model = keras.models.load_model(
                model_descriptor["resolved_path"], compile=False
            )
            validate_template_model(template_model)
        except Exception as exc:
            raise ValueError(f"modèle-gabarit incompatible : {exc}") from exc
    for position, run in enumerate(pending, start=1):
        print(
            f"[{position}/{len(pending)}] {run['config_id']} | seed={run['seed']}"
        )
        row = raw_result_base(
            args, protocol_id, display_name, run, seeds, sample_counts
        )
        started = time.perf_counter()
        fit_history: dict[str, Sequence[float]] | None = None
        try:
            keras.backend.clear_session()
            set_training_seed(run["seed"], args.deterministe, keras, np)
            model = build_model(keras, run, args, template_model)
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=args.restore_best_weights,
                )
            ]
            fitted = model.fit(
                x_train,
                y_train,
                validation_data=(x_validation, y_validation),
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=args.verbose,
                shuffle=True,
            )
            fit_history = fitted.history
            evaluation = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
            validated = validate_success_metrics(
                fit_history, evaluation, args.epochs
            )
            arrays = validated["arrays"]
            best_index = int(np.argmin(arrays["val_loss"]))
            row.update(
                {
                    "epochs_completed": validated["epochs_completed"],
                    "best_epoch": best_index + 1,
                    "model_parameters": model.count_params(),
                    "best_train_accuracy": float(arrays["accuracy"][best_index]),
                    "best_val_accuracy": float(arrays["val_accuracy"][best_index]),
                    "best_val_loss": float(arrays["val_loss"][best_index]),
                    "test_accuracy": validated["test_accuracy"],
                    "test_loss": validated["test_loss"],
                    "status": "success",
                }
            )
            print(f"  accuracy test = {row['test_accuracy']:.4%}")
        except Exception as exc:  # Chaque erreur reste locale à son run.
            row.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(
                f"  erreur cataloguée ({row['error_type']}) : {row['error_message']}",
                file=sys.stderr,
            )
        finally:
            row["duration_seconds"] = round(time.perf_counter() - started, 3)
            row["completed_at_utc"] = utc_now()
            histories = upsert_history(
                histories,
                history_rows(protocol_id, run, args.epochs, fit_history),
                paths.history_csv,
            )
            # Le CSV brut joue le rôle de marqueur de commit du run : l'historique
            # est donc écrit avant lui, afin qu'un succès ne puisse pas masquer un
            # historique manquant après une interruption.
            raw_results = upsert_raw_result(raw_results, row, paths.raw_csv)
            gc.collect()

    return raw_results, histories


def student_summary(values: Any, confidence_level: float) -> dict[str, float | int]:
    import numpy as np
    from scipy.stats import t as student_t

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    mean = float(np.mean(array))
    median = float(np.median(array))
    q1, q3 = (float(value) for value in np.quantile(array, [0.25, 0.75]))
    if count < 2:
        standard_deviation = standard_error = ci_low = ci_high = math.nan
    else:
        standard_deviation = float(np.std(array, ddof=1))
        standard_error = standard_deviation / math.sqrt(count)
        critical = float(
            student_t.ppf((1.0 + confidence_level) / 2.0, df=count - 1)
        )
        half_width = critical * standard_error
        ci_low = mean - half_width
        ci_high = mean + half_width
    return {
        "n": count,
        "mean": mean,
        "std": standard_deviation,
        "sem": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def result_exit_code(raw_results: Any) -> int:
    """Signaler un échec à l'appelant lorsqu'aucun run n'est exploitable."""
    if raw_results.empty:
        return 1
    return 0 if bool((raw_results["status"] == "success").any()) else 1


def aggregate_results(
    raw_results: Any,
    confidence_level: float,
    expected_seed_count: int,
    protocol_id: str,
) -> Any:
    import pandas as pd

    validate_result_frames(raw_results, None, protocol_id)
    working = raw_results.copy()
    for column in (
        "test_accuracy",
        "test_loss",
        "best_val_accuracy",
        "best_val_loss",
        "epochs_completed",
        "duration_seconds",
    ):
        working[column] = pd.to_numeric(working[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for config_id, attempt_group in working.groupby("config_id", dropna=False):
        successful = attempt_group[attempt_group["status"] == "success"]
        first = attempt_group.iloc[0]
        accuracy = student_summary(successful["test_accuracy"], confidence_level)
        loss = student_summary(successful["test_loss"], confidence_level)
        val_accuracy = student_summary(
            successful["best_val_accuracy"], confidence_level
        )
        duration = student_summary(successful["duration_seconds"], confidence_level)
        epochs = student_summary(successful["epochs_completed"], confidence_level)
        rows.append(
            {
                "protocol_id": protocol_id,
                "config_id": config_id,
                "model_mode": first["model_mode"],
                "filter_1": first["filter_1"],
                "filter_2": first["filter_2"],
                "kernel_1": first["kernel_1"],
                "kernel_2": first["kernel_2"],
                "activation_conv": first["activation_conv"],
                "dropout": first["dropout"],
                "runs_expected": expected_seed_count,
                "runs_attempted": int(attempt_group["run_id"].nunique()),
                "runs_successful": accuracy["n"],
                "runs_failed": int((attempt_group["status"] != "success").sum()),
                "is_complete": accuracy["n"] >= expected_seed_count,
                "confidence_level": confidence_level,
                "test_accuracy_mean": accuracy["mean"],
                "test_accuracy_std": accuracy["std"],
                "test_accuracy_sem": accuracy["sem"],
                "test_accuracy_ci_low": accuracy["ci_low"],
                "test_accuracy_ci_high": accuracy["ci_high"],
                "test_accuracy_median": accuracy["median"],
                "test_accuracy_q1": accuracy["q1"],
                "test_accuracy_q3": accuracy["q3"],
                "test_accuracy_min": accuracy["min"],
                "test_accuracy_max": accuracy["max"],
                "test_loss_mean": loss["mean"],
                "test_loss_std": loss["std"],
                "best_val_accuracy_mean": val_accuracy["mean"],
                "best_val_accuracy_std": val_accuracy["std"],
                "duration_seconds_mean": duration["mean"],
                "duration_seconds_median": duration["median"],
                "epochs_completed_mean": epochs["mean"],
                "epochs_completed_std": epochs["std"],
            }
        )
    aggregated = pd.DataFrame(rows, columns=AGGREGATED_COLUMNS)
    if not aggregated.empty:
        aggregated = aggregated.sort_values(
            "test_accuracy_mean", ascending=False, kind="stable", na_position="last"
        ).reset_index(drop=True)
    return aggregated


def paired_comparisons(
    raw_results: Any,
    reference_config: str | None,
    confidence_level: float,
    protocol_id: str,
) -> Any:
    import numpy as np
    import pandas as pd

    validate_result_frames(raw_results, None, protocol_id)
    if reference_config is None:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    successful = raw_results[raw_results["status"] == "success"].copy()
    successful["seed"] = validated_integer_series(
        successful["seed"], "resultats_bruts.csv.seed", 2**32 - 1
    )
    successful = successful.drop_duplicates(["config_id", "seed"], keep="first")
    successful["test_accuracy"] = pd.to_numeric(
        successful["test_accuracy"], errors="coerce"
    )
    reference = successful[successful["config_id"] == reference_config][
        ["seed", "test_accuracy"]
    ].rename(columns={"test_accuracy": "reference_accuracy"})
    rows: list[dict[str, Any]] = []
    candidate_ids = sorted(
        set(successful["config_id"].astype(str)) - {reference_config}
    )
    for candidate_id in candidate_ids:
        candidate = successful[successful["config_id"] == candidate_id][
            ["seed", "test_accuracy"]
        ].rename(columns={"test_accuracy": "candidate_accuracy"})
        paired = candidate.merge(
            reference, on="seed", how="inner", validate="one_to_one"
        ).dropna()
        differences = (
            paired["candidate_accuracy"] - paired["reference_accuracy"]
        ).to_numpy(dtype=float)
        summary = student_summary(differences, confidence_level)
        tolerance = 1e-12
        rows.append(
            {
                "protocol_id": protocol_id,
                "reference_config_id": reference_config,
                "candidate_config_id": candidate_id,
                "paired_seeds": summary["n"],
                "confidence_level": confidence_level,
                "delta_test_accuracy_mean": summary["mean"],
                "delta_test_accuracy_std": summary["std"],
                "delta_test_accuracy_sem": summary["sem"],
                "delta_test_accuracy_ci_low": summary["ci_low"],
                "delta_test_accuracy_ci_high": summary["ci_high"],
                "delta_test_accuracy_median": summary["median"],
                "candidate_wins": int(np.sum(differences > tolerance)),
                "ties": int(np.sum(np.abs(differences) <= tolerance)),
                "reference_wins": int(np.sum(differences < -tolerance)),
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def config_label(row: Any) -> str:
    if str(row.get("model_mode")) == "gabarit_keras":
        return str(row.get("config_id"))
    return (
        f"f1={int(float(row.get('filter_1')))}, "
        f"f2={int(float(row.get('filter_2')))}, "
        f"d={float(row.get('dropout')):g}, "
        f"act={row.get('activation_conv')}"
    )


def finite_aggregated_results(aggregated: Any) -> Any:
    import pandas as pd

    if aggregated.empty:
        return aggregated.copy()
    means = pd.to_numeric(aggregated["test_accuracy_mean"], errors="coerce")
    return aggregated[means.notna()].copy()


def selected_config_ids(aggregated: Any, maximum: int, reference: str | None) -> list[str]:
    finite = finite_aggregated_results(aggregated)
    if finite.empty:
        return []
    ordered = list(finite["config_id"].astype(str).head(maximum))
    if reference is not None and reference in set(finite["config_id"].astype(str)):
        if reference in ordered:
            ordered.remove(reference)
        ordered.insert(0, reference)
        ordered = ordered[:maximum]
    return ordered


def save_placeholder(output_path: Path, title: str, message: str, show: bool) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    axis.set_title(title, fontsize=14, loc="left")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_mean_ci(aggregated: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    finite = finite_aggregated_results(aggregated)
    if finite.empty:
        save_placeholder(
            output_path,
            "Accuracy moyenne et intervalle de confiance",
            "Aucun entraînement réussi.",
            show,
        )
        return
    ordered = finite.sort_values("test_accuracy_mean", ascending=True)
    labels = [config_label(row) for _, row in ordered.iterrows()]
    means = ordered["test_accuracy_mean"].to_numpy(dtype=float)
    lows = ordered["test_accuracy_ci_low"].to_numpy(dtype=float)
    highs = ordered["test_accuracy_ci_high"].to_numpy(dtype=float)
    lower_errors = np.where(np.isfinite(lows), means - lows, 0.0)
    upper_errors = np.where(np.isfinite(highs), highs - means, 0.0)
    height = max(5.5, 0.45 * len(labels) + 2.8)
    figure, axis = plt.subplots(figsize=(12, height))
    positions = np.arange(len(labels))
    axis.errorbar(
        means,
        positions,
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        color="#2457A7",
        ecolor="#7697C8",
        capsize=4,
        linewidth=1.4,
    )
    axis.set_yticks(positions, labels=labels)
    axis.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    axis.grid(axis="x", color="#D9DEE7", linewidth=0.8)
    axis.set_xlabel("Accuracy sur le jeu de test")
    confidence = float(finite["confidence_level"].iloc[0])
    axis.set_title(
        "Accuracy moyenne et intervalle de confiance de Student",
        loc="left",
        fontsize=14,
        pad=20,
    )
    axis.text(
        0.0,
        1.01,
        f"Échelle focalisée | moyenne et IC {confidence:.0%} | une ligne par configuration",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_distributions(raw_results: Any, aggregated: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    import pandas as pd

    successful = raw_results[raw_results["status"] == "success"].copy()
    successful["test_accuracy"] = pd.to_numeric(
        successful["test_accuracy"], errors="coerce"
    )
    successful = successful.dropna(subset=["test_accuracy"])
    if successful.empty:
        save_placeholder(
            output_path,
            "Distribution des accuracies",
            "Aucun entraînement réussi.",
            show,
        )
        return
    finite = finite_aggregated_results(aggregated)
    order = list(
        finite.sort_values("test_accuracy_mean", ascending=True)["config_id"].astype(str)
    )
    groups = [
        successful.loc[successful["config_id"].astype(str) == config_id, "test_accuracy"]
        .to_numpy(dtype=float)
        for config_id in order
    ]
    labels_by_id = {
        str(row["config_id"]): config_label(row) for _, row in finite.iterrows()
    }
    labels = [labels_by_id[config_id] for config_id in order]
    height = max(5.5, 0.45 * len(labels) + 2.8)
    figure, axis = plt.subplots(figsize=(12, height))
    boxes = axis.boxplot(
        groups,
        vert=False,
        patch_artist=True,
        labels=labels,
        showfliers=False,
    )
    for box in boxes["boxes"]:
        box.set(facecolor="#DCE8F7", edgecolor="#2457A7")
    for median in boxes["medians"]:
        median.set(color="#172B4D", linewidth=1.6)
    generator = np.random.default_rng(2026)
    for position, values in enumerate(groups, start=1):
        jitter = generator.uniform(-0.10, 0.10, size=len(values))
        axis.scatter(
            values,
            position + jitter,
            s=18,
            facecolors="white",
            edgecolors="#2457A7",
            linewidths=0.8,
            alpha=0.85,
            zorder=3,
        )
    axis.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    axis.grid(axis="x", color="#D9DEE7", linewidth=0.8)
    axis.set_xlabel("Accuracy sur le jeu de test")
    axis.set_title(
        "Distribution de l'accuracy entre les entraînements",
        loc="left",
        fontsize=14,
        pad=20,
    )
    axis.text(
        0.0,
        1.01,
        "Chaque cercle représente une graine ; boîte = quartiles et médiane",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_paired_trajectories(
    raw_results: Any,
    aggregated: Any,
    reference_config: str | None,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    import pandas as pd

    selected = selected_config_ids(aggregated, maximum=8, reference=reference_config)
    successful = raw_results[
        (raw_results["status"] == "success")
        & raw_results["config_id"].astype(str).isin(selected)
    ].copy()
    successful["seed"] = pd.to_numeric(successful["seed"], errors="coerce")
    successful["test_accuracy"] = pd.to_numeric(
        successful["test_accuracy"], errors="coerce"
    )
    pivot = successful.pivot_table(
        index="seed", columns="config_id", values="test_accuracy", aggfunc="first"
    ).reindex(columns=selected)
    if len(selected) < 2 or pivot.dropna(thresh=2).empty:
        save_placeholder(
            output_path,
            "Trajectoires appariées par graine",
            "Au moins deux configurations partageant des graines sont nécessaires.",
            show,
        )
        return
    labels_by_id = {
        str(row["config_id"]): config_label(row) for _, row in aggregated.iterrows()
    }
    figure, axis = plt.subplots(figsize=(max(11, 1.6 * len(selected)), 6.5))
    positions = np.arange(len(selected))
    for _, row in pivot.iterrows():
        values = row.to_numpy(dtype=float)
        if np.isfinite(values).sum() >= 2:
            axis.plot(
                positions,
                values,
                color="#9AA5B1",
                linewidth=0.8,
                alpha=0.32,
                marker="o",
                markersize=2.8,
            )
    means = pivot.mean(axis=0, skipna=True).to_numpy(dtype=float)
    axis.plot(
        positions,
        means,
        color="#2457A7",
        linewidth=2.4,
        marker="o",
        markersize=6,
        label="moyenne",
    )
    axis.set_xticks(
        positions,
        labels=[labels_by_id[config_id] for config_id in selected],
        rotation=28,
        ha="right",
    )
    axis.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.set_ylabel("Accuracy sur le jeu de test")
    axis.set_title(
        "Trajectoires appariées par graine",
        loc="left",
        fontsize=14,
        pad=20,
    )
    axis.text(
        0.0,
        1.01,
        "Une ligne grise = une même graine | huit meilleures configurations au maximum",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    axis.legend(frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_learning_curves(
    histories: Any,
    aggregated: Any,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import pandas as pd

    selected = selected_config_ids(aggregated, maximum=6, reference=None)
    working = histories[
        histories["config_id"].astype(str).isin(selected)
    ].copy()
    working["epoch"] = pd.to_numeric(working["epoch"], errors="coerce")
    working["val_accuracy"] = pd.to_numeric(
        working["val_accuracy"], errors="coerce"
    )
    reached = working[working["reached"].astype(str).str.lower().isin({"true", "1"})]
    reached = reached.dropna(subset=["epoch", "val_accuracy"])
    if reached.empty:
        save_placeholder(
            output_path,
            "Courbes moyennes d'apprentissage",
            "Aucune époque terminée n'est disponible.",
            show,
        )
        return
    labels_by_id = {
        str(row["config_id"]): config_label(row) for _, row in aggregated.iterrows()
    }
    columns = min(2, len(selected))
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2 * columns, 4.2 * rows),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()
    for axis, config_id in zip(axes_flat, selected):
        subset = reached[reached["config_id"].astype(str) == config_id]
        summary = (
            subset.groupby("epoch")["val_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        x = summary["epoch"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0.0).to_numpy(dtype=float)
        axis.plot(x, mean, color="#2457A7", linewidth=2.0)
        axis.fill_between(
            x,
            mean - std,
            mean + std,
            color="#9DB9DE",
            alpha=0.35,
            label="±1 écart-type",
        )
        minimum_count = int(summary["count"].min())
        maximum_count = int(summary["count"].max())
        axis.set_title(labels_by_id[config_id], fontsize=10.5, loc="left")
        axis.text(
            0.99,
            0.02,
            f"n/époque : {minimum_count}–{maximum_count}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#4B5563",
        )
        axis.grid(color="#D9DEE7", linewidth=0.7)
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        axis.set_xlabel("Époque réellement atteinte")
        axis.set_ylabel("Accuracy de validation")
    for axis in axes_flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle(
        "Courbes moyennes de validation et variabilité entre graines",
        x=0.02,
        ha="left",
        fontsize=15,
    )
    figure.text(
        0.02,
        0.96,
        "Top 6 au maximum | aucune valeur n'est ajoutée après l'early stopping",
        ha="left",
        fontsize=9.5,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def write_error_catalog(raw_results: Any, path: Path) -> None:
    errors = raw_results[raw_results["status"] != "success"].copy()
    columns = [
        "protocol_id",
        "run_id",
        "config_id",
        "seed",
        "error_type",
        "error_message",
        "duration_seconds",
        "completed_at_utc",
    ]
    write_dataframe_atomic(errors.reindex(columns=columns), path)


def write_readme(
    paths: StabilityPaths,
    protocol_id: str,
    reference_config: str | None,
    confidence_level: float,
) -> None:
    reference_text = reference_config or "aucune (une seule configuration)"
    content = f"""ÉTUDE DE STABILITÉ
===================

Identifiant du protocole : {protocol_id}
Configuration de référence : {reference_text}
Niveau de confiance : {confidence_level:.1%}

REPÉRAGE
--------
configuration.json
    Protocole immuable, provenance, graines demandées et chemins.
donnees/resultats_bruts.csv
    Une ligne par configuration et par graine ; écrit après chaque run.
donnees/courbes_par_epoque.csv
    Une ligne par époque demandée. Les époques non atteintes après l'early
    stopping portent reached=False et des valeurs NaN ; elles ne sont pas
    inventées lors de l'agrégation.
donnees/resultats_agreges.csv
    Moyenne, écart-type échantillonnal (ddof=1), SEM, intervalle de Student,
    médiane, quartiles, minimum et maximum.
donnees/comparaisons_pairees.csv
    Différences d'accuracy candidate-référence pour les mêmes graines.
donnees/erreurs.csv
    Runs en erreur, conservés sans interrompre toute la grille.
donnees/indices_split.npz
    Indices fixes train/validation/test utilisés par tous les runs.
graphiques/
    Figures de moyenne+IC, distributions, paires par seed et apprentissage.
catalogue_sorties.csv
    Inventaire de tous les fichiers et indication de leur présence.

REPRISE
-------
Relancer le même protocole et le même nom d'expérience ignore les run_id déjà
réussis. Les runs en erreur sont retentés. --force remplace aussi les succès.
Un protocol_id différent est refusé dans ce dossier afin d'éviter tout mélange.

INTERPRÉTATION
--------------
L'écart-type décrit la variabilité entre entraînements. L'intervalle de
confiance décrit l'incertitude sur leur moyenne. Ces deux quantités ne mesurent
pas l'incertitude d'échantillonnage du jeu de test MNIST.
"""
    paths.readme.write_text(content, encoding="utf-8")


def write_output_catalog(paths: StabilityPaths) -> None:
    import pandas as pd

    entries = [
        (paths.configuration, "configuration", "Protocole, provenance et exécution."),
        (paths.readme, "documentation", "Guide de repérage, reprise et interprétation."),
        (paths.raw_csv, "données brutes", "Une ligne par configuration et graine."),
        (paths.history_csv, "données brutes", "Valeurs par époque, NaN après arrêt."),
        (paths.aggregated_csv, "données agrégées", "Moyenne, SD, SEM, IC Student et quartiles."),
        (paths.comparisons_csv, "données agrégées", "Comparaisons appariées par graine."),
        (paths.errors_csv, "journal d'erreurs", "Runs échoués sans arrêt global."),
        (paths.split_indices, "provenance", "Indices fixes des trois partitions."),
        (paths.mean_ci_figure, "figure PNG", "Accuracy moyenne et intervalle de confiance."),
        (paths.distributions_figure, "figure PNG", "Distribution et points par run."),
        (paths.paired_figure, "figure PNG", "Trajectoires de configurations pour une même seed."),
        (paths.learning_curves_figure, "figure PNG", "Validation moyenne ± SD sans padding."),
    ]
    rows = []
    generated_at = utc_now()
    for path, kind, description in entries:
        rows.append(
            {
                "path": str(path.relative_to(paths.root)),
                "type": kind,
                "description": description,
                "exists": path.exists(),
                "catalogued_at_utc": generated_at,
            }
        )
    write_dataframe_atomic(pd.DataFrame(rows), paths.catalog)


def refresh_analysis_outputs(
    paths: StabilityPaths,
    raw_results: Any,
    histories: Any,
    protocol_id: str,
    reference_config: str | None,
    confidence_level: float,
    expected_seed_count: int,
    show: bool,
) -> tuple[Any, Any]:
    validate_result_frames(raw_results, histories, protocol_id)
    aggregated = aggregate_results(
        raw_results, confidence_level, expected_seed_count, protocol_id
    )
    comparisons = paired_comparisons(
        raw_results, reference_config, confidence_level, protocol_id
    )
    write_dataframe_atomic(aggregated, paths.aggregated_csv)
    write_dataframe_atomic(comparisons, paths.comparisons_csv)
    write_error_catalog(raw_results, paths.errors_csv)
    plot_mean_ci(aggregated, paths.mean_ci_figure, show)
    plot_distributions(raw_results, aggregated, paths.distributions_figure, show)
    plot_paired_trajectories(
        raw_results, aggregated, reference_config, paths.paired_figure, show
    )
    successful_run_ids = set(
        raw_results.loc[raw_results["status"] == "success", "run_id"].astype(str)
    )
    successful_histories = histories[
        histories["run_id"].astype(str).isin(successful_run_ids)
    ].copy()
    plot_learning_curves(
        successful_histories, aggregated, paths.learning_curves_figure, show
    )
    return aggregated, comparisons


def print_plan(
    paths: StabilityPaths,
    protocol_id: str,
    configurations: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    reference_config: str | None,
    dataset_descriptor: dict[str, Any],
    model_descriptor: dict[str, Any],
) -> None:
    print(f"Dossier de sortie : {paths.root}")
    print(f"protocol_id : {protocol_id}")
    print(f"Dataset : {dataset_descriptor['display_name']}")
    print(f"Mode modèle : {model_descriptor['mode']}")
    print(f"Configurations ({len(configurations)}) :")
    for configuration in configurations:
        print(f"  - {configuration['config_id']}")
    print(f"Graines ({len(seeds)}) : {list(seeds)}")
    print(f"Référence : {reference_config or 'aucune'}")
    print(f"Nombre total de runs : {len(configurations) * len(seeds)}")


def resolve_output_root(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = env_config.PROJECT_ROOT / expanded
    return expanded.resolve()


def load_existing_configuration(parser: argparse.ArgumentParser, paths: StabilityPaths) -> dict[str, Any]:
    if not paths.configuration.is_file():
        parser.error(f"--plot-only nécessite {paths.configuration}")
    try:
        configuration = json.loads(paths.configuration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"configuration illisible : {exc}")
    protocol_id = configuration.get("protocol_id")
    protocol = configuration.get("protocol")
    if (
        not isinstance(protocol_id, str)
        or not isinstance(protocol, dict)
        or make_protocol_id(protocol) != protocol_id
    ):
        parser.error("configuration.json contient un protocole incohérent")
    return configuration


def run_plot_only(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    paths: StabilityPaths,
) -> int:
    configuration = load_existing_configuration(parser, paths)
    if not paths.raw_csv.is_file():
        parser.error(f"--plot-only nécessite {paths.raw_csv}")
    raw_results = load_dataframe(paths.raw_csv, RAW_COLUMNS)
    histories = load_dataframe(paths.history_csv, HISTORY_COLUMNS)
    protocol_id = str(configuration.get("protocol_id", ""))
    if not protocol_id:
        parser.error("configuration.json ne contient pas de protocol_id")
    if not paths.split_indices.is_file():
        parser.error(
            "--plot-only refuse des résultats sans donnees/indices_split.npz"
        )
    try:
        load_split_indices(paths.split_indices, protocol_id)
        validate_result_frames(raw_results, histories, protocol_id)
    except ValueError as exc:
        parser.error(str(exc))
    analysis = configuration.get("analysis", {})
    execution = configuration.get("execution", {})
    reference = args.reference_config or analysis.get("reference_config_id")
    available_config_ids = set(raw_results["config_id"].dropna().astype(str))
    if reference is not None and reference not in available_config_ids:
        parser.error(
            f"configuration de référence absente des résultats : {reference}"
        )
    seeds = execution.get("all_requested_seeds", [])
    confidence = (
        args.niveau_confiance
        if args.niveau_confiance is not None
        else float(analysis.get("confidence_level", 0.95))
    )
    paths.create()
    configuration.setdefault("analysis", {})
    configuration["analysis"]["reference_config_id"] = reference
    configuration["analysis"]["confidence_level"] = confidence
    configuration.setdefault("experiment", {})
    configuration["experiment"]["last_plot_only_at_utc"] = utc_now()
    write_json_atomic(paths.configuration, configuration)
    refresh_analysis_outputs(
        paths,
        raw_results,
        histories,
        protocol_id,
        reference,
        confidence,
        len(seeds),
        args.show,
    )
    write_readme(paths, protocol_id, reference, confidence)
    write_output_catalog(paths)
    print(f"Agrégats et graphiques régénérés dans : {paths.root}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")
    apply_quick_preset(args)
    validate_args(parser, args)

    display_name = args.nom_experience.strip()
    try:
        experiment_slug = slugify(display_name)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    paths = StabilityPaths(resolve_output_root(args.sortie_racine) / experiment_slug)

    if args.plot_only:
        return run_plot_only(parser, args, paths)

    if args.niveau_confiance is None:
        args.niveau_confiance = 0.95

    dataset_descriptor = resolve_dataset_descriptor(parser, args)
    model_descriptor = resolve_model_descriptor(parser, args)
    configurations = build_configurations(args, model_descriptor)
    reference_config = resolve_reference_config(
        parser, args.reference_config, configurations
    )
    try:
        seeds = resolve_seeds(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    protocol = protocol_payload(
        args, dataset_descriptor, model_descriptor, configurations
    )
    protocol_id = make_protocol_id(protocol)
    runs = planned_runs(protocol_id, configurations, seeds)
    print_plan(
        paths,
        protocol_id,
        configurations,
        seeds,
        reference_config,
        dataset_descriptor,
        model_descriptor,
    )
    if args.dry_run:
        print("Mode --dry-run : aucun dossier créé et aucun entraînement lancé.")
        return 0
    if len(runs) >= CONFIRMATION_THRESHOLD and not args.confirmer_grande_etude:
        parser.error(
            f"ce protocole demande {len(runs)} runs. Vérifiez-le avec --dry-run "
            "puis ajoutez --confirmer-grande-etude."
        )

    configure_runtime_environment(args.deterministe)
    previous = validate_existing_experiment(parser, paths, protocol_id)
    paths.create()
    configuration = write_configuration(
        paths,
        args,
        display_name,
        protocol_id,
        protocol,
        dataset_descriptor,
        model_descriptor,
        seeds,
        reference_config,
        previous,
        argv,
    )
    write_readme(paths, protocol_id, reference_config, args.niveau_confiance)
    try:
        raw_results, histories = train_runs(
            args,
            paths,
            display_name,
            protocol_id,
            runs,
            seeds,
            dataset_descriptor,
            model_descriptor,
        )
    except ValueError as exc:
        parser.error(str(exc))
    expected_seed_count = len(
        configuration.get("execution", {}).get("all_requested_seeds", seeds)
    )
    aggregated, _ = refresh_analysis_outputs(
        paths,
        raw_results,
        histories,
        protocol_id,
        reference_config,
        args.niveau_confiance,
        expected_seed_count,
        args.show,
    )
    write_output_catalog(paths)

    success_count = int((raw_results["status"] == "success").sum())
    error_count = int((raw_results["status"] != "success").sum())
    print(f"\nRésultats : {paths.root}")
    print(f"Runs réussis catalogués : {success_count}")
    print(f"Runs en erreur catalogués : {error_count}")
    print(f"Résultats bruts : {paths.raw_csv}")
    print(f"Résultats agrégés : {paths.aggregated_csv}")
    print(f"Catalogue : {paths.catalog}")
    return result_exit_code(raw_results)


if __name__ == "__main__":
    raise SystemExit(main())
