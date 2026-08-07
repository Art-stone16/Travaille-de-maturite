#!/usr/bin/env python3
"""Explorer les hyperparamètres d'un CNN MNIST et produire des cartes lisibles.

Le script sépare volontairement les résultats bruts, les résultats agrégés et les
graphiques. Une exécution interrompue peut être reprise : les essais déjà réussis
dans ``resultats_bruts.csv`` sont ignorés, sauf si ``--force`` est demandé.

Exemple d'essai court :
    python scripts/generer_color_map.py --nom-experience essai_rapide \
        --filters-1 4,8 --filters-2 8,16 --dropouts 0.2,0.4 \
        --repetitions 1 --epochs 1 --train-limit 2000 --test-limit 500

Régénération des graphiques sans réentraîner :
    python scripts/generer_color_map.py --nom-experience essai_rapide --plot-only

Surface dense autour du meilleur dropout observé :
    python scripts/generer_color_map.py \
        --nom-experience surface_dense_dropout_04 \
        --preset-surface-dropout-04 --confirmer-grande-grille
"""

from __future__ import annotations

import argparse
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

import env_config


os.environ["KERAS_BACKEND"] = "tensorflow"

OUTPUT_ROOT = (
    env_config.PROJECT_ROOT / "sorties" / "experiences_hyperparametres"
)
SURFACE_DENSE_FILTERS_1 = (4, 8, 12, 16, 20, 24, 32)
SURFACE_DENSE_FILTERS_2 = (8, 16, 24, 32, 40, 48, 64)
SURFACE_DENSE_DROPOUT = 0.4
RAW_COLUMNS = [
    "experiment_name",
    "run_id",
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
        self.raw_csv = self.data / "resultats_bruts.csv"
        self.aggregated_csv = self.data / "resultats_agreges.csv"
        self.configuration = self.root / "configuration.json"
        self.catalog = self.root / "catalogue_sorties.csv"
        self.guide = self.root / "LISEZ_MOI.txt"
        self.heatmaps = self.figures / "heatmaps_2d_accuracy_moyenne.png"
        self.scatter_3d = self.figures / "nuage_3d_accuracy_moyenne.png"
        self.surfaces_3d = (
            self.figures / "surfaces_3d_accuracy_par_dropout.png"
        )

    def create(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    if not values or any(item < 0 or item >= 1 for item in values):
        raise argparse.ArgumentTypeError("chaque dropout doit vérifier 0 <= dropout < 1")
    return values


def comma_separated_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "utiliser des graines entières séparées par des virgules"
        ) from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("les graines doivent être positives ou nulles")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("les graines doivent être uniques")
    return seeds


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
    if not 0 < parsed < 1:
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
        "--confirmer-grande-grille",
        action="store_true",
        help="autoriser explicitement une invocation de 50 entraînements ou plus",
    )
    return parser


def resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.seeds is not None:
        return args.seeds
    return tuple(args.seed_base + offset for offset in range(args.repetitions))


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.learning_rate <= 0:
        parser.error("--learning-rate doit être strictement positif")
    if args.plot_only and args.force:
        parser.error("--plot-only et --force ne peuvent pas être utilisés ensemble")
    if args.preset_rapide and args.preset_surface_dropout_04:
        parser.error(
            "--preset-rapide et --preset-surface-dropout-04 sont incompatibles"
        )


def apply_quick_preset(args: argparse.Namespace) -> None:
    """Appliquer un petit preset déterministe, pratique pour valider le pipeline."""
    if not args.preset_rapide:
        return
    args.filters_1 = (4, 8)
    args.filters_2 = (8, 16)
    args.dropouts = (0.2, 0.4)
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


def active_preset_name(args: argparse.Namespace) -> str | None:
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
        return

    candidates = (
        env_config.PROJECT_CACHE / "keras" / "datasets" / "mnist.npz",
        Path.home() / ".keras" / "datasets" / "mnist.npz",
    )
    args.mnist_path_resolved = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        None,
    )


def make_run_id(filter_1: int, filter_2: int, dropout: float, seed: int) -> str:
    dropout_token = f"{dropout:.8g}".replace(".", "p")
    return f"f1_{filter_1}__f2_{filter_2}__dropout_{dropout_token}__seed_{seed}"


def planned_runs(args: argparse.Namespace, seeds: Sequence[int]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for filter_1 in args.filters_1:
        for filter_2 in args.filters_2:
            for dropout in args.dropouts:
                for repetition, seed in enumerate(seeds, start=1):
                    runs.append(
                        {
                            "run_id": make_run_id(filter_1, filter_2, dropout, seed),
                            "filter_1": filter_1,
                            "filter_2": filter_2,
                            "dropout": dropout,
                            "repetition": repetition,
                            "seed": seed,
                        }
                    )
    return runs


def relative_to_experiment(path: Path, paths: ExperimentPaths) -> str:
    return str(path.relative_to(paths.root))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_dataframe_atomic(dataframe: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_existing_experiment(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    paths: ExperimentPaths,
    seeds: Sequence[int],
) -> None:
    """Empêcher le mélange silencieux de protocoles dans un même CSV brut."""
    if args.plot_only or not paths.raw_csv.exists() or not paths.configuration.exists():
        return
    try:
        previous = json.loads(paths.configuration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(
            "configuration.json est illisible alors que des résultats existent : "
            f"{exc}. Utilisez un nouveau --nom-experience."
        )

    previous_input = previous.get("input", {})
    previous_grid = previous.get("grid", {})
    previous_training = previous.get("training", {})
    expected = {
        "input.train_limit": (previous_input.get("train_limit"), args.train_limit),
        "input.test_limit": (previous_input.get("test_limit"), args.test_limit),
        "input.dataset_seed": (previous_input.get("dataset_seed"), args.dataset_seed),
        "input.dataset_file": (
            previous_input.get("dataset_file"),
            (
                str(args.mnist_path_resolved)
                if args.mnist_path_resolved is not None
                else "cache Keras ou téléchargement automatique"
            ),
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
        "training.learning_rate": (
            previous_training.get("learning_rate"),
            args.learning_rate,
        ),
    }
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


def write_experiment_documentation(
    paths: ExperimentPaths,
    args: argparse.Namespace,
    display_name: str,
    slug: str,
    seeds: Sequence[int],
    number_of_runs: int,
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
        "schema_version": 1,
        "experiment": {
            "display_name": display_name,
            "folder_name": slug,
            "created_at_utc": created_at,
            "last_invocation_at_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
            "mode": "plot_only" if args.plot_only else "training",
        },
        "input": {
            "dataset": "MNIST (fichier mnist.npz au format Keras)",
            "dataset_file": (
                str(args.mnist_path_resolved)
                if args.mnist_path_resolved is not None
                else "cache Keras ou téléchargement automatique"
            ),
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
            "seeds": list(seeds),
            "runs_planned_for_this_invocation": number_of_runs,
        },
        "training": {
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "learning_rate": args.learning_rate,
            "kernel_sizes": [[5, 5], [5, 5]],
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy",
        },
        "outputs": {
            "raw_results": relative_to_experiment(paths.raw_csv, paths),
            "aggregated_results": relative_to_experiment(paths.aggregated_csv, paths),
            "heatmaps_2d": relative_to_experiment(paths.heatmaps, paths),
            "scatter_3d": relative_to_experiment(paths.scatter_3d, paths),
            "surfaces_3d": relative_to_experiment(paths.surfaces_3d, paths),
            "catalog": relative_to_experiment(paths.catalog, paths),
        },
    }
    if args.plot_only and previous:
        configuration = previous
        configuration["schema_version"] = 1
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

    guide = f"""EXPÉRIENCE : {display_name}
DOSSIER : {slug}

ENTRÉE
- MNIST est chargé par Keras, normalisé entre 0 et 1, puis remodelé en 28 x 28 x 1.
- Les paramètres exacts de la dernière invocation sont dans configuration.json.

SORTIES
- donnees/resultats_bruts.csv : une ligne par combinaison et par graine.
- donnees/resultats_agreges.csv : moyenne, écart-type et étendue par combinaison.
- graphiques/heatmaps_2d_accuracy_moyenne.png : une heatmap filter_1 x filter_2 par dropout.
- graphiques/nuage_3d_accuracy_moyenne.png : filter_1, filter_2 et dropout en axes ; accuracy en couleur.
- graphiques/surfaces_3d_accuracy_par_dropout.png : une surface filter_1 x filter_2 par dropout ;
  accuracy en hauteur et en couleur, points mesurés visibles.
- catalogue_sorties.csv : inventaire des fichiers produits et de leur rôle.

REPRISE
Relancer la même commande reprend automatiquement le CSV. Les lignes dont status=success
sont ignorées. --force les réentraîne ; --plot-only ne fait que recalculer l'agrégation et
les figures à partir du CSV brut.
"""
    paths.guide.write_text(guide, encoding="utf-8")


def load_raw_results(paths: ExperimentPaths) -> Any:
    import pandas as pd

    if not paths.raw_csv.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    dataframe = pd.read_csv(paths.raw_csv)
    for column in RAW_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None
    return dataframe[RAW_COLUMNS]


def upsert_raw_result(dataframe: Any, row: dict[str, Any], paths: ExperimentPaths) -> Any:
    import pandas as pd

    if not dataframe.empty:
        dataframe = dataframe[dataframe["run_id"].astype(str) != str(row["run_id"])]
    dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)
    dataframe = dataframe.sort_values(
        ["filter_1", "filter_2", "dropout", "seed"], kind="stable"
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


def build_model(keras: Any, filter_1: int, filter_2: int, dropout: float, args: argparse.Namespace) -> Any:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(filter_1, kernel_size=(5, 5), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(filter_2, kernel_size=(5, 5), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(10, activation="softmax"),
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
    train_samples: int,
    test_samples: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    training_samples = int(train_samples * (1.0 - args.validation_split))
    validation_samples = train_samples - training_samples
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
) -> Any:
    raw_results = load_raw_results(paths)
    successful_ids = set(
        raw_results.loc[raw_results["status"] == "success", "run_id"].astype(str)
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

    x_train, y_train, x_test, y_test = load_mnist(args)
    total_pending = len(pending_runs)

    for index, run in enumerate(pending_runs, start=1):
        print(
            f"\n[{index}/{total_pending}] filter_1={run['filter_1']}, "
            f"filter_2={run['filter_2']}, dropout={run['dropout']}, seed={run['seed']}"
        )
        row = result_row_base(
            args,
            display_name,
            run,
            len(x_train),
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
                args,
            )
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
                validation_split=args.validation_split,
                callbacks=callbacks,
                verbose=args.verbose,
                shuffle=True,
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
            raise

        raw_results = upsert_raw_result(raw_results, row, paths)

    return raw_results


def aggregate_results(raw_results: Any, paths: ExperimentPaths) -> Any:
    import pandas as pd

    if raw_results.empty:
        raise ValueError("le fichier de résultats bruts est vide")

    numeric_columns = [
        "filter_1",
        "filter_2",
        "dropout",
        "repetitions_requested",
        "best_val_accuracy",
        "best_val_loss",
        "test_accuracy",
        "test_loss",
        "duration_seconds",
    ]
    working = raw_results.copy()
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    group_columns = ["filter_1", "filter_2", "dropout"]
    attempts = (
        working.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            runs_attempted=("run_id", "nunique"),
            runs_failed=("status", lambda values: int((values != "success").sum())),
            runs_expected=("repetitions_requested", "max"),
        )
    )
    successful = working[working["status"] == "success"].dropna(
        subset=["filter_1", "filter_2", "dropout", "test_accuracy"]
    )
    if successful.empty:
        raise ValueError("aucun essai avec status=success n'est disponible")

    aggregated = (
        successful.groupby(group_columns, as_index=False)
        .agg(
            runs_completed=("run_id", "nunique"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_accuracy_min=("test_accuracy", "min"),
            test_accuracy_max=("test_accuracy", "max"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            best_val_accuracy_mean=("best_val_accuracy", "mean"),
            best_val_loss_mean=("best_val_loss", "mean"),
            duration_seconds_mean=("duration_seconds", "mean"),
        )
        .merge(attempts, on=group_columns, how="left")
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


def plot_heatmaps(aggregated: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    dropouts = sorted(aggregated["dropout"].dropna().unique())
    if not dropouts:
        raise ValueError("aucune valeur de dropout à représenter")

    column_count = min(3, len(dropouts))
    row_count = math.ceil(len(dropouts) / column_count)
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

    for axis, dropout in zip(axes_flat, dropouts):
        subset = aggregated[aggregated["dropout"] == dropout]
        grid = subset.pivot(
            index="filter_1", columns="filter_2", values="test_accuracy_mean"
        ).sort_index(ascending=True).sort_index(axis=1, ascending=True)
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
        axis.set_title(f"Dropout = {dropout:g}")
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

    for axis in axes_flat[len(dropouts) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Accuracy moyenne selon filter_1, filter_2 et le dropout",
        fontsize=15,
        y=1.01,
    )
    if last_image is not None:
        color_bar = figure.colorbar(
            last_image,
            ax=axes_flat[: len(dropouts)],
            fraction=0.025,
            pad=0.03,
        )
        color_bar.set_label("Accuracy moyenne sur le jeu de test")
    figure.subplots_adjust(top=0.90, right=0.90, hspace=0.35, wspace=0.30)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_scatter_3d(aggregated: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
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
        scatter = axis.scatter(
            x_values,
            y_values,
            z_values,
            c=accuracies,
            cmap=color_map,
            norm=normalisation,
            s=105,
            marker="o",
            edgecolors="#1f2933",
            linewidths=0.65,
            alpha=0.92,
            depthshade=False,
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
            f"f2={int(ligne['filter_2'])}, dropout={ligne['dropout']:g}  "
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


def plot_surfaces_3d(aggregated: Any, output_path: Path, show: bool) -> None:
    """Afficher un paysage d'accuracy filter_1 x filter_2 pour chaque dropout."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    dropouts = sorted(aggregated["dropout"].dropna().unique())
    if not dropouts:
        raise ValueError("aucune valeur de dropout à représenter en surface")

    column_count = min(3, len(dropouts))
    row_count = math.ceil(len(dropouts) / column_count)
    single_row = row_count == 1
    figure = plt.figure(
        figsize=(
            5.7 * column_count + 0.9,
            4.6 * row_count + (2.4 if single_row else 1.4),
        ),
        facecolor="white",
    )
    axes = []
    for index in range(row_count * column_count):
        if index < len(dropouts):
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

    for axis, dropout in zip(axes, dropouts):
        subset = aggregated[aggregated["dropout"] == dropout]
        grid = subset.pivot(
            index="filter_1",
            columns="filter_2",
            values="test_accuracy_mean",
        ).sort_index(ascending=True).sort_index(axis=1, ascending=True)
        filter_1_levels = grid.index.to_numpy(dtype=float)
        filter_2_levels = grid.columns.to_numpy(dtype=float)
        matrix = grid.to_numpy(dtype=float)
        standard_deviation_grid = subset.pivot(
            index="filter_1",
            columns="filter_2",
            values="test_accuracy_std",
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

        axis.set_title(
            f"Dropout = {dropout:g}\n"
            f"max mesuré : {best_accuracy:.2%} "
            f"(f1={int(best_filter_1)}, f2={int(best_filter_2)})",
            fontsize=10.5,
            pad=8,
        )
        axis.set_xlabel("Filtres conv. 1", labelpad=7)
        axis.set_ylabel("Filtres conv. 2", labelpad=7)
        axis.set_zlabel("Accuracy moyenne", labelpad=7)
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
        axis.tick_params(labelsize=8, pad=0)
        axis.view_init(elev=28, azim=-55)
        axis.set_box_aspect((1.0, 1.0, 0.85))
        axis.grid(True, color="#d9dee3", linewidth=0.6)
        for cartesian_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            cartesian_axis.pane.set_facecolor((0.98, 0.98, 0.98, 0.75))
            cartesian_axis.pane.set_edgecolor("#d9dee3")

    unused_axes = axes[len(dropouts) :]
    for axis in unused_axes:
        axis.set_axis_off()
    if unused_axes:
        uncertainty_explanation = (
            "• Les traits verticaux montrent ±1 écart-type.\n"
            if has_uncertainty
            else "• Écart-type indisponible avec une seule répétition.\n"
        )
        global_best_index = int(aggregated["test_accuracy_mean"].idxmax())
        global_best = aggregated.loc[global_best_index]
        unused_axes[0].text(
            0.08,
            0.86,
            "Comment lire la figure",
            transform=unused_axes[0].transAxes,
            fontsize=13,
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
            fontsize=10.5,
            color="#45525e",
            linespacing=1.55,
            va="top",
        )
        unused_axes[0].text(
            0.08,
            0.31,
            (
                "Meilleur résultat global\n"
                f"f1={int(global_best['filter_1'])}, "
                f"f2={int(global_best['filter_2'])}, "
                f"dropout={global_best['dropout']:g}\n"
                f"accuracy = {global_best['test_accuracy_mean']:.2%}"
            ),
            transform=unused_axes[0].transAxes,
            fontsize=11,
            fontweight="bold",
            color="#1f2933",
            linespacing=1.4,
            va="top",
        )

    figure.suptitle(
        "Surfaces 3D de l'accuracy moyenne par dropout",
        fontsize=17,
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
        fontsize=10.3,
        color="#45525e",
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
        fontsize=9.7,
        color="#5d6974",
    )

    color_reference = cm.ScalarMappable(norm=normalisation, cmap=color_map)
    color_reference.set_array([])
    color_axis = figure.add_axes([0.92, 0.25, 0.015, 0.52])
    color_bar = figure.colorbar(color_reference, cax=color_axis)
    color_bar.set_label("Accuracy moyenne sur le jeu de test", labelpad=10)
    color_bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

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
        fontsize=9.5,
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.89,
        top=0.775 if single_row else 0.875,
        bottom=0.085,
        wspace=0.02,
        hspace=0.12,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def write_output_catalog(paths: ExperimentPaths) -> None:
    import pandas as pd

    generated_at = utc_now()
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
        {
            "path": relative_to_experiment(paths.surfaces_3d, paths),
            "type": "figure PNG",
            "description": (
                "Surfaces 3D filter_1 x filter_2 de l'accuracy moyenne, "
                "une facette par dropout avec mesures et écarts-types."
            ),
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
) -> None:
    run_count = len(list(runs))
    print(f"Nom de dossier : {slug}")
    print(f"Dossier de sortie : {paths.root}")
    if args.preset_surface_dropout_04:
        print("Preset : surface dense 7 x 7, dropout=0.4 uniquement")
    if args.plot_only:
        print(
            "Mode --plot-only : aucune grille ne sera entraînée ; "
            "le CSV brut existant sera relu."
        )
        return
    print(f"filter_1 : {list(args.filters_1)}")
    print(f"filter_2 : {list(args.filters_2)}")
    print(f"dropout : {list(args.dropouts)}")
    print(f"graines : {list(seeds)}")
    print(
        "Source MNIST : "
        + (
            str(args.mnist_path_resolved)
            if args.mnist_path_resolved is not None
            else "cache Keras / téléchargement automatique"
        )
    )
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
    resolve_mnist_path(parser, args)

    display_name = args.nom_experience.strip()
    try:
        slug = slugify_experiment_name(display_name)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    paths = ExperimentPaths(slug)
    seeds = resolve_seeds(args)
    runs = planned_runs(args, seeds)

    print_plan(args, paths, slug, seeds, runs)
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
    if args.plot_only and not paths.raw_csv.exists():
        parser.error(
            f"--plot-only nécessite le fichier existant : {paths.raw_csv}"
        )
    validate_existing_experiment(parser, args, paths, seeds)

    paths.create()
    write_experiment_documentation(
        paths,
        args,
        display_name,
        slug,
        seeds,
        len(runs),
    )

    if args.plot_only:
        raw_results = load_raw_results(paths)
    else:
        raw_results = train_grid(args, paths, display_name, runs, seeds)

    try:
        aggregated = aggregate_results(raw_results, paths)
        plot_heatmaps(aggregated, paths.heatmaps, args.show)
        plot_scatter_3d(aggregated, paths.scatter_3d, args.show)
        plot_surfaces_3d(aggregated, paths.surfaces_3d, args.show)
    except ValueError as exc:
        parser.error(str(exc))

    write_output_catalog(paths)
    print(f"\nRésultats bruts : {paths.raw_csv}")
    print(f"Résultats agrégés : {paths.aggregated_csv}")
    print(f"Heatmaps 2D : {paths.heatmaps}")
    print(f"Nuage 3D : {paths.scatter_3d}")
    print(f"Surfaces 3D : {paths.surfaces_3d}")
    print(f"Catalogue : {paths.catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
