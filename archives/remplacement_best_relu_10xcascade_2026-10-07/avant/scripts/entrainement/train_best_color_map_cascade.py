"""Entraîne Best_COLOR_MAP_cascade avec l'architecture exacte de Best_COLOR_MAP.

Le complément Cascade validé est ajouté exclusivement au jeu d'entraînement.
La validation et le test restent composés uniquement d'images MNIST.
"""

from __future__ import annotations

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401


import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from reconnaissance_chiffres import config as env_config
from reconnaissance_chiffres.datasets import charger_cascade_entrainement


os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np


MODEL_NAME = "Best_COLOR_MAP_cascade"
REFERENCE_MODEL = (
    env_config.PROJECT_ROOT
    / "modeles"
    / "actifs"
    / "Best_COLOR_MAP"
    / "best_model.keras"
)
OUTPUT_DIR = (
    env_config.PROJECT_ROOT / "modeles" / "actifs" / MODEL_NAME
)
CASCADE_DIR = (
    env_config.PROJECT_ROOT
    / "donnees"
    / "preparees" / "cascade"
    / "cascade_top_n_v1"
    / "dataset_numpy"
)
MNIST_PATH = env_config.PROJECT_ROOT / ".cache" / "keras" / "datasets" / "mnist.npz"
CURVE_PATH = (
    env_config.SORTIES_ENTRAINEMENTS_MANUELS
    / "courbes"
    / f"{MODEL_NAME}_training_curves.png"
)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 2
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 1e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture_signature(model: keras.Model) -> list[dict[str, object]]:
    """Retourne seulement les éléments structurels, jamais les poids."""
    signature: list[dict[str, object]] = []
    for layer in model.layers:
        config = layer.get_config()
        signature.append(
            {
                "type": layer.__class__.__name__,
                "config": {
                    key: config[key]
                    for key in (
                        "filters",
                        "kernel_size",
                        "strides",
                        "padding",
                        "activation",
                        "pool_size",
                        "rate",
                        "units",
                    )
                    if key in config
                },
            }
        )
    return signature


def load_datasets() -> tuple[np.ndarray, ...]:
    if not MNIST_PATH.is_file():
        raise FileNotFoundError(f"Dataset MNIST local introuvable : {MNIST_PATH}")
    with np.load(MNIST_PATH, allow_pickle=False) as dataset:
        x_mnist = dataset["x_train"].astype("float32") / 255.0
        y_mnist = dataset["y_train"]
        x_test = dataset["x_test"].astype("float32") / 255.0
        y_test = dataset["y_test"]
    x_mnist = np.expand_dims(x_mnist, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    split_at = int(len(x_mnist) * (1.0 - VALIDATION_SPLIT))
    x_train, y_train = x_mnist[:split_at], y_mnist[:split_at]
    x_validation, y_validation = x_mnist[split_at:], y_mnist[split_at:]

    x_cascade, y_cascade = charger_cascade_entrainement(CASCADE_DIR)
    ids_cascade = np.load(CASCADE_DIR / "ids_train_cascade.npy", allow_pickle=False)
    if x_cascade.shape[1:] != (28, 28, 1):
        raise ValueError(f"Forme Cascade invalide : {x_cascade.shape}")
    if len(x_cascade) != len(y_cascade) or len(x_cascade) != len(ids_cascade):
        raise ValueError("Les tableaux Cascade n'ont pas la même longueur")
    if len(set(map(str, ids_cascade.tolist()))) != len(ids_cascade):
        raise ValueError("Les identifiants Cascade ne sont pas uniques")
    if not np.isfinite(x_cascade).all() or x_cascade.min() < 0 or x_cascade.max() > 1:
        raise ValueError("Les pixels Cascade doivent être compris entre 0 et 1")

    x_train = np.concatenate((x_train, x_cascade.astype("float32")), axis=0)
    y_train = np.concatenate((y_train, y_cascade), axis=0)
    permutation = np.random.default_rng(SEED).permutation(len(x_train))
    return (
        x_train[permutation],
        y_train[permutation],
        x_validation,
        y_validation,
        x_test,
        y_test,
        x_cascade,
        y_cascade,
    )


def main() -> None:
    keras.utils.set_random_seed(SEED)
    reference = keras.models.load_model(REFERENCE_MODEL)
    reference_signature = architecture_signature(reference)

    # clone_model copie la structure et sa configuration, mais crée de nouveaux poids.
    model = keras.models.clone_model(reference)
    if architecture_signature(model) != reference_signature:
        raise RuntimeError("L'architecture clonée diffère de Best_COLOR_MAP")
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.summary()

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        x_cascade,
        y_cascade,
    ) = load_datasets()
    print(
        f"Données : {len(x_train)} entraînement (51000 MNIST + "
        f"{len(x_cascade)} Cascade), {len(x_validation)} validation MNIST, "
        f"{len(x_test)} test MNIST."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_path = OUTPUT_DIR / "best_model.keras"
    final_path = OUTPUT_DIR / "final_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE),
    ]

    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    duration = time.perf_counter() - started
    model.save(final_path)

    best_model = keras.models.load_model(best_path)
    if architecture_signature(best_model) != reference_signature:
        raise RuntimeError("Le modèle sauvegardé ne conserve pas l'architecture de référence")
    best_score = best_model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    final_score = model.evaluate(x_test, y_test, verbose=0, return_dict=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history.history["acc"], label="Train Acc")
    axes[1].plot(history.history["val_acc"], label="Val Acc")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(CURVE_PATH, dpi=160)
    plt.close(figure)

    labels, counts = np.unique(y_cascade.astype(int), return_counts=True)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    report = {
        "model_name": MODEL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_model": str(REFERENCE_MODEL),
        "reference_model_sha256": sha256(REFERENCE_MODEL),
        "architecture_identical_to_reference": True,
        "architecture": reference_signature,
        "model_parameters": int(model.count_params()),
        "training": {
            "seed": SEED,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "SparseCategoricalCrossentropy",
            "batch_size": BATCH_SIZE,
            "epochs_requested": EPOCHS,
            "epochs_completed": len(history.history["loss"]),
            "best_epoch_by_val_loss": best_epoch,
            "early_stopping_patience": PATIENCE,
            "duration_seconds": round(duration, 3),
        },
        "datasets": {
            "mnist_sha256": sha256(MNIST_PATH),
            "mnist_train": 51000,
            "mnist_validation": 9000,
            "mnist_test": 10000,
            "cascade_train": int(len(x_cascade)),
            "cascade_class_distribution": {
                str(int(label)): int(count) for label, count in zip(labels, counts)
            },
            "cascade_x_sha256": sha256(CASCADE_DIR / "x_train_cascade.npy"),
            "cascade_y_sha256": sha256(CASCADE_DIR / "y_train_cascade.npy"),
            "cascade_ids_sha256": sha256(CASCADE_DIR / "ids_train_cascade.npy"),
            "policy": "Cascade uniquement dans l'entraînement",
        },
        "best_model_mnist_test": {key: float(value) for key, value in best_score.items()},
        "final_model_mnist_test": {key: float(value) for key, value in final_score.items()},
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "files": {
            "best_model": str(best_path),
            "final_model": str(final_path),
            "training_curves": str(CURVE_PATH),
        },
    }
    (OUTPUT_DIR / "entrainement.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Meilleure epoch : {best_epoch}")
    print(f"Accuracy MNIST du meilleur modèle : {best_score['acc']:.6f}")
    print(f"Modèle enregistré dans : {best_path}")


if __name__ == "__main__":
    main()
