"""Protocole commun aux expériences ReLU ×2 et ×10, sans augmentation.

La validation MNIST est séparée avant toute répétition de Cascade. La version
historique séparait les copies après mélange et contaminait la validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config
from .datasets import charger_cascade_entrainement, chemin_dataset_cascade

GRAINE = 42
VALIDATION_SPLIT = 0.15
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 2


def sha256(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def empreintes_images(images: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in images}


def repeter_cascade(x: np.ndarray, y: np.ndarray, facteur: int):
    """Même concaténation par blocs que le ×2 : pixels et labels inchangés."""
    if not isinstance(facteur, int) or facteur < 1:
        raise ValueError("Le facteur doit être un entier positif.")
    if x.ndim != 4 or x.shape[1:] != (28, 28, 1) or x.dtype != np.float32:
        raise ValueError("Cascade doit être un tableau float32 (N, 28, 28, 1).")
    if y.shape != (len(x),) or not len(x):
        raise ValueError("Images et étiquettes Cascade non alignées ou vides.")
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Les pixels Cascade doivent être finis et compris entre 0 et 1.")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() > 9:
        raise ValueError("Les labels Cascade doivent être des entiers de 0 à 9.")
    images = np.concatenate([x] * facteur, axis=0)
    labels = np.concatenate([y] * facteur, axis=0)
    for bloc in range(facteur):
        debut = bloc * len(x)
        if not (np.array_equal(images[debut:debut + len(x)], x)
                and np.array_equal(labels[debut:debut + len(x)], y)):
            raise ValueError("Une copie Cascade diffère de son original.")
    return images, labels


def preparer_donnees(x_mnist, y_mnist, x_cascade, y_cascade, facteur):
    """Réserve 15 % de MNIST, puis ajoute toutes les copies au train seulement."""
    if x_mnist.shape[1:] != (28, 28, 1) or y_mnist.shape != (len(x_mnist),):
        raise ValueError("MNIST : images et labels non alignés.")
    indices = np.random.default_rng(GRAINE).permutation(len(x_mnist))
    limite = int(len(indices) * (1 - VALIDATION_SPLIT))
    train, validation = indices[:limite], indices[limite:]
    x_repete, y_repete = repeter_cascade(x_cascade, y_cascade, facteur)
    x_train = np.concatenate((x_mnist[train], x_repete), axis=0)
    y_train = np.concatenate((y_mnist[train], y_repete), axis=0)
    melange = np.random.default_rng(GRAINE).permutation(len(x_train))
    return (x_train[melange], y_train[melange], x_mnist[validation],
            y_mnist[validation], train, validation, melange)


def construire_modele(keras):
    modele = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.6),
        keras.layers.Dense(10, activation="softmax"),
    ])
    modele.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return modele


def entrainer(facteur: int, nom: str, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sortie", type=Path, default=config.MODELES_VALIDES / nom)
    parser.add_argument("--courbes", type=Path,
                        default=config.SORTIES_ENTRAINEMENTS_MANUELS / "courbes")
    parser.add_argument("--mnist-path", type=Path,
                        default=config.PROJECT_CACHE / "keras/datasets/mnist.npz")
    args = parser.parse_args(argv)

    # Fixer Python, NumPy, TensorFlow, l'initialisation et le dropout ensemble.
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(GRAINE)
    tf.config.experimental.enable_op_determinism()

    if not args.mnist_path.is_file():
        raise FileNotFoundError(f"MNIST local introuvable : {args.mnist_path}")
    with np.load(args.mnist_path, allow_pickle=False) as donnees:
        x_mnist = donnees["x_train"].astype("float32")[..., None] / 255
        y_mnist = donnees["y_train"]
        x_test = donnees["x_test"].astype("float32")[..., None] / 255
        y_test = donnees["y_test"]
    dossier_cascade = chemin_dataset_cascade()
    x_cascade, y_cascade = charger_cascade_entrainement(dossier_cascade)
    (x_train, y_train, x_val, y_val, train, validation, melange) = preparer_donnees(
        x_mnist, y_mnist, x_cascade, y_cascade, facteur)
    h_cascade = empreintes_images(x_cascade)
    fuite_val = len(h_cascade & empreintes_images(x_val))
    fuite_test = len(h_cascade & empreintes_images(x_test))
    if fuite_val or fuite_test:
        raise ValueError(f"Fuite Cascade : validation={fuite_val}, test={fuite_test}")
    if len(x_test) != len(y_test):
        raise ValueError("Images et labels de test non alignés.")
    args.sortie.mkdir(parents=True, exist_ok=True)
    args.courbes.mkdir(parents=True, exist_ok=True)
    if any((args.sortie / f).exists() for f in ("best_model.keras", "final_model.keras")):
        raise FileExistsError("La destination contient déjà un modèle ; utiliser --sortie.")
    np.savez_compressed(args.sortie / "indices_split.npz", mnist_train=train,
                        mnist_validation=validation, melange_train=melange)
    effectifs = {
        "mnist_source_train": len(x_mnist), "mnist_train": len(train),
        "mnist_validation": len(validation), "mnist_test": len(x_test),
        "cascade_originaux": len(x_cascade),
        "cascade_copies_exactes": len(x_cascade) * (facteur - 1),
        "cascade_total": len(x_cascade) * facteur,
        "train_total": len(x_train), "validation_total": len(x_val),
        "cascade_validation": 0, "cascade_test": 0,
        "repartition_cascade_originaux": np.bincount(y_cascade, minlength=10).tolist(),
        "repartition_cascade_train": (np.bincount(y_cascade, minlength=10) * facteur).tolist(),
    }
    print(json.dumps(effectifs, indent=2), flush=True)
    modele = construire_modele(keras)
    modele.summary()
    history = modele.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(x_val, y_val), shuffle=True, verbose=2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(str(args.sortie / "best_model.keras"),
                                            monitor="val_loss", save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                          restore_best_weights=False),
            keras.callbacks.CSVLogger(str(args.sortie / "courbes_par_epoque.csv")),
        ],
    )
    score_final = modele.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    modele.save(args.sortie / "final_model.keras")
    meilleur = keras.models.load_model(args.sortie / "best_model.keras")
    score_best = meilleur.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    verification = {}
    for fichier, original in (("best_model.keras", meilleur), ("final_model.keras", modele)):
        recharge = keras.models.load_model(args.sortie / fichier)
        predictions = recharge.predict(x_test[:32], verbose=0)
        np.testing.assert_allclose(predictions, original.predict(x_test[:32], verbose=0),
                                   rtol=1e-6, atol=1e-7)
        if predictions.shape != (32, 10) or not np.isfinite(predictions).all():
            raise ValueError("Prédictions invalides après rechargement.")
        np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-5)
        verification[fichier] = {"sha256": sha256(args.sortie / fichier),
                                  "forme_predictions": list(predictions.shape),
                                  "predictions_finies": True, "rechargement_identique": True}
    metadata = {
        "protocole": "relu_cascade_copies_exactes_validation_mnist_v2",
        "modele_reference": "best_relu_2xcascade (script corrigé, poids historiques conservés)",
        "nom_modele": nom, "facteur_repetition": facteur, "augmentation": None,
        **effectifs,
        "reglages": {"graine": GRAINE, "deterministic_ops": True,
                     "validation_split_mnist": VALIDATION_SPLIT,
                     "batch_size": BATCH_SIZE, "epochs_max": EPOCHS,
                     "early_stopping": {"monitor": "val_loss", "patience": PATIENCE,
                                        "restore_best_weights": False},
                     "optimizer": "Adam", "learning_rate": 0.001,
                     "loss": "SparseCategoricalCrossentropy", "shuffle": True},
        "sources": {"mnist": {"chemin": str(args.mnist_path), "sha256": sha256(args.mnist_path)},
                    "cascade": {f.name: sha256(f) for f in dossier_cascade.glob("*.npy")},
                    "routine_sha256": sha256(Path(__file__)),
                    "indices_split_sha256": sha256(args.sortie / "indices_split.npz")},
        "controle_fuite": {"cascade_validation_empreintes_communes": fuite_val,
                           "cascade_test_empreintes_communes": fuite_test},
        "epochs_realises": len(history.history["loss"]),
        "meilleure_epoque": int(np.argmin(history.history["val_loss"])) + 1,
        "mnist_test_loss": float(score_best[0]), "mnist_test_accuracy": float(score_best[1]),
        "etat_evalue": "best_model.keras",
        "mnist_test_final": {"loss": float(score_final[0]), "accuracy": float(score_final[1])},
        "verification_rechargement": verification,
        "versions": {"keras": keras.__version__, "tensorflow": tf.__version__, "numpy": np.__version__},
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
    }
    (args.sortie / "entrainement.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epoques = range(1, len(history.history["loss"]) + 1)
    for ax, train_key, val_key, titre in zip(axes, ("loss", "acc"), ("val_loss", "val_acc"),
                                           ("Fonction de coût", "Précision")):
        ax.plot(epoques, history.history[train_key], label="Entraînement MNIST + Cascade")
        ax.plot(epoques, history.history[val_key], label="Validation MNIST indépendante")
        ax.axvline(metadata["meilleure_epoque"], color="grey", linestyle=":", label="Checkpoint retenu")
        ax.set(title=titre, xlabel="Époque")
        ax.legend(fontsize=8)
    fig.suptitle(f"{nom} — copies exactes, graine {GRAINE}")
    fig.tight_layout()
    fig.savefig(args.courbes / f"{nom}_training_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"mnist_best": score_best, "mnist_final": score_final,
                      "meilleure_epoque": metadata["meilleure_epoque"]}), flush=True)
    return args.sortie
