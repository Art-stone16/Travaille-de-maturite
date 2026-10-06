
# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401

#Importation d'OS pour définir quel moteur Keras utilise. 
import os
from reconnaissance_chiffres import config as env_config
from reconnaissance_chiffres.datasets import charger_cascade_entrainement
# Définition du moteur pour Keras, important de faire avant l'importation de Keras. 
os.environ["KERAS_BACKEND"] = "tensorflow"
# Importation de numpy pour manipuler mes données. 
import numpy as np
import keras
import json
import cv2
# Importation de matplotlib pour la création de graphiques. 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Importation du dataset. 
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
# Fait en sorte que les pixels des images soient entre 0 et 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Fait en sorte que les images soient en 28 x 28 x 1
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Ajout des chiffres réels extraits et validés depuis les feuilles Cascade Top-N.
CASCADE_DATASET_DIR = (
    env_config.PROJECT_ROOT
    / "donnees"
    / "preparees" / "cascade"
    / "cascade_top_n_v1"
    / "dataset_numpy"
)
x_train_cascade, y_train_cascade = charger_cascade_entrainement(CASCADE_DATASET_DIR)
# Neuf transformations par image, en plus de chaque original : 10 fois le lot.
generateur = np.random.default_rng(42)
variantes = np.empty((len(x_train_cascade) * 9, 28, 28, 1), dtype=np.float32)
for index, image in enumerate(x_train_cascade):
    for copie in range(9):
        angle = float(generateur.uniform(-8, 8))
        dx, dy = generateur.uniform(-1.5, 1.5, size=2)
        transformation = cv2.getRotationMatrix2D((13.5, 13.5), angle, 1.0)
        transformation[:, 2] += (dx, dy)
        variantes[index * 9 + copie, :, :, 0] = cv2.warpAffine(
            image[:, :, 0], transformation, (28, 28),
            flags=cv2.INTER_LINEAR, borderValue=0,
        )
x_cascade_10x = np.concatenate((x_train_cascade, variantes), axis=0)
y_cascade_10x = np.concatenate((y_train_cascade, np.repeat(y_train_cascade, 9)), axis=0)
x_train = np.concatenate((x_train, x_cascade_10x), axis=0)
y_train = np.concatenate((y_train, y_cascade_10x), axis=0)

# Keras prélève la validation à la fin du tableau : on mélange donc MNIST et
# Cascade avant l'entraînement pour que les deux sources soient représentées.
indices_melanges = np.random.default_rng(42).permutation(len(x_train))
x_train = x_train[indices_melanges]
y_train = y_train[indices_melanges]

print(f"{len(x_cascade_10x)} chiffres Cascade ajoutés à l'entraînement ({len(x_train_cascade)} originaux et {len(variantes)} variantes)")
# Visualisation des données importées
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Nombre de possibilités en output. ( je l'utilise dans ma dernière couche de mon modèle)
num_classes = 10
# Format d'entré
input_shape = (28, 28, 1)
# Création de l'architecture
model = keras.Sequential(
    [   # Input
        keras.layers.Input(shape=input_shape),
        # Première couche de convolution 
        keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        # BatchNormalization va réduire les trop grands écarts entre les valeurs de ma convolution. 
        keras.layers.BatchNormalization(),
        # MaxPooling va nous permettre de se concentrer sur ce qui compte
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Deuxième couche de convolution 
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Dépliemment de notre matrice
        keras.layers.Flatten(),
        # Désactivaiton d'un partie de nos neurones
        keras.layers.Dropout(0.6),
        # Output
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Visualisation de notre modèle 
model.summary()

# Création de l'apprentissage
model.compile(
    # Calcule la fonction loss
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # Optimise les paramètres
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    # Permet de voir la précision. 
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
# Nom.
MODEL_NAME = "best_relu_10xcascade"
MODEL_DIR = env_config.PROJECT_ROOT / "modeles" / "actifs"
GRAPH_DIR = env_config.SORTIES_ENTRAINEMENTS_MANUELS / "courbes"
os.makedirs(MODEL_DIR / MODEL_NAME, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)



callbacks = [   
    keras.callbacks.ModelCheckpoint(filepath=str(MODEL_DIR / MODEL_NAME / "best_model.keras"), save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]
# Application de toutes les règles prédéfinies précédemment
history = model.fit(
    # Images d'entraînement
    x_train,
    # Images de test
    y_train,
    # Bouchée
    batch_size=128,
    # Nombre de visualisations maximale.
    epochs=30,
    # Met de côté 15% des images d'entraînement pour tester le modèle avec des images qu'il ne connaît pas.
    validation_split=0.15,
    # Exécute des actions après chaque epoch
    callbacks=callbacks,
)

score = model.evaluate(x_test, y_test, verbose=0)
model.save(MODEL_DIR / MODEL_NAME / "final_model.keras")
(MODEL_DIR / MODEL_NAME / "entrainement.json").write_text(json.dumps({
    "modele_reference": "Best_relu_cascade_V2",
    "mnist_train": 60000,
    "cascade_originaux": int(len(x_train_cascade)),
    "cascade_variantes_augmentees": int(len(variantes)),
    "cascade_total": int(len(x_cascade_10x)),
    "augmentation": {"rotation_max_deg": 8, "translation_max_px": 1.5, "seed": 42},
    "validation_split": 0.15,
    "mnist_test_loss": float(score[0]),
    "mnist_test_accuracy": float(score[1]),
    "history": {key: [float(value) for value in values] for key, values in history.history.items()},
}, indent=2) + "\n", encoding="utf-8")


# --- Visualisation à la fin ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='Train Acc')
plt.plot(history.history['val_acc'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_DIR / f"{MODEL_NAME}_training_curves.png")
plt.close()

# Visualisation de son score final avec les données d'entraînement.
print(score)
print(history.history)
