import os
import env_config
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd


# Importation du dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Ajout du canal couleur
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

num_classes = 10
input_shape = (28, 28, 1)

# Valeurs à tester
dropout_values = [0.2, 0.3, 0.4, 0.5]
filters_values = [(4, 8), (8, 16), (16, 32), (32, 64)]

# Liste dans laquelle on va stocker les résultats
Color_map = []
COLOR_MAP_DIR = env_config.PROJECT_ROOT / "sorties" / "color_maps"
os.makedirs(COLOR_MAP_DIR, exist_ok=True)

for dropout in dropout_values:
    for filter_1, filter_2 in filters_values:

        print("--------------------------------------")
        print(f"Test avec dropout = {dropout}, filtres = ({filter_1}, {filter_2})")

        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),

            keras.layers.Conv2D(filter_1, kernel_size=(5, 5), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(filter_2, kernel_size=(5, 5), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])

        # Règle d'entraînement
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
            ],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]

        # Entraînement du modèle
        history = model.fit(
            x_train,
            y_train,
            batch_size=128,
            epochs=20,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )

        # Évaluation du modèle sur les données de test
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        print(f"Accuracy test : {test_acc}")

        # Stockage des résultats
        Color_map.append({
            "dropout": dropout,
            "filters": f"{filter_1}-{filter_2}",
            "filter_1": filter_1,
            "filter_2": filter_2,
            "test_accuracy": test_acc,
            "test_loss": test_loss
        })


# Transformation des résultats en dataframe
df_results = pd.DataFrame(Color_map)

print(df_results)


# Création d'une grille pour la color map
accuracy_grid = df_results.pivot(
    index="filters",
    columns="dropout",
    values="test_accuracy"
)

print(accuracy_grid)


# Affichage de la color map
plt.figure(figsize=(6, 6))

plt.imshow(
    accuracy_grid,
    cmap="viridis",
    aspect="auto",
    origin="lower"
)

plt.colorbar(label="Accuracy sur les données de test")

plt.xticks(
    ticks=np.arange(len(dropout_values)),
    labels=dropout_values
)

plt.yticks(
    ticks=np.arange(len(filters_values)),
    labels=[f"{f1}-{f2}" for f1, f2 in filters_values]
)

plt.xlabel("Dropout")
plt.ylabel("Nombre de filtres")
plt.title("Accuracy en fonction du dropout et du nombre de filtres")

plt.savefig(COLOR_MAP_DIR / "color_map_accuracy.png")
plt.show()
