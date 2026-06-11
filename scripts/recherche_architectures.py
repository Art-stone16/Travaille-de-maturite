import os
import env_config
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
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
filters_values = [(4, 8), (8,16), (16, 32), (32, 64)]
kernel_value = [
    ((2,2), (2,2)),
    ((3,3), (2,2)),
    ((4,4), (3,3)),
    ((5,5), (3,3)),
    ((5,5), (4,4))
]

activation_value = [
    ("relu", "relu", "softmax"),
    ("relu", "softmax", "softmax"),
    ("softmax", "softmax", "softmax"), 
]

# Liste qui va contenir les résultats
results = []

# Création des dossiers
MODEL_DIR = env_config.PROJECT_ROOT / "modeles" / "recherche_architectures"
RESULTS_DIR = env_config.PROJECT_ROOT / "sorties" / "resultats"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
#boucle pour tester plusieurs architecture
for dropout in dropout_values:
    for filtres_1, filtres_2 in filters_values:
        for kernel_1, kernel_2 in kernel_value:
            for activation_1, activation_2, activation_3 in activation_value:

                MODEL_NAME = f"conv_{filtres_1}{activation_1}-{filtres_2}{activation_2}_dropout_{dropout}{activation_3}_kernel_{kernel_1}_{kernel_2}"

                print(f"\n--- Entraînement : {MODEL_NAME} ---")

                model = keras.Sequential([
                    keras.layers.Input(shape=input_shape),

                    keras.layers.Conv2D(filtres_1, kernel_size=kernel_1, activation=activation_1),
                    keras.layers.BatchNormalization(),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),

                    keras.layers.Conv2D(filtres_2, kernel_size=kernel_2, activation=activation_2),
                    keras.layers.BatchNormalization(),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),

                    keras.layers.Flatten(),
                    keras.layers.Dropout(dropout),
                    keras.layers.Dense(num_classes, activation=activation_3),
                ])
                # Règle d'entrainement
                model.compile(
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    metrics=[
                        keras.metrics.SparseCategoricalAccuracy(name="acc"),
                    ],
                )
                # Autre règle d'entrainement
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=2,
                        restore_best_weights=True
                    )
                ]
                #application de toutes les règles
                history = model.fit(
                    x_train,
                    y_train,
                    batch_size=128,
                    epochs=20,
                    validation_split=0.15,
                    callbacks=callbacks,
                    verbose=1
                )

                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

                # Récupérer les meilleures valeurs pendant l'entraînement
                best_val_loss = min(history.history["val_loss"])
                best_val_acc = max(history.history["val_acc"])

                # Nombre réel d'epochs effectuées
                epochs_done = len(history.history["loss"])

                results.append({
                    "model_name": MODEL_NAME,
                    "dropout": dropout,
                    "filtres_1": filtres_1,
                    "filtres_2": filtres_2,
                    "kernel_1" : kernel_1,
                    "kernel_2" : kernel_2,
                    "activation_1": activation_1,
                    "activation_2": activation_2,
                    "activation_3": activation_3,
                    "epochs_effectuees": epochs_done,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                })

                model.save(MODEL_DIR / f"{MODEL_NAME}.keras")

# Création du DataFrame final
df_results = pd.DataFrame(results)

print(df_results)

# Sauvegarde en CSV
df_results.to_csv(RESULTS_DIR / "comparaison_modeles.csv", index=False)
