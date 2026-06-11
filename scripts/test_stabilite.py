import os
import env_config
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
import matplotlib.pyplot as plt

OUTPUT_DIR = env_config.PROJECT_ROOT / "sorties" / "tests_condition_reelle"
os.makedirs(OUTPUT_DIR, exist_ok=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

epochs = 30
n_runs = 100  # 1000 c'est idéal mais très long, 100 est un bon compromis
all_val_acc = []

for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(4, kernel_size=(5, 5), activation="softmax"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(8, kernel_size=(4, 4), activation="softmax"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )

    val_acc = history.history["val_acc"]
    # Padding : si early stopping coupe à 7 époques, on complète avec la dernière valeur
    val_acc += [val_acc[-1]] * (epochs - len(val_acc))
    all_val_acc.append(val_acc)

# --- Calcul moyenne + écart-type ---
all_val_acc = np.array(all_val_acc)       # shape (n_runs, epochs)
moyenne  = np.mean(all_val_acc, axis=0)
ecart_type = np.std(all_val_acc, axis=0)
x = np.arange(1, epochs + 1)

# --- Graphique ---
plt.plot(x, moyenne, label="Moyenne", color="blue")
plt.fill_between(
    x,
    moyenne - ecart_type,
    moyenne + ecart_type,
    alpha=0.3,
    color="blue",
    label="± écart-type"
)
plt.xlabel("Époque")
plt.ylabel("val_acc")
plt.title(f"Moyenne + écart-type sur {n_runs} runs")
plt.legend()
plt.savefig(OUTPUT_DIR / "test_stabilite_validation_accuracy.png")
plt.show()
