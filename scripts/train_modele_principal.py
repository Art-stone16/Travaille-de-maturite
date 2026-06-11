#Importation d'OS pour définir quel moteur Keras utilise. 
import os
import env_config
# Définition du moteur pour Keras, important de faire avant l'importation de Keras. 
os.environ["KERAS_BACKEND"] = "tensorflow"
# Importation de numpy pour manipuler mes données. 
import numpy as np
import keras
# Importation de matplotlib pour la création de graphiques. 
import matplotlib.pyplot as plt

# Importation du dataset. 
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
# Fait en sorte que les pixels des images soient entre 0 et 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Fait en sorte que les images soient en 28 x 28 x 1
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# Visualisation des données importées
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# num_classes c'est le nombre de possibilités en output. ( je l'utilise dans ma dernière couche de mon modèle)
num_classes = 10
# C'est la "bouchée" que va regarder mon réseau de neurones. (c'est une variable qui influence l'entraînement.)
input_shape = (28, 28, 1)
# Création de l'architecture
model = keras.Sequential(
    [   # Première couche, qui va prendre en input une image de 28 x 28 x 1.
        keras.layers.Input(shape=input_shape),
        # Première couche de convolution avec 16 filtres, une taille de kernel de 5 x 5 et une fonction d'activation en relu
        keras.layers.Conv2D(4, kernel_size=(5, 5), activation="softmax"),
        # BatchNormalization va réduire les trop grands écarts entre les valeurs de ma convolution. 
        keras.layers.BatchNormalization(),
        # MaxPooling va nous permettre de se concentrer sur ce qui compte
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Deuxième couche de convolution avec 32 filtres, une taille de kernel de 3 x 3 et une fonction d'activation en relu
        keras.layers.Conv2D(8, kernel_size=(4, 4), activation="softmax"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # On déplie notre matrice sur une ligne 
        keras.layers.Flatten(),
        # On désactive une partie des neurones.
        keras.layers.Dropout(0.2),
        # Le neurone qui s'active le plus sera notre output. (plus le neurone activé a une valeur proche de 1, plus il est sûr de lui)
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
# Nom du modèle
MODEL_NAME = "A_SUprimer"
MODEL_DIR = env_config.PROJECT_ROOT / "modeles" / "modeles_valides"
GRAPH_DIR = env_config.PROJECT_ROOT / "sorties" / "graphiques"
os.makedirs(MODEL_DIR / MODEL_NAME, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)



callbacks = [   
    keras.callbacks.ModelCheckpoint(filepath=str(MODEL_DIR / MODEL_NAME / "best_model.keras"), save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]
# Application de toutes les règles prédéfinies précédemment
history = model.fit(
    # Images
    x_train,
    # Étiquettes
    y_train,
    # Bouchée
    batch_size=128,
    # Nombre de visualisations
    epochs=30,
    # Met de côté 15% des images pour tester le modèle avec des images qu'il ne connaît pas.
    validation_split=0.15,
    # Exécute des actions après chaque epoch
    callbacks=callbacks,
)

score = model.evaluate(x_test, y_test, verbose=0)
model.save(MODEL_DIR / MODEL_NAME / "final_model.keras")


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
plt.show()

# Visualisation de son score final avec les données d'entraînement.
print(score)
print(history.history)
