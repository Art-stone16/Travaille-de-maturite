#Importation d'OS pour définir quelle moteur Keras utilise. 
import os
# définition du moteur pour Keras, Important de faire avant l'importation de Keras. 
os.environ["KERAS_BACKEND"] = "tensorflow"
#importation de numpy pour manipulé mes donnés. 
import numpy as np
import keras
#importation de matpotlib pour la créationd de graphique. 
import matplotlib.pyplot as plt

#Importation du data set. 
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
# Fait en sorte que les pixel des images soie entre 0 et 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Fait en sorte que les images soient en 28 x 28 x 1
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# visualisation des donné importé
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# num casse c'est le nombre de possiblité en output. ( je l'uitlise dans ma dernièire couche de mon modèle)
num_classes = 20
# c'est l'as "bouché" que vas regardé mon réseaux de neurones. (est un varaible qui influance l'entrainement.)
input_shape = (28, 28, 1)
# création de l'architecture
model = keras.Sequential(
    [   # permière couche, qui va prendnre en input un image de 28 x 28 x 1.
        keras.layers.Input(shape=input_shape),
        # première chouche de covloution avec 16 filtre un taille de kernel de 5 x 5 et une fonciton d'activation en relu
        keras.layers.Conv2D(32, kernel_size=(5, 5), activation="softmax"),
        # batchnomalisation va réduire les  trop grandes écarats entre les valeure de ma convolution. 
        keras.layers.BatchNormalization(),
        # max pooling va nous permettre de se concentré sur se qu'il compte
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # deuxième chouche de covloution avec 32 filtre un taille de kernel de 3 x 3 et une fonciton d'activation en relu
        keras.layers.Conv2D(64, kernel_size=(4, 4), activation="softmax"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # on déplie notre matrice sur une ligne 
        keras.layers.Flatten(),
        # on desactive une partie des neurones.
        keras.layers.Dropout(0.5),
        # et le neurones qui s'acitve le plus sea notre output. (plus le neurones activé à un valeur poroche de 1 plus il est sûr de luis)
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

#visualisation de notre modèle 
model.summary()

# création de l'apprentisage
model.compile(
    #calcule la fonction loss
    loss=keras.losses.SparseCategoricalCrossentropy(),
    #optimise les parmètre
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    #permet de voir la precission. 
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
# nom du modèle
MODEL_NAME = "Best_TEST"
os.makedirs(f"Models/{MODEL_NAME}", exist_ok=True)
os.makedirs("Graphe", exist_ok=True)



callbacks = [   
    keras.callbacks.ModelCheckpoint(filepath=f"Models/{MODEL_NAME}/best_model.keras", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]
# application de toutes les règles prédéfinis précédemment
history = model.fit(
    #Image
    x_train,
    #Ethiquette
    y_train,
    #bouché
    batch_size=128,
    # nombre de visualisation
    epochs=20,
    # met de côté 15% des image pour tester le modèle avec des image qu'il ne connait pas.
    validation_split=0.15,
    # exéctue des action après chaque epoch
    callbacks=callbacks,
)

score = model.evaluate(x_test, y_test, verbose=0)
model.save(f"Models/{MODEL_NAME}/final_model.keras")


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
plt.savefig(f"Graphe/{MODEL_NAME}_training_curves.png")
plt.show()

# visualisation de son score finale avec donné d'entrainement.
print(score)
print(history.history)