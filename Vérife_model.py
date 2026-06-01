import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
import matplotlib.pyplot as plt


# Importation du dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Ajout du canal couleur
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
epochs = 20
for i in range(100):
    print(f"Run {i+1}")
    model = keras.Sequential(
        [   # permière couche, qui va prendnre en input un image de 28 x 28 x 1.
            keras.layers.Input(shape=(28,28,1)),
            # première chouche de covloution avec 16 filtre un taille de kernel de 5 x 5 et une fonciton d'activation en relu
            keras.layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
            # batchnomalisation va réduire les  trop grandes écarats entre les valeure de ma convolution. 
            keras.layers.BatchNormalization(),
            # max pooling va nous permettre de se concentré sur se qu'il compte
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # deuxième chouche de covloution avec 32 filtre un taille de kernel de 3 x 3 et une fonciton d'activation en relu
            keras.layers.Conv2D(32, kernel_size=(4, 4), activation="softmax"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # on déplie notre matrice sur une ligne 
            keras.layers.Flatten(),
            # on desactive une partie des neurones.
            keras.layers.Dropout(0.5),
            # et le neurones qui s'acitve le plus sea notre output. (plus le neurones activé à un valeur poroche de 1 plus il est sûr de luis)
            keras.layers.Dense(10, activation="softmax"),
        ])
        #calcule de la loss, la décente de gradiant(Adam) et de l'acc
    model.compile(
            loss = keras.losses.SparseCategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )
        #early stoping
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        ]

    
    history = model.fit(
        x_train,
        y_train,
        batch_size= 128,
        epochs = epochs,
        validation_split=0.15,
        callbacks= callbacks
        )

    #cration de graphique
    plt.plot(history.history["val_acc"], 
         label=f"Run {i+1}" if i % 10 == 0 else None)
   
plt.legend()
plt.xlabel("époque")
plt.ylabel("val_acc")
plt.show()
