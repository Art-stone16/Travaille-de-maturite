import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# 1. Chargement des données MNIST (Téléchargement auto si nécessaire)
print("Chargement des données...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2. Prétraitement des données
# On normalise les pixels (valeurs entre 0 et 1 au lieu de 0 à 255)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# On convertit les étiquettes (ex: le chiffre '5' devient [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 3. Création de l'architecture du Réseau de Neurones (CNN)
model = models.Sequential()

# Couches de convolution (pour extraire les caractéristiques visuelles)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Couches de classification (le "cerveau" qui décide)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # 10 neurones pour les 10 chiffres (0-9)

# 4. Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Entraînement
print("Début de l'entraînement...")
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# 6. Sauvegarde du modèle
model.save('mon_modele_chiffres.h5')
print("Modèle sauvegardé sous 'mon_modele_chiffres.h5' !")