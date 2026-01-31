import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Chemin vers le dossier contenant les dossiers '0', '1', ..., '9'
data_dir = '/Users/arthurperret/Documents/MNIST Dataset/trainingSet/trainingSet'

# Initialiser les listes pour les images et les labels
images = []
labels = []

# Parcourir chaque dossier (0 à 9)
for digit in range(10):
    digit_dir = os.path.join(data_dir, str(digit))
    for img_file in os.listdir(digit_dir):
        if img_file.endswith('.jpg'):
            # Charger l'image en niveaux de gris et redimensionner à 28x28
            img_path = os.path.join(digit_dir, img_file)
            img = tf.keras.preprocessing.image.load_img(
                img_path, color_mode='grayscale', target_size=(28, 28)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(digit)

# Convertir les listes en tableaux NumPy
train_images = np.array(images, dtype='float32') / 255.0
train_labels = np.array(labels, dtype='int32')

# Ajouter une dimension pour le canal (gris)
train_images = train_images.reshape((-1, 28, 28, 1))

# Mélanger les données
indices = np.arange(len(train_images))
np.random.shuffle(indices)
train_images = train_images[indices]
train_labels = train_labels[indices]

# Définir le modèle (identique au précédent)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_images, train_labels, epochs=5)

# Note : Pour les données de test, utilise un chemin similaire vers 'testSet'
