import tensorflow as tf
from tensorflow.keras import layers, models

# 1️⃣ Charger le dataset (ex: MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2️⃣ Normalisation des pixels (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3️⃣ Ajouter une dimension "canal" (28,28) → (28,28,1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 4️⃣ Définition du modèle
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 chiffres (0–9)
])

# 5️⃣ Compilation du modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6️⃣ Entraînement
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 7️⃣ Évaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Précision sur les données de test :", test_accuracy)
