print("demarage des importation ...")
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print("os réuissit")
import numpy as np
print("numpy réuissit")
import matplotlib.pyplot as plt
print("matplotlib réuissit")
import seaborn as sns
print("seaborn réuissit")
from sklearn.metrics import confusion_matrix, classification_report
print("sklearn réuissit")
import keras
print("keras réuissit")

# ====== Charger les données de test ======
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

# ====== Charger le modèle ======
model = keras.models.load_model("final_model.keras")

# ====== Faire les prédictions ======
y_pred_proba = model.predict(x_test)          # Probabilités pour chaque classe
y_pred = np.argmax(y_pred_proba, axis=1)      # Classe prédite (0-9)

# ====== Créer la matrice de confusion ======
cm = confusion_matrix(y_test, y_pred)

# ====== Afficher en nombres ======
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,           # Afficher les nombres
    fmt="d",              # Format entier
    cmap="Blues",          # Couleurs bleues
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.title("Matrice de Confusion")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ====== Afficher en pourcentages ======
cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",            # 1 décimale
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.title("Matrice de Confusion (en %)")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.tight_layout()
plt.savefig("confusion_matrix_percent.png")
plt.show()

# ====== Rapport détaillé par classe ======
print("\n📊 Rapport de classification :\n")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
