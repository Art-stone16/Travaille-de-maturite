import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES ET DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32")  / 255
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test,  -1)

num_classes  = 10
input_shape  = (28, 28, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCTION DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64,  kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64,  kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
model.build(input_shape=(None, 28, 28, 1))  
# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATION 1 : STRUCTURE DU RÉSEAU
#    On dessine chaque couche comme une boîte avec ses paramètres.
# ─────────────────────────────────────────────────────────────────────────────

def visualiser_structure(model):
    """
    Dessine un schéma vertical du réseau couche par couche.
    Chaque boîte montre : le nom de la couche, son type, et sa forme de sortie.
    """
    layers = model.layers
    n = len(layers)

    # Couleurs par type de couche pour mieux les distinguer
    couleurs = {
        "Conv2D":               "#4A90D9",   # bleu
        "MaxPooling2D":         "#E67E22",   # orange
        "GlobalAveragePooling2D": "#27AE60", # vert
        "Dropout":              "#8E44AD",   # violet
        "Dense":                "#E74C3C",   # rouge
    }
    couleur_defaut = "#95A5A6"

    fig, ax = plt.subplots(figsize=(8, n * 1.1 + 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n + 0.5)
    ax.axis("off")
    ax.set_title("Structure du réseau de neurones", fontsize=15, fontweight="bold", pad=15)

    for i, layer in enumerate(layers):
        y = n - 1 - i   # on dessine de haut en bas

        # Couleur selon le type
        type_couche = layer.__class__.__name__
        couleur = couleurs.get(type_couche, couleur_defaut)

        # Rectangle représentant la couche
        rect = plt.Rectangle((1, y - 0.35), 8, 0.7,
                              linewidth=1.5, edgecolor="white",
                              facecolor=couleur, alpha=0.85, zorder=2)
        ax.add_patch(rect)

        # Forme de sortie (ex: (None, 26, 26, 64))
        forme = str(layer.output_shape) if hasattr(layer, "output_shape") else "?"

        # Nombre de paramètres
        params = layer.count_params()
        params_txt = f"{params:,} params" if params > 0 else "0 params"

        # Texte principal : nom + type
        ax.text(5, y + 0.05, f"{layer.name}  [{type_couche}]",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", zorder=3)

        # Texte secondaire : forme + params
        ax.text(5, y - 0.18, f"sortie : {forme}   •   {params_txt}",
                ha="center", va="center", fontsize=7.5,
                color="white", alpha=0.9, zorder=3)

        # Flèche entre couches
        if i < n - 1:
            ax.annotate("", xy=(5, y - 0.35), xytext=(5, y - 0.65),
                        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    plt.tight_layout()
    plt.savefig("structure_reseau.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅  Structure sauvegardée : structure_reseau.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT AVEC HISTORIQUE
#    On récupère les métriques à chaque époque pour les tracer ensuite.
# ─────────────────────────────────────────────────────────────────────────────

# Callback d'arrêt précoce (évite d'overfitter)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
]

print("\n🚀 Entraînement en cours...\n")
historique = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION 2 : COURBES D'ENTRAÎNEMENT
#    loss = erreur du modèle (plus c'est bas, mieux c'est)
#    accuracy = précision (plus c'est haut, mieux c'est)
# ─────────────────────────────────────────────────────────────────────────────

def visualiser_courbes(historique):
    """
    Trace 2 graphiques côte à côte :
      - Gauche : perte (loss) sur entraînement et validation
      - Droite : précision (accuracy) sur entraînement et validation
    La différence entre train et val indique si le modèle overfitte.
    """
    hist = historique.history
    epoques = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Courbes d'entraînement", fontsize=15, fontweight="bold")

    # --- Graphique LOSS ---
    ax1.plot(epoques, hist["loss"],     "o-", color="#4A90D9", label="Entraînement", linewidth=2)
    ax1.plot(epoques, hist["val_loss"], "s--", color="#E74C3C", label="Validation",   linewidth=2)
    ax1.set_title("Perte (Loss)\n↓ plus c'est bas, mieux c'est", fontsize=11)
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Graphique ACCURACY ---
    ax2.plot(epoques, hist["acc"],     "o-", color="#27AE60", label="Entraînement", linewidth=2)
    ax2.plot(epoques, hist["val_acc"], "s--", color="#E67E22", label="Validation",   linewidth=2)
    ax2.set_title("Précision (Accuracy)\n↑ plus c'est haut, mieux c'est", fontsize=11)
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("courbes_entrainement.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅  Courbes sauvegardées : courbes_entrainement.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION 3 : PRÉDICTIONS DU MODÈLE
#    On choisit 20 images de test au hasard et on regarde ce que le modèle
#    prédit. Vert = bonne prédiction, Rouge = erreur.
# ─────────────────────────────────────────────────────────────────────────────

def visualiser_predictions(model, x_test, y_test, n=20):
    """
    Affiche n images du jeu de test avec :
      - La vraie étiquette (label réel)
      - La prédiction du modèle
      - En VERT si correct, en ROUGE si erreur
    """
    # Choisir n images aléatoires
    indices = np.random.choice(len(x_test), n, replace=False)
    images  = x_test[indices]
    labels  = y_test[indices]

    # Obtenir les prédictions (vecteur de probabilités pour chaque chiffre 0-9)
    probabilites = model.predict(images, verbose=0)
    predictions  = np.argmax(probabilites, axis=1)  # chiffre le plus probable

    # Affichage en grille 4 x 5
    cols = 5
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 2.8))
    fig.suptitle("Prédictions du modèle sur des images test", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze()   # enlève la dimension couleur (28,28,1) → (28,28)

        correct = (predictions[i] == labels[i])
        couleur_bord = "#27AE60" if correct else "#E74C3C"
        symbole      = "✓" if correct else "✗"

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Réel : {labels[i]}   Prédit : {predictions[i]} {symbole}",
            fontsize=9,
            color=couleur_bord,
            fontweight="bold",
        )

        # Encadrer l'image en vert ou rouge
        for spine in ax.spines.values():
            spine.set_edgecolor(couleur_bord)
            spine.set_linewidth(3)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅  Prédictions sauvegardées : predictions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. LANCEMENT DE TOUTES LES VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n📊 Génération des visualisations...\n")

visualiser_structure(model)
visualiser_courbes(historique)
visualiser_predictions(model, x_test, y_test, n=20)

# Score final sur le jeu de test
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n🏆 Score final sur le jeu de test :")
print(f"   Loss     : {loss:.4f}")
print(f"   Accuracy : {acc*100:.2f}%")