import env_config
import cv2
import keras
import numpy as np


# Image a analyser.
IMAGE_PATH = env_config.PROJECT_ROOT / "donnees" / "image_reelle_Grand.jpg"

# Nom du modele utilise. Il sera aussi ecrit sur l'image finale.
MODEL_NAME = "Best_COLOR_MAP"

# Modele qui reconnait les chiffres. Il attend une image en 28 x 28 x 1,
# comme les images du dataset MNIST utilisees pendant l'entrainement.
MODEL_PATH = (
    env_config.PROJECT_ROOT
    / "modeles"
    / "modeles_valides"
    / MODEL_NAME
    / "best_model.keras"
)

# Image finale avec les rectangles rouges autour des chiffres detectes.
OUTPUT_DIR = env_config.PROJECT_ROOT / "sorties" / "tests_condition_reelle"

# Mettre cette constante a True si tu veux afficher les images dans des fenetres
# OpenCV en plus de sauvegarder le resultat.
AFFICHER_IMAGES = False

# Parametres de detection. Ils sont regroupes ici pour pouvoir les ajuster
# facilement si tu prends une nouvelle photo avec un cadrage ou un eclairage
# different.
SIGMA_FOND_LOCAL = 35
TAILLE_FOND_MORPHOLOGIQUE = 51
TAILLE_REGROUPEMENT = 15
MARGE_RECTANGLE = 20
EPAISSEUR_RECTANGLE = 6
LIMITE_BASSE_IMAGE = 0.85
SAUVEGARDER_DIAGNOSTICS = True


def charger_image(image_path):
    """Charge l'image et signale clairement l'erreur si le fichier manque."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Image introuvable: {image_path}")

    return image


def charger_modele(model_path):
    """Charge le modele Keras qui va reconnaitre les chiffres."""
    if not model_path.exists():
        raise ValueError(f"Modele introuvable: {model_path}")

    return keras.models.load_model(model_path)


def creer_chemin_sortie(output_dir):
    """Cree un nouveau chemin test_1.jpg, test_2.jpg, etc. sans ecraser."""
    numero_test = 1

    while True:
        output_path = output_dir / f"test_{numero_test}.jpg"

        if not output_path.exists():
            return output_path

        numero_test += 1


def formater_confiance(confiance):
    """Affiche trois decimales sans suggerer une certitude absolue."""
    if confiance >= 0.999995:
        return ">99.999%"

    return f"{confiance:.3%}"


def detecter_chiffres(image):
    """Detecte les zones qui ressemblent a des chiffres manuscrits."""
    hauteur_image, largeur_image = image.shape[:2]
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Une fermeture morphologique remplit les traits sombres et estime ainsi
    # la couleur locale du papier, meme lorsqu'il est gris ou mal eclaire.
    noyau_fond = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (TAILLE_FOND_MORPHOLOGIQUE, TAILLE_FOND_MORPHOLOGIQUE),
    )
    fond_local = cv2.morphologyEx(
        gris,
        cv2.MORPH_CLOSE,
        noyau_fond,
    )
    traits_sombres = cv2.subtract(fond_local, gris)

    # Otsu choisit automatiquement le seuil qui separe le bruit clair des
    # traits fonces. Il est plus robuste qu'une constante pour une autre photo.
    _, masque_brut = cv2.threshold(
        traits_sombres,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Cette dilatation sert seulement a reunir les morceaux d'un meme chiffre.
    # Le masque d'Otsu est deja propre: un nettoyage supplementaire retirerait
    # certaines parties horizontales utiles des chiffres.
    noyau_regroupement = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (TAILLE_REGROUPEMENT, TAILLE_REGROUPEMENT),
    )
    masque_regroupe = cv2.dilate(masque_brut, noyau_regroupement)

    if SAUVEGARDER_DIAGNOSTICS:
        cv2.imwrite(str(OUTPUT_DIR / "diagnostic_1_traits_sombres.jpg"), traits_sombres)
        cv2.imwrite(str(OUTPUT_DIR / "diagnostic_2_masque_brut.jpg"), masque_brut)

    contours, _ = cv2.findContours(
        masque_regroupe,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rectangles = []

    for contour in contours:
        x, y, largeur, hauteur = cv2.boundingRect(contour)
        aire = largeur * hauteur
        rapport_largeur_hauteur = largeur / hauteur

        # Les limites sont assez larges pour accepter les 1 tres fins et les 9
        # plus hauts, tout en rejetant les points et les longues lignes.
        est_taille_chiffre = (
            45 <= largeur <= 320
            and 100 <= hauteur <= 600
            and 4_500 <= aire <= 180_000
        )
        est_forme_chiffre = 0.10 <= rapport_largeur_hauteur <= 1.30
        est_dans_zone_utile = (
            100 <= y
            and y + hauteur <= hauteur_image * LIMITE_BASSE_IMAGE
            and x + largeur <= largeur_image
        )

        if est_taille_chiffre and est_forme_chiffre and est_dans_zone_utile:
            rectangles.append((x, y, largeur, hauteur))

    # Tri de haut en bas puis de gauche a droite pour obtenir un ordre stable.
    return sorted(rectangles, key=lambda rectangle: (rectangle[1], rectangle[0]))


def preparer_chiffre_pour_modele(image, rectangle):
    """Transforme un chiffre detecte en image 28 x 28 x 1 pour le modele."""
    x, y, largeur, hauteur = rectangle

    # On coupe seulement la partie de l'image qui contient le chiffre detecte.
    chiffre = image[y : y + hauteur, x : x + largeur]
    gris = cv2.cvtColor(chiffre, cv2.COLOR_BGR2GRAY)

    # Meme idee que pour la detection: on fait ressortir les traits sombres du
    # chiffre par rapport au fond de papier local.
    fond_local = cv2.GaussianBlur(
        gris,
        (0, 0),
        sigmaX=SIGMA_FOND_LOCAL,
        sigmaY=SIGMA_FOND_LOCAL,
    )
    traits_sombres = cv2.subtract(fond_local, gris)
    contraste = cv2.normalize(traits_sombres, None, 0, 255, cv2.NORM_MINMAX)

    # Le modele MNIST a appris avec des chiffres clairs sur fond noir. Ce seuil
    # donne donc une petite image binaire: fond noir, chiffre blanc.
    _, chiffre_binaire = cv2.threshold(
        contraste,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # On recentre le chiffre dans un carre avant de le redimensionner. Cela
    # evite de deformer un chiffre haut et fin comme le 1 ou le 9.
    contours, _ = cv2.findContours(
        chiffre_binaire,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x_chiffre, y_chiffre, largeur_chiffre, hauteur_chiffre = cv2.boundingRect(
            contour
        )
        chiffre_binaire = chiffre_binaire[
            y_chiffre : y_chiffre + hauteur_chiffre,
            x_chiffre : x_chiffre + largeur_chiffre,
        ]

    hauteur_chiffre, largeur_chiffre = chiffre_binaire.shape[:2]
    cote = max(hauteur_chiffre, largeur_chiffre)
    marge = max(4, int(cote * 0.20))
    carre = np.zeros((cote + 2 * marge, cote + 2 * marge), dtype=np.uint8)

    y_depart = marge + (cote - hauteur_chiffre) // 2
    x_depart = marge + (cote - largeur_chiffre) // 2
    carre[
        y_depart : y_depart + hauteur_chiffre,
        x_depart : x_depart + largeur_chiffre,
    ] = chiffre_binaire

    # Passage au format exact attendu par le modele: 28 x 28 pixels, valeurs
    # entre 0 et 1, puis ajout des dimensions batch et canal: 1 x 28 x 28 x 1.
    chiffre_28 = cv2.resize(carre, (28, 28), interpolation=cv2.INTER_AREA)
    chiffre_28 = chiffre_28.astype("float32") / 255
    chiffre_28 = np.expand_dims(chiffre_28, axis=(0, -1))

    return chiffre_28


def reconnaitre_chiffres(image, rectangles, modele):
    """Utilise le modele pour predire le chiffre dans chaque rectangle."""
    predictions = []

    for rectangle in rectangles:
        chiffre_28 = preparer_chiffre_pour_modele(image, rectangle)
        probabilites = modele.predict(chiffre_28, verbose=0)[0]
        chiffre_predit = int(np.argmax(probabilites))
        confiance = float(probabilites[chiffre_predit])
        print(confiance)
        predictions.append((rectangle, chiffre_predit, confiance))

    return predictions


def dessiner_rectangles(image, predictions):
    """Dessine un rectangle rouge et le chiffre predit par le modele."""
    image_encadree = image.copy()
    hauteur_image, largeur_image = image_encadree.shape[:2]

    # Ecriture du nom du modele utilise sur l'image finale.
    cv2.putText(
        image_encadree,
        f"Modele utilise: {MODEL_NAME}",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )

    for rectangle, chiffre_predit, confiance in predictions:
        x, y, largeur, hauteur = rectangle

        # La marge evite que le cadre touche directement le trait du chiffre.
        x1 = max(x - MARGE_RECTANGLE, 0)
        y1 = max(y - MARGE_RECTANGLE, 0)
        x2 = min(x + largeur + MARGE_RECTANGLE, largeur_image - 1)
        y2 = min(y + hauteur + MARGE_RECTANGLE, hauteur_image - 1)

        cv2.rectangle(
            image_encadree,
            (x1, y1),
            (x2, y2),
            color=(0, 0, 255),
            thickness=EPAISSEUR_RECTANGLE,
        )

        # Texte au-dessus du rectangle: chiffre predit + confiance du modele.
        cv2.putText(
            image_encadree,
            f"{chiffre_predit} ({formater_confiance(confiance)})",
            (x1, max(y1 - 15, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )

    return image_encadree


def main():
    """Lance toute la detection et sauvegarde l'image finale."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = creer_chemin_sortie(OUTPUT_DIR)

    image = charger_image(IMAGE_PATH)
    modele = charger_modele(MODEL_PATH)
    rectangles = detecter_chiffres(image)
    predictions = reconnaitre_chiffres(image, rectangles, modele)
    image_encadree = dessiner_rectangles(image, predictions)

    cv2.imwrite(str(output_path), image_encadree)

    print(f"{len(rectangles)} chiffre(s) encadre(s).")
    for _, chiffre_predit, confiance in predictions:
        confiance_affichee = formater_confiance(confiance)
        print(
            f"Prediction: {chiffre_predit} avec "
            f"{confiance_affichee} de confiance"
        )

    print(f"Image sauvegardee ici: {output_path}")

    if AFFICHER_IMAGES:
        cv2.imshow("Image originale", image)
        cv2.imshow("Chiffres encadres", image_encadree)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
