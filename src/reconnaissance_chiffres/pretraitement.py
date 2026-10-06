"""Préparation des zones détectées pour le modèle et le contrôle qualité."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreparationChiffre:
    """Images intermediaires produites pour une entree du modele."""

    tenseur_modele: np.ndarray
    recadrage_original: np.ndarray
    image_28_niveaux_gris: np.ndarray
    matrice_binaire_0_1: np.ndarray


def preparer_chiffre_avec_details(image, rectangle, sigma_fond_local=35):
    """Prepare un chiffre et conserve les images utiles au controle qualite.

    ``image_28_niveaux_gris`` correspond exactement aux valeurs envoyees au
    modele, avant leur normalisation entre 0 et 1. La matrice binaire est une
    vue de controle supplementaire et ne remplace donc pas l'entree du modele.
    """
    x, y, largeur, hauteur = rectangle

    if largeur <= 0 or hauteur <= 0:
        raise ValueError(f"Rectangle vide ou invalide: {rectangle}")

    chiffre = image[y : y + hauteur, x : x + largeur]

    if chiffre.size == 0:
        raise ValueError(f"Rectangle en dehors de l'image: {rectangle}")

    gris = cv2.cvtColor(chiffre, cv2.COLOR_BGR2GRAY)

    fond_local = cv2.GaussianBlur(
        gris,
        (0, 0),
        sigmaX=sigma_fond_local,
        sigmaY=sigma_fond_local,
    )
    traits_sombres = cv2.subtract(fond_local, gris)
    contraste = cv2.normalize(traits_sombres, None, 0, 255, cv2.NORM_MINMAX)
    _, chiffre_binaire = cv2.threshold(
        contraste,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

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

    chiffre_28 = cv2.resize(carre, (28, 28), interpolation=cv2.INTER_AREA)
    matrice_binaire = (chiffre_28 >= 128).astype(np.uint8)
    tenseur_modele = chiffre_28.astype("float32") / 255
    tenseur_modele = np.expand_dims(tenseur_modele, axis=(0, -1))

    return PreparationChiffre(
        tenseur_modele=tenseur_modele,
        recadrage_original=chiffre.copy(),
        image_28_niveaux_gris=chiffre_28,
        matrice_binaire_0_1=matrice_binaire,
    )


def preparer_chiffre_pour_modele(image, rectangle, sigma_fond_local=35):
    """Transforme une zone detectee en tenseur 1 x 28 x 28 x 1.

    Cette interface historique est conservee pour les autres scripts. Pour un
    controle visuel complet, utiliser :func:`preparer_chiffre_avec_details`.
    """
    preparation = preparer_chiffre_avec_details(
        image,
        rectangle,
        sigma_fond_local=sigma_fond_local,
    )

    return preparation.tenseur_modele
