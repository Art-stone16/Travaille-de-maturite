"""Detection generale de chiffres manuscrits dans une image."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    """Parametres independants de la resolution de l'image."""

    proportion_fond: float = 0.04
    taille_fond_minimum: int = 31
    taille_fond_maximum: int = 151
    proportion_regroupement: float = 0.0075
    taille_regroupement_minimum: int = 3

    proportion_largeur_minimum: float = 0.01
    proportion_largeur_maximum: float = 0.16
    proportion_hauteur_minimum: float = 0.014
    proportion_hauteur_maximum: float = 0.20
    proportion_aire_minimum: float = 0.0002
    proportion_aire_maximum: float = 0.04
    limite_basse_image: float = 0.995

    seuil_otsu_faible: float = 30
    percentile_contraste_faible: float = 99.5
    proportion_aire_minimum_faible: float = 0.001

    facteur_seuil_fort: float = 1.35
    percentile_seuil_fort: float = 98
    support_fort_minimum: float = 0.01
    pixels_forts_minimum: int = 2


@dataclass(frozen=True)
class Candidat:
    """Zone examinee par le detecteur."""

    rectangle: tuple[int, int, int, int]
    accepte: bool
    raisons_rejet: tuple[str, ...]
    densite_encre: float
    support_fort: float


@dataclass
class ResultatDetection:
    """Resultat complet, y compris les images utiles au diagnostic."""

    rectangles: list[tuple[int, int, int, int]]
    candidats: list[Candidat]
    traits_sombres: np.ndarray
    masque_faible: np.ndarray
    masque_fort: np.ndarray
    masque_filtre: np.ndarray
    masque_regroupe: np.ndarray
    seuil_otsu: float
    seuil_faible: float
    seuil_fort: float
    taille_fond: int
    taille_regroupement: int


@dataclass
class PreparationChiffre:
    """Images intermediaires produites pour une entree du modele."""

    tenseur_modele: np.ndarray
    recadrage_original: np.ndarray
    image_28_niveaux_gris: np.ndarray
    matrice_binaire_0_1: np.ndarray


CONFIG_PAR_DEFAUT = DetectionConfig()


def charger_image(image_path):
    """Charge une image et produit une erreur claire si elle est absente."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Image introuvable: {image_path}")

    return image


def rendre_impair(valeur, minimum, maximum=None):
    """Retourne une taille impaire utilisable comme noyau OpenCV."""
    valeur = max(minimum, round(valeur))

    if maximum is not None:
        valeur = min(maximum, valeur)

    if valeur % 2 == 0:
        valeur += 1

    return valeur


def _filtrer_par_hysteresis(
    masque_faible,
    masque_fort,
    config,
):
    """Garde les traits faibles uniquement s'ils possedent un noyau fort."""
    nombre, etiquettes, statistiques, _ = cv2.connectedComponentsWithStats(
        masque_faible,
        connectivity=8,
    )

    if nombre <= 1:
        return np.zeros_like(masque_faible)

    aires = statistiques[:, cv2.CC_STAT_AREA].astype(np.float64)
    etiquettes_fortes = etiquettes[masque_fort > 0]
    comptes_forts = np.bincount(etiquettes_fortes, minlength=nombre)
    supports = np.divide(
        comptes_forts,
        aires,
        out=np.zeros_like(aires),
        where=aires > 0,
    )

    etiquettes_gardees = (
        (comptes_forts >= config.pixels_forts_minimum)
        & (supports >= config.support_fort_minimum)
    )
    etiquettes_gardees[0] = False

    return np.where(etiquettes_gardees[etiquettes], 255, 0).astype(np.uint8)


def detecter_chiffres(image, config=CONFIG_PAR_DEFAUT):
    """Detecte des chiffres sans utiliser leur classe ni leur disposition."""
    hauteur_image, largeur_image = image.shape[:2]
    aire_image = hauteur_image * largeur_image
    petit_cote = min(hauteur_image, largeur_image)
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    taille_fond = rendre_impair(
        petit_cote * config.proportion_fond,
        config.taille_fond_minimum,
        config.taille_fond_maximum,
    )
    noyau_fond = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (taille_fond, taille_fond),
    )
    fond_local = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, noyau_fond)
    traits_sombres = cv2.subtract(fond_local, gris)

    seuil_otsu, _ = cv2.threshold(
        traits_sombres,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    seuil_faible = float(seuil_otsu)
    proportion_aire_minimum = config.proportion_aire_minimum

    if seuil_otsu < config.seuil_otsu_faible:
        seuil_faible = max(
            seuil_faible,
            float(
                np.percentile(
                    traits_sombres,
                    config.percentile_contraste_faible,
                )
            ),
        )
        proportion_aire_minimum = config.proportion_aire_minimum_faible

    seuil_fort = min(
        254.0,
        max(
            seuil_faible * config.facteur_seuil_fort,
            float(
                np.percentile(
                    traits_sombres,
                    config.percentile_seuil_fort,
                )
            ),
        ),
    )

    _, masque_faible = cv2.threshold(
        traits_sombres,
        seuil_faible,
        255,
        cv2.THRESH_BINARY,
    )
    _, masque_fort = cv2.threshold(
        traits_sombres,
        seuil_fort,
        255,
        cv2.THRESH_BINARY,
    )
    masque_filtre = _filtrer_par_hysteresis(
        masque_faible,
        masque_fort,
        config,
    )

    taille_regroupement = rendre_impair(
        petit_cote * config.proportion_regroupement,
        config.taille_regroupement_minimum,
    )
    noyau_regroupement = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (taille_regroupement, taille_regroupement),
    )
    masque_regroupe = cv2.dilate(masque_filtre, noyau_regroupement)

    contours, _ = cv2.findContours(
        masque_regroupe,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rectangles = []
    candidats = []

    for contour in contours:
        x, y, largeur, hauteur = cv2.boundingRect(contour)
        aire = largeur * hauteur
        rapport = largeur / hauteur
        raisons_rejet = []

        if not (
            largeur_image * config.proportion_largeur_minimum
            <= largeur
            <= largeur_image * config.proportion_largeur_maximum
        ):
            raisons_rejet.append("largeur")

        if not (
            hauteur_image * config.proportion_hauteur_minimum
            <= hauteur
            <= hauteur_image * config.proportion_hauteur_maximum
        ):
            raisons_rejet.append("hauteur")

        if not (
            aire_image * proportion_aire_minimum
            <= aire
            <= aire_image * config.proportion_aire_maximum
        ):
            raisons_rejet.append("aire")

        if not 0.08 <= rapport <= 2.0:
            raisons_rejet.append("forme")

        if y + hauteur > hauteur_image * config.limite_basse_image:
            raisons_rejet.append("position")

        zone_faible = masque_faible[y : y + hauteur, x : x + largeur]
        zone_forte = masque_fort[y : y + hauteur, x : x + largeur]
        pixels_encre = cv2.countNonZero(zone_faible)
        pixels_forts = cv2.countNonZero(zone_forte)
        densite_encre = pixels_encre / aire if aire else 0.0
        support_fort = pixels_forts / pixels_encre if pixels_encre else 0.0
        accepte = not raisons_rejet
        rectangle = (x, y, largeur, hauteur)

        candidats.append(
            Candidat(
                rectangle=rectangle,
                accepte=accepte,
                raisons_rejet=tuple(raisons_rejet),
                densite_encre=densite_encre,
                support_fort=support_fort,
            )
        )

        if accepte:
            rectangles.append(rectangle)

    rectangles.sort(key=lambda rectangle: (rectangle[1], rectangle[0]))

    return ResultatDetection(
        rectangles=rectangles,
        candidats=candidats,
        traits_sombres=traits_sombres,
        masque_faible=masque_faible,
        masque_fort=masque_fort,
        masque_filtre=masque_filtre,
        masque_regroupe=masque_regroupe,
        seuil_otsu=float(seuil_otsu),
        seuil_faible=seuil_faible,
        seuil_fort=seuil_fort,
        taille_fond=taille_fond,
        taille_regroupement=taille_regroupement,
    )


def sauvegarder_diagnostics(resultat, image, output_dir):
    """Sauvegarde les etapes de detection et les composantes examinees."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_dir / "01_traits_sombres.jpg"),
        resultat.traits_sombres,
    )
    cv2.imwrite(
        str(output_dir / "02_masque_faible.png"),
        resultat.masque_faible,
    )
    cv2.imwrite(
        str(output_dir / "03_masque_fort.png"),
        resultat.masque_fort,
    )
    cv2.imwrite(
        str(output_dir / "04_masque_filtre.png"),
        resultat.masque_filtre,
    )

    diagnostic = image.copy()
    hauteur_image, largeur_image = image.shape[:2]
    aire_image = hauteur_image * largeur_image
    petit_cote = min(hauteur_image, largeur_image)
    echelle = max(0.35, min(0.8, petit_cote / 2000))

    for candidat in resultat.candidats:
        x, y, largeur, hauteur = candidat.rectangle
        couleur = (0, 180, 0) if candidat.accepte else (0, 0, 255)
        cv2.rectangle(
            diagnostic,
            (x, y),
            (x + largeur, y + hauteur),
            couleur,
            2,
        )

        if (
            not candidat.accepte
            and largeur * hauteur >= aire_image * 0.0001
        ):
            cv2.putText(
                diagnostic,
                ",".join(candidat.raisons_rejet),
                (x, max(y - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                echelle,
                couleur,
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(
        str(output_dir / "05_composantes.jpg"),
        diagnostic,
    )


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
