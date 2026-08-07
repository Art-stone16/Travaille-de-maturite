"""Mesure la cascade Top-N sur une feuille contenant un chiffre repete."""

import csv
import re
from datetime import datetime

import env_config
import cv2
import keras
import numpy as np

import detection_chiffres as detection


# Reglages du test -----------------------------------------------------------

IMAGE_PATH = env_config.PROJECT_ROOT / "donnees" / "CTN_9.jpg"
CHIFFRE_ATTENDU = 9
NOMBRE_ATTENDU = 100

MODEL_NAME = "Best_COLOR_MAP"
MODEL_PATH = (
    env_config.PROJECT_ROOT
    / "modeles"
    / "modeles_valides"
    / MODEL_NAME
    / "best_model.keras"
)
OUTPUT_DIR = env_config.PROJECT_ROOT / "sorties" / "cascade_top_n"
TOP_N_ORANGE = 9


def verifier_configuration():
    """Verifie les reglages avant de charger le modele."""
    if not 0 <= CHIFFRE_ATTENDU <= 9:
        raise ValueError("CHIFFRE_ATTENDU doit etre compris entre 0 et 9.")

    if NOMBRE_ATTENDU <= 0:
        raise ValueError("NOMBRE_ATTENDU doit etre strictement positif.")

    if not IMAGE_PATH.exists():
        raise ValueError(f"Photo introuvable: {IMAGE_PATH}")

    if not MODEL_PATH.exists():
        raise ValueError(f"Modele introuvable: {MODEL_PATH}")


def nettoyer_nom_dossier(nom):
    """Transforme un nom d'image en nom de dossier simple et stable."""
    nom_nettoye = re.sub(r"[^A-Za-z0-9._-]+", "_", nom).strip("._")
    return nom_nettoye or "image"


def creer_dossier_execution(date_test):
    """Cree un dossier unique organise par image et par execution."""
    dossier_image = OUTPUT_DIR / nettoyer_nom_dossier(IMAGE_PATH.stem)
    dossier_image.mkdir(parents=True, exist_ok=True)
    base = date_test.strftime("%Y-%m-%d_%H-%M-%S")
    dossier_execution = dossier_image / base
    suffixe = 2

    while dossier_execution.exists():
        dossier_execution = dossier_image / f"{base}_{suffixe:02d}"
        suffixe += 1

    dossier_execution.mkdir()
    return dossier_execution


def calculer_predictions(image, rectangles, modele):
    """Calcule en une fois les probabilites des dix classes."""
    if not rectangles:
        return np.empty((0, 10), dtype=np.float32)

    chiffres_prepares = [
        detection.preparer_chiffre_pour_modele(image, rectangle)
        for rectangle in rectangles
    ]
    lot = np.concatenate(chiffres_prepares, axis=0)
    probabilites = modele.predict(lot, verbose=0)

    if probabilites.ndim != 2 or probabilites.shape[1] != 10:
        raise ValueError(
            "Le modele doit produire dix probabilites. "
            f"Forme recue: {probabilites.shape}"
        )

    return probabilites


def analyser_predictions(rectangles, probabilites):
    """Construit le classement et le rang attendu pour chaque zone."""
    resultats = []

    for numero, (rectangle, probabilites_chiffre) in enumerate(
        zip(rectangles, probabilites),
        start=1,
    ):
        classement = np.argsort(probabilites_chiffre)[::-1]
        rang_attendu = int(np.where(classement == CHIFFRE_ATTENDU)[0][0]) + 1
        prediction = int(classement[0])

        resultats.append(
            {
                "numero": numero,
                "rectangle": rectangle,
                "prediction": prediction,
                "confiance_prediction": float(probabilites_chiffre[prediction]),
                "rang_attendu": rang_attendu,
                "confiance_attendue": float(
                    probabilites_chiffre[CHIFFRE_ATTENDU]
                ),
                "classement": [int(chiffre) for chiffre in classement],
                "probabilites": [
                    float(probabilites_chiffre[chiffre])
                    for chiffre in classement
                ],
            }
        )

    return resultats


def calculer_classement_global(resultats):
    """Classe les chiffres selon leur probabilite moyenne sur la feuille."""
    if not resultats:
        return []

    sommes = np.zeros(10, dtype=np.float64)

    for resultat in resultats:
        for chiffre, probabilite in zip(
            resultat["classement"],
            resultat["probabilites"],
        ):
            sommes[chiffre] += probabilite

    probabilites_moyennes = sommes / len(resultats)
    classement = np.argsort(probabilites_moyennes)[::-1]

    return [
        (int(chiffre), float(probabilites_moyennes[chiffre]))
        for chiffre in classement
    ]


def dessiner_resultats(image, resultats):
    """Encadre chaque chiffre selon le rang de la bonne reponse."""
    image_annotee = image.copy()
    _, largeur_image = image_annotee.shape[:2]
    echelle = max(0.45, min(1.0, largeur_image / 3000))
    epaisseur = max(2, round(echelle * 3))
    titre = (
        f"Image: {IMAGE_PATH.name} | Modele: {MODEL_NAME} | "
        f"Attendu: {CHIFFRE_ATTENDU}"
    )

    cv2.putText(
        image_annotee,
        titre,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        echelle,
        (0, 0, 255),
        epaisseur,
        cv2.LINE_AA,
    )

    for resultat in resultats:
        x, y, largeur, hauteur = resultat["rectangle"]
        rang = resultat["rang_attendu"]

        if rang == 1:
            couleur = (0, 180, 0)
        elif rang <= TOP_N_ORANGE:
            couleur = (0, 165, 255)
        else:
            couleur = (0, 0, 255)

        cv2.rectangle(
            image_annotee,
            (x, y),
            (x + largeur, y + hauteur),
            couleur,
            epaisseur,
        )
        texte = (
            f"#{resultat['numero']} "
            f"p={resultat['prediction']} rang={rang}"
        )
        cv2.putText(
            image_annotee,
            texte,
            (x, max(y - 8, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            echelle * 0.55,
            couleur,
            max(1, epaisseur - 1),
            cv2.LINE_AA,
        )

    return image_annotee


def sauvegarder_csv(csv_path, resultats):
    """Sauvegarde le classement detaille de chaque zone detectee."""
    entetes = [
        "image_source",
        "modele",
        "numero",
        "x",
        "y",
        "largeur",
        "hauteur",
        "chiffre_attendu",
        "prediction_top_1",
        "confiance_top_1",
        "rang_chiffre_attendu",
        "confiance_chiffre_attendu",
    ]

    for rang in range(1, 11):
        entetes.extend([f"top_{rang}_chiffre", f"top_{rang}_confiance"])

    with csv_path.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=entetes, delimiter=";")
        writer.writeheader()

        for resultat in resultats:
            x, y, largeur, hauteur = resultat["rectangle"]
            ligne = {
                "image_source": IMAGE_PATH.name,
                "modele": MODEL_NAME,
                "numero": resultat["numero"],
                "x": x,
                "y": y,
                "largeur": largeur,
                "hauteur": hauteur,
                "chiffre_attendu": CHIFFRE_ATTENDU,
                "prediction_top_1": resultat["prediction"],
                "confiance_top_1": f"{resultat['confiance_prediction']:.8f}",
                "rang_chiffre_attendu": resultat["rang_attendu"],
                "confiance_chiffre_attendu": (
                    f"{resultat['confiance_attendue']:.8f}"
                ),
            }

            for index, (chiffre, confiance) in enumerate(
                zip(resultat["classement"], resultat["probabilites"]),
                start=1,
            ):
                ligne[f"top_{index}_chiffre"] = chiffre
                ligne[f"top_{index}_confiance"] = f"{confiance:.8f}"

            writer.writerow(ligne)


def creer_resume(
    resultats,
    classement_global,
    resultat_detection,
    date_test,
):
    """Cree un resume organise et lisible."""
    total = len(resultats)
    date_affichee = date_test.strftime("%d.%m.%Y a %H:%M:%S %Z")
    lignes = [
        "TEST CASCADE TOP-N",
        "==================",
        f"Image utilisee : {IMAGE_PATH.name}",
        f"Chemin source  : {IMAGE_PATH}",
        f"Date du test   : {date_affichee}",
        f"Modele         : {MODEL_NAME}",
        f"Chiffre attendu: {CHIFFRE_ATTENDU}",
        "",
        "DETECTION",
        "---------",
        f"Nombre attendu : {NOMBRE_ATTENDU}",
        f"Nombre detecte : {total}",
        (
            f"Seuils faible/fort : "
            f"{resultat_detection.seuil_faible:.1f} / "
            f"{resultat_detection.seuil_fort:.1f}"
        ),
    ]

    if total != NOMBRE_ATTENDU:
        ecart = total - NOMBRE_ATTENDU
        lignes.append(
            "ATTENTION : le nombre detecte ne correspond pas au nombre "
            f"attendu (ecart {ecart:+d})."
        )

    lignes.extend(["", "CLASSEMENT GLOBAL", "-----------------"])

    for rang, (chiffre, probabilite) in enumerate(
        classement_global,
        start=1,
    ):
        lignes.append(
            f"Top {rang} : chiffre {chiffre} "
            f"avec {probabilite * 100:.2f} %"
        )

    if not classement_global:
        lignes.append("Aucun chiffre detecte: classement indisponible.")

    return "\n".join(lignes)


def main():
    """Lance la detection, la cascade Top-N et les sauvegardes."""
    verifier_configuration()
    date_test = datetime.now().astimezone()
    dossier_execution = creer_dossier_execution(date_test)
    image_path_sortie = dossier_execution / "resultat_annote.jpg"
    csv_path = dossier_execution / "predictions.csv"
    resume_path = dossier_execution / "resume.txt"
    diagnostics_dir = dossier_execution / "diagnostics"

    image = detection.charger_image(IMAGE_PATH)
    modele = keras.models.load_model(MODEL_PATH)
    resultat_detection = detection.detecter_chiffres(image)
    detection.sauvegarder_diagnostics(
        resultat_detection,
        image,
        diagnostics_dir,
    )

    rectangles = resultat_detection.rectangles
    probabilites = calculer_predictions(image, rectangles, modele)
    resultats = analyser_predictions(rectangles, probabilites)
    classement_global = calculer_classement_global(resultats)
    image_annotee = dessiner_resultats(image, resultats)

    if not cv2.imwrite(str(image_path_sortie), image_annotee):
        raise OSError(f"Impossible de sauvegarder l'image: {image_path_sortie}")

    sauvegarder_csv(csv_path, resultats)
    resume = creer_resume(
        resultats,
        classement_global,
        resultat_detection,
        date_test,
    )
    resume_path.write_text(resume + "\n", encoding="utf-8")

    print(resume)
    print(f"\nDossier complet du test: {dossier_execution}")


if __name__ == "__main__":
    main()
