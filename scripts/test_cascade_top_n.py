"""Mesure la cascade Top-N sur une feuille contenant un chiffre repete."""

import csv

import env_config
import cv2
import keras
import numpy as np

import test_condition_reelle as detection


# Reglages du test -----------------------------------------------------------

# Photo de la feuille sur laquelle le meme chiffre est ecrit 100 fois.
IMAGE_PATH = env_config.PROJECT_ROOT / "donnees" / "CTN_2.2.jpg"

# Verite terrain commune a tous les chiffres de la feuille.
CHIFFRE_ATTENDU = 2
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

# Un resultat qui n'est pas Top-1 est orange si la bonne reponse se trouve
# encore parmi les trois premieres propositions du modele.
TOP_N_ORANGE = 3


def verifier_configuration():
    """Verifie les reglages avant de charger le modele."""
    if not 0 <= CHIFFRE_ATTENDU <= 9:
        raise ValueError("CHIFFRE_ATTENDU doit etre compris entre 0 et 9.")

    if NOMBRE_ATTENDU <= 0:
        raise ValueError("NOMBRE_ATTENDU doit etre strictement positif.")

    if not IMAGE_PATH.exists():
        raise ValueError(
            "Photo introuvable. Place la photo ici ou modifie IMAGE_PATH: "
            f"{IMAGE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise ValueError(f"Modele introuvable: {MODEL_PATH}")


def creer_chemins_sortie():
    """Cree trois chemins portant le meme numero sans ecraser un ancien test."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numero = 1

    while (OUTPUT_DIR / f"cascade_{numero}.jpg").exists():
        numero += 1

    return (
        OUTPUT_DIR / f"cascade_{numero}.jpg",
        OUTPUT_DIR / f"cascade_{numero}.csv",
        OUTPUT_DIR / f"cascade_{numero}_resume.txt",
    )


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
            "Le modele doit produire dix probabilites, une pour chaque "
            f"chiffre. Forme recue: {probabilites.shape}"
        )

    return probabilites


def analyser_predictions(rectangles, probabilites):
    """Construit le classement et le rang du bon chiffre pour chaque zone."""
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
    """Classe les chiffres selon leur probabilite moyenne sur toute la feuille."""
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
    hauteur_image, largeur_image = image_annotee.shape[:2]
    echelle = max(0.45, min(1.0, largeur_image / 3000))
    epaisseur = max(2, round(echelle * 3))

    cv2.putText(
        image_annotee,
        f"Modele: {MODEL_NAME} | chiffre attendu: {CHIFFRE_ATTENDU}",
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
            couleur = (0, 180, 0)  # vert
        elif rang <= TOP_N_ORANGE:
            couleur = (0, 165, 255)  # orange
        else:
            couleur = (0, 0, 255)  # rouge

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
        position_y = max(y - 8, 25)
        cv2.putText(
            image_annotee,
            texte,
            (x, position_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            echelle * 0.55,
            couleur,
            max(1, epaisseur - 1),
            cv2.LINE_AA,
        )

    return image_annotee


def sauvegarder_csv(csv_path, resultats):
    """Sauvegarde le classement detaille de chaque chiffre."""
    entetes = [
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


def creer_resume(resultats, classement_global):
    """Cree le compte rendu lisible dans le terminal et dans un fichier."""
    total = len(resultats)
    lignes = []

    for rang, (chiffre, probabilite) in enumerate(
        classement_global,
        start=1,
    ):
        pourcentage = probabilite * 100
        lignes.append(
            f"Top {rang} : chiffre {chiffre} avec {pourcentage:.2f} %"
        )

    if not classement_global:
        lignes.append("Aucun chiffre detecte: classement Top-N indisponible.")

    lignes.extend([
        "",
        f"Modele: {MODEL_NAME}",
        f"Image: {IMAGE_PATH}",
        f"Chiffre attendu: {CHIFFRE_ATTENDU}",
        f"Chiffres attendus: {NOMBRE_ATTENDU}",
        f"Chiffres detectes: {total}",
    ])

    if total != NOMBRE_ATTENDU:
        ecart = total - NOMBRE_ATTENDU
        lignes.append(
            "ATTENTION: le nombre detecte ne correspond pas au nombre "
            f"attendu (ecart: {ecart:+d})."
        )

    return "\n".join(lignes)


def main():
    """Lance la detection, la cascade Top-N et les sauvegardes."""
    verifier_configuration()
    image_path_sortie, csv_path, resume_path = creer_chemins_sortie()

    image = detection.charger_image(IMAGE_PATH)
    modele = keras.models.load_model(MODEL_PATH)

    # Evite que le module reutilise sauvegarde ses diagnostics dans le dossier
    # d'un autre test. Le nouveau script ne garde que ses propres resultats.
    detection.SAUVEGARDER_DIAGNOSTICS = False
    rectangles = detection.detecter_chiffres(image)
    probabilites = calculer_predictions(image, rectangles, modele)
    resultats = analyser_predictions(rectangles, probabilites)
    classement_global = calculer_classement_global(resultats)

    image_annotee = dessiner_resultats(image, resultats)
    if not cv2.imwrite(str(image_path_sortie), image_annotee):
        raise OSError(f"Impossible de sauvegarder l'image: {image_path_sortie}")

    sauvegarder_csv(csv_path, resultats)
    resume = creer_resume(resultats, classement_global)
    resume_path.write_text(resume + "\n", encoding="utf-8")

    print(resume)
    print("\nFichiers sauvegardes:")
    print(f"- Image annotee: {image_path_sortie}")
    print(f"- Details CSV: {csv_path}")
    print(f"- Resume: {resume_path}")


if __name__ == "__main__":
    main()
