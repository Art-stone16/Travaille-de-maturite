"""Teste la detection et la reconnaissance sur une photographie terrain.

Chaque lancement produit un dossier autonome et lisible contenant les
parametres, les diagnostics de detection, le controle du pretraitement et les
resultats. Les constantes ci-dessous restent les valeurs par defaut de la CLI.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import env_config
import cv2
import keras
import numpy as np

import detection_chiffres as detection


# Reglages par defaut --------------------------------------------------------

IMAGE_PATH = env_config.DONNEES_TESTS_TERRAIN / "historiques" / "test_terrain.jpg"
MODEL_NAME = "Best_COLOR_MAP"
MODEL_PATH = (
    env_config.MODELES_VALIDES
    / MODEL_NAME
    / "best_model.keras"
)
OUTPUT_DIR = env_config.SORTIES_TESTS_TERRAIN

# Mettre un entier de 0 a 9 pour forcer la valeur. Avec None, le script tente
# de la deduire du nom de l'image (par exemple CTN_7.jpg ou chiffre_7.jpg).
CHIFFRE_REEL = None
DEDUIRE_CHIFFRE_DEPUIS_NOM = True
NOM_EXPERIENCE = "test_terrain"

AFFICHER_IMAGES = False
SAUVEGARDER_DIAGNOSTICS = True


def analyser_arguments(argv=None):
    """Lit la CLI tout en conservant les anciens reglages comme valeurs par defaut."""
    chiffre_defaut = CHIFFRE_REEL

    if chiffre_defaut is None:
        chiffre_defaut = "auto" if DEDUIRE_CHIFFRE_DEPUIS_NOM else "inconnu"

    parser = argparse.ArgumentParser(
        description=(
            "Tester une photo terrain et sauvegarder les controles 28 x 28 "
            "dans un dossier d'execution structure."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=IMAGE_PATH,
        help=f"photo a tester (defaut: {IMAGE_PATH})",
    )
    parser.add_argument(
        "--modele",
        type=Path,
        default=MODEL_PATH,
        help=f"modele Keras a utiliser (defaut: {MODEL_PATH})",
    )
    parser.add_argument(
        "--nom-modele",
        default=None,
        help="nom affiche dans les resultats (defaut: dossier du modele)",
    )
    parser.add_argument(
        "--chiffre-reel",
        default=chiffre_defaut,
        metavar="AUTO|INCONNU|0..9",
        help=(
            "valeur attendue pour toute l'image; 'auto' la deduit du nom et "
            "'inconnu' laisse la colonne vide"
        ),
    )
    parser.add_argument(
        "--nom-experience",
        default=NOM_EXPERIENCE,
        help="nom du groupe de tests, par exemple papier_blanc_stylo_noir",
    )
    parser.add_argument(
        "--sortie",
        type=Path,
        default=OUTPUT_DIR,
        help=f"dossier racine des sorties (defaut: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--afficher",
        action="store_true",
        default=AFFICHER_IMAGES,
        help="afficher aussi les images dans des fenetres OpenCV",
    )
    parser.add_argument(
        "--sans-diagnostics",
        action="store_true",
        help="ne pas sauvegarder les masques de detection",
    )
    return parser.parse_args(argv)


def charger_modele(model_path):
    """Charge le modele Keras et signale clairement un chemin incorrect."""
    if not model_path.exists():
        raise ValueError(f"Modele introuvable: {model_path}")

    return keras.models.load_model(model_path)


def creer_chemin_sortie(output_dir):
    """Cree un chemin test_N.jpg sans ecraser une ancienne image.

    Cette fonction historique reste disponible pour les scripts qui
    l'importeraient. Le flux principal utilise des dossiers horodates.
    """
    numero_test = 1

    while True:
        output_path = Path(output_dir) / f"test_{numero_test}.jpg"

        if not output_path.exists():
            return output_path

        numero_test += 1


def nettoyer_nom_dossier(nom):
    """Transforme un libelle en nom de dossier simple et stable."""
    nom_nettoye = re.sub(r"[^A-Za-z0-9._-]+", "_", str(nom)).strip("._")
    return nom_nettoye or "sans_nom"


def creer_dossier_execution(
    output_dir,
    nom_experience,
    image_path,
    date_test,
):
    """Cree un dossier unique organise par experience, image et execution."""
    dossier_experience = Path(output_dir) / nettoyer_nom_dossier(nom_experience)
    dossier_image = dossier_experience / nettoyer_nom_dossier(image_path.stem)
    dossier_image.mkdir(parents=True, exist_ok=True)
    base = date_test.strftime("%Y-%m-%d_%H-%M-%S")
    dossier_execution = dossier_image / base
    suffixe = 2

    while dossier_execution.exists():
        dossier_execution = dossier_image / f"{base}_{suffixe:02d}"
        suffixe += 1

    dossier_execution.mkdir()

    dossiers = {
        "execution": dossier_execution,
        "source": dossier_execution / "00_source",
        "detection": dossier_execution / "01_detection",
        "pretraitement": dossier_execution / "02_pretraitement",
        "resultats": dossier_execution / "03_resultats",
    }

    for dossier in dossiers.values():
        dossier.mkdir(exist_ok=True)

    return dossiers


def formater_confiance(confiance):
    """Affiche trois decimales sans suggerer une certitude absolue."""
    if confiance >= 0.999995:
        return ">99.999%"

    return f"{confiance:.3%}"


def interpreter_chiffre_reel(valeur, image_path):
    """Retourne le chiffre attendu et l'origine de cette information."""
    if isinstance(valeur, int):
        if not 0 <= valeur <= 9:
            raise ValueError("CHIFFRE_REEL doit etre compris entre 0 et 9.")

        return valeur, "configuration"

    texte = str(valeur).strip().lower()

    if texte in {"inconnu", "vide", "none", "?"}:
        return None, "non_renseigne"

    if texte != "auto":
        if len(texte) == 1 and texte.isdigit():
            return int(texte), "argument_cli"

        raise ValueError(
            "--chiffre-reel doit valoir 'auto', 'inconnu' ou un chiffre de 0 a 9."
        )

    chiffres_trouves = re.findall(r"(?<!\d)([0-9])(?!\d)", image_path.stem)
    chiffres_uniques = {int(chiffre) for chiffre in chiffres_trouves}

    if len(chiffres_uniques) == 1:
        return chiffres_uniques.pop(), "nom_fichier"

    return None, "nom_fichier_ambigu_ou_sans_chiffre"


def verifier_configuration(args):
    """Valide les entrees avant de creer le dossier de sortie."""
    if not args.image.is_file():
        raise ValueError(f"Photo introuvable: {args.image}")

    if not args.modele.is_file():
        raise ValueError(f"Modele introuvable: {args.modele}")

    if not str(args.nom_experience).strip():
        raise ValueError("--nom-experience ne doit pas etre vide.")

    interpreter_chiffre_reel(args.chiffre_reel, args.image)


def preparer_chiffres(image, rectangles):
    """Prepare toutes les zones et conserve leurs controles intermediaires."""
    return [
        detection.preparer_chiffre_avec_details(image, rectangle)
        for rectangle in rectangles
    ]


def reconnaitre_chiffres(
    image,
    rectangles,
    modele,
    preparations=None,
    retourner_probabilites=False,
):
    """Calcule les predictions de toutes les zones en un seul lot.

    Par defaut, la valeur de retour historique (liste de tuples) est conservee.
    Le flux principal demande aussi les probabilites pour enrichir le CSV.
    """
    if not rectangles:
        predictions = []
        probabilites_lot = np.empty((0, 10), dtype=np.float32)
        return (
            (predictions, probabilites_lot)
            if retourner_probabilites
            else predictions
        )

    if preparations is None:
        preparations = preparer_chiffres(image, rectangles)

    if len(preparations) != len(rectangles):
        raise ValueError("Une preparation est requise pour chaque rectangle.")

    lot = np.concatenate(
        [preparation.tenseur_modele for preparation in preparations],
        axis=0,
    )
    probabilites_lot = np.asarray(modele.predict(lot, verbose=0))

    if probabilites_lot.ndim != 2 or len(probabilites_lot) != len(rectangles):
        raise ValueError(
            "Sortie du modele inattendue: "
            f"forme recue {probabilites_lot.shape}."
        )

    predictions = []

    for rectangle, probabilites in zip(rectangles, probabilites_lot):
        chiffre_predit = int(np.argmax(probabilites))
        confiance = float(probabilites[chiffre_predit])
        predictions.append((rectangle, chiffre_predit, confiance))

    if retourner_probabilites:
        return predictions, probabilites_lot

    return predictions


def dessiner_rectangles(
    image,
    predictions,
    nom_modele=MODEL_NAME,
    nom_image=None,
):
    """Dessine les rectangles, les predictions et leurs confiances."""
    image_encadree = image.copy()
    hauteur_image, largeur_image = image_encadree.shape[:2]
    echelle_titre = max(0.6, min(1.6, largeur_image / 1800))
    epaisseur_titre = max(2, round(echelle_titre * 3))
    nom_image = nom_image or IMAGE_PATH.name

    cv2.putText(
        image_encadree,
        f"Modele: {nom_modele} | Image: {nom_image}",
        (20, max(40, round(55 * echelle_titre))),
        cv2.FONT_HERSHEY_SIMPLEX,
        echelle_titre,
        (0, 0, 255),
        epaisseur_titre,
        cv2.LINE_AA,
    )

    for rectangle, chiffre_predit, confiance in predictions:
        x, y, largeur, hauteur = rectangle
        petite_dimension = min(largeur, hauteur)
        marge = max(3, round(petite_dimension * 0.08))
        echelle_texte = max(0.3, min(1.1, petite_dimension / 110))
        epaisseur = max(1, round(echelle_texte * 3))
        x1 = max(x - marge, 0)
        y1 = max(y - marge, 0)
        x2 = min(x + largeur + marge, largeur_image - 1)
        y2 = min(y + hauteur + marge, hauteur_image - 1)

        cv2.rectangle(
            image_encadree,
            (x1, y1),
            (x2, y2),
            color=(0, 0, 255),
            thickness=max(2, epaisseur),
        )
        cv2.putText(
            image_encadree,
            f"{chiffre_predit} ({formater_confiance(confiance)})",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            echelle_texte,
            (0, 0, 255),
            epaisseur,
            cv2.LINE_AA,
        )

    return image_encadree


def sauvegarder_image(chemin, image):
    """Sauvegarde une image et refuse les echecs silencieux d'OpenCV."""
    chemin.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(chemin), image):
        raise OSError(f"Impossible de sauvegarder l'image: {chemin}")


def _placer_image(planche, image, x, y, largeur, hauteur, interpolation):
    """Centre une image dans une zone sans la deformer."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    hauteur_image, largeur_image = image.shape[:2]
    facteur = min(largeur / largeur_image, hauteur / hauteur_image)
    nouvelle_largeur = max(1, round(largeur_image * facteur))
    nouvelle_hauteur = max(1, round(hauteur_image * facteur))
    redimensionnee = cv2.resize(
        image,
        (nouvelle_largeur, nouvelle_hauteur),
        interpolation=interpolation,
    )
    x_depart = x + (largeur - nouvelle_largeur) // 2
    y_depart = y + (hauteur - nouvelle_hauteur) // 2
    planche[
        y_depart : y_depart + nouvelle_hauteur,
        x_depart : x_depart + nouvelle_largeur,
    ] = redimensionnee


def creer_planche_qualite(
    image,
    numero,
    rectangle,
    preparation,
    prediction,
    confiance,
    chiffre_reel,
    origine_chiffre_reel,
    nom_modele,
):
    """Cree la planche original/recadrage/28 x 28 gris/binaire."""
    largeur_case = 320
    hauteur_visuel = 300
    marge = 18
    hauteur_entete = 65
    hauteur_titre = 42
    hauteur_pied = 92
    largeur_planche = largeur_case * 4 + marge * 2
    hauteur_planche = (
        hauteur_entete + hauteur_titre + hauteur_visuel + hauteur_pied
    )
    planche = np.full(
        (hauteur_planche, largeur_planche, 3),
        245,
        dtype=np.uint8,
    )

    cv2.putText(
        planche,
        f"Controle qualite #{numero:03d} | Modele: {nom_modele}",
        (marge, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )

    original_localise = image.copy()
    x, y, largeur, hauteur = rectangle
    epaisseur = max(2, round(min(image.shape[:2]) / 700))
    cv2.rectangle(
        original_localise,
        (x, y),
        (x + largeur, y + hauteur),
        (0, 0, 255),
        epaisseur,
    )

    binaire_visible = preparation.matrice_binaire_0_1 * 255
    panneaux = [
        ("Original localise", original_localise, cv2.INTER_AREA),
        ("Recadrage detecte", preparation.recadrage_original, cv2.INTER_AREA),
        (
            "28 x 28 - niveaux de gris",
            preparation.image_28_niveaux_gris,
            cv2.INTER_NEAREST,
        ),
        ("28 x 28 - matrice 0/1", binaire_visible, cv2.INTER_NEAREST),
    ]
    y_titre = hauteur_entete + 26
    y_visuel = hauteur_entete + hauteur_titre

    for index, (titre, contenu, interpolation) in enumerate(panneaux):
        x_case = marge + index * largeur_case
        cv2.putText(
            planche,
            titre,
            (x_case + 6, y_titre),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            planche,
            (x_case + 5, y_visuel),
            (x_case + largeur_case - 5, y_visuel + hauteur_visuel),
            (180, 180, 180),
            1,
        )
        _placer_image(
            planche,
            contenu,
            x_case + 6,
            y_visuel + 1,
            largeur_case - 12,
            hauteur_visuel - 2,
            interpolation,
        )

    reel_affiche = "inconnu" if chiffre_reel is None else str(chiffre_reel)
    correct = (
        "non evaluable"
        if chiffre_reel is None
        else ("oui" if prediction == chiffre_reel else "non")
    )
    y_pied = y_visuel + hauteur_visuel + 36
    cv2.putText(
        planche,
        (
            f"Reel: {reel_affiche} ({origine_chiffre_reel}) | "
            f"Prediction: {prediction} | "
            f"Confiance: {formater_confiance(confiance)} | Correct: {correct}"
        ),
        (marge, y_pied),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        planche,
        "Le modele recoit la version en niveaux de gris normalisee entre 0 et 1.",
        (marge, y_pied + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return planche


def sauvegarder_controles_qualite(
    dossier_pretraitement,
    dossier_execution,
    image,
    predictions,
    preparations,
    chiffre_reel,
    origine_chiffre_reel,
    nom_modele,
):
    """Sauvegarde une fiche complete et une matrice 0/1 par chiffre."""
    artefacts = []

    for numero, (prediction_info, preparation) in enumerate(
        zip(predictions, preparations),
        start=1,
    ):
        rectangle, prediction, confiance = prediction_info
        dossier_chiffre = dossier_pretraitement / f"chiffre_{numero:03d}"
        dossier_chiffre.mkdir(parents=True, exist_ok=True)
        chemin_recadrage = dossier_chiffre / "01_recadrage_original.png"
        chemin_gris = dossier_chiffre / "02_entree_modele_28x28_gris.png"
        chemin_binaire = dossier_chiffre / "03_matrice_28x28_binaire.png"
        chemin_matrice = dossier_chiffre / "04_matrice_28x28_0_1.csv"
        chemin_planche = dossier_chiffre / "planche_controle_qualite.png"

        sauvegarder_image(chemin_recadrage, preparation.recadrage_original)
        sauvegarder_image(chemin_gris, preparation.image_28_niveaux_gris)
        sauvegarder_image(
            chemin_binaire,
            preparation.matrice_binaire_0_1 * 255,
        )
        np.savetxt(
            chemin_matrice,
            preparation.matrice_binaire_0_1,
            fmt="%d",
            delimiter=";",
        )
        planche = creer_planche_qualite(
            image,
            numero,
            rectangle,
            preparation,
            prediction,
            confiance,
            chiffre_reel,
            origine_chiffre_reel,
            nom_modele,
        )
        sauvegarder_image(chemin_planche, planche)
        artefacts.append(
            {
                "planche_qc": chemin_planche.relative_to(
                    dossier_execution
                ).as_posix(),
                "matrice_0_1": chemin_matrice.relative_to(
                    dossier_execution
                ).as_posix(),
                "image_28_gris": chemin_gris.relative_to(
                    dossier_execution
                ).as_posix(),
            }
        )

    return artefacts


def construire_lignes_csv(
    predictions,
    probabilites_lot,
    artefacts,
    date_test,
    dossier_execution,
    image_path,
    model_path,
    nom_modele,
    nom_experience,
    chiffre_reel,
    origine_chiffre_reel,
):
    """Assemble les lignes detaillees du resultat terrain."""
    lignes = []

    for numero, (prediction_info, probabilites, fichiers) in enumerate(
        zip(predictions, probabilites_lot, artefacts),
        start=1,
    ):
        rectangle, prediction, confiance = prediction_info
        x, y, largeur, hauteur = rectangle
        ligne = {
            "date_iso": date_test.isoformat(),
            "experience": nom_experience,
            "execution": dossier_execution.name,
            "dossier_execution": str(dossier_execution.resolve()),
            "image_source": str(image_path.resolve()),
            "modele": nom_modele,
            "chemin_modele": str(model_path.resolve()),
            "numero": numero,
            "x": x,
            "y": y,
            "largeur": largeur,
            "hauteur": hauteur,
            "chiffre_reel": "" if chiffre_reel is None else chiffre_reel,
            "origine_chiffre_reel": origine_chiffre_reel,
            "prediction": prediction,
            "confiance": f"{confiance:.8f}",
            "correct": (
                ""
                if chiffre_reel is None
                else int(prediction == chiffre_reel)
            ),
            "planche_qc": fichiers["planche_qc"],
            "matrice_0_1": fichiers["matrice_0_1"],
            "image_28_gris": fichiers["image_28_gris"],
        }

        for classe, probabilite in enumerate(probabilites):
            ligne[f"probabilite_{classe}"] = f"{float(probabilite):.8f}"

        lignes.append(ligne)

    return lignes


def entetes_csv(nombre_classes):
    """Retourne l'ordre stable des colonnes des CSV."""
    entetes = [
        "date_iso",
        "experience",
        "execution",
        "dossier_execution",
        "image_source",
        "modele",
        "chemin_modele",
        "numero",
        "x",
        "y",
        "largeur",
        "hauteur",
        "chiffre_reel",
        "origine_chiffre_reel",
        "prediction",
        "confiance",
        "correct",
        "planche_qc",
        "matrice_0_1",
        "image_28_gris",
    ]
    entetes.extend(
        f"probabilite_{classe}" for classe in range(nombre_classes)
    )
    return entetes


def sauvegarder_csv(csv_path, lignes, nombre_classes=10, ajouter=False):
    """Ecrit un CSV par execution ou complete le journal de l'experience."""
    entetes = entetes_csv(nombre_classes)
    mode = "a" if ajouter else "w"
    ecrire_entete = not ajouter or not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(mode, newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=entetes, delimiter=";")

        if ecrire_entete:
            writer.writeheader()

        writer.writerows(lignes)


def creer_resume(
    args,
    nom_modele,
    date_test,
    resultat_detection,
    predictions,
    chiffre_reel,
    origine_chiffre_reel,
    dossier_execution,
):
    """Cree un resume court, lisible sans ouvrir le CSV."""
    nombre_correct = None
    accuracy = None

    if chiffre_reel is not None and predictions:
        nombre_correct = sum(
            prediction == chiffre_reel
            for _, prediction, _ in predictions
        )
        accuracy = nombre_correct / len(predictions)

    lignes = [
        "TEST EN CONDITIONS REELLES",
        "===========================",
        f"Date                 : {date_test.isoformat()}",
        f"Experience           : {args.nom_experience}",
        f"Image source         : {args.image.resolve()}",
        f"Modele               : {nom_modele}",
        f"Chemin du modele     : {args.modele.resolve()}",
        (
            "Chiffre reel         : "
            + ("inconnu" if chiffre_reel is None else str(chiffre_reel))
        ),
        f"Origine valeur reelle: {origine_chiffre_reel}",
        "",
        "DETECTION",
        "---------",
        f"Chiffres detectes    : {len(predictions)}",
        f"Seuil Otsu           : {resultat_detection.seuil_otsu:.3f}",
        f"Seuil faible         : {resultat_detection.seuil_faible:.3f}",
        f"Seuil fort           : {resultat_detection.seuil_fort:.3f}",
        f"Taille fond          : {resultat_detection.taille_fond}",
        f"Taille regroupement  : {resultat_detection.taille_regroupement}",
        "",
        "RESULTATS",
        "---------",
    ]

    if accuracy is None:
        lignes.append("Accuracy terrain     : non calculable (chiffre reel inconnu)")
    else:
        lignes.append(
            f"Accuracy terrain     : {nombre_correct}/{len(predictions)} "
            f"({accuracy:.2%})"
        )

    for numero, (_, prediction, confiance) in enumerate(predictions, start=1):
        lignes.append(
            f"Chiffre {numero:03d}        : {prediction} "
            f"({formater_confiance(confiance)})"
        )

    if not predictions:
        lignes.append("Aucun chiffre detecte.")

    lignes.extend(
        [
            "",
            "REPERAGE DES SORTIES",
            "--------------------",
            "00_source        : parametres et chemins des entrees",
            "01_detection     : masques et composantes de detection",
            "02_pretraitement : un dossier QC et une matrice 0/1 par chiffre",
            "03_resultats     : image annotee, CSV et ce resume",
            f"Dossier complet : {dossier_execution.resolve()}",
        ]
    )
    return "\n".join(lignes)


def sauvegarder_parametres(
    chemin,
    args,
    nom_modele,
    date_test,
    chiffre_reel,
    origine_chiffre_reel,
    resultat_detection,
    dossier_execution,
    diagnostics_actifs,
):
    """Ecrit les metadonnees necessaires pour reproduire et classer le test."""
    parametres = {
        "format": "test_condition_reelle_v2",
        "execution": {
            "date_iso": date_test.isoformat(),
            "identifiant": dossier_execution.name,
            "experience": args.nom_experience,
            "commande": sys.argv,
        },
        "entrees": {
            "image_source": str(args.image.resolve()),
            "modele": str(args.modele.resolve()),
            "nom_modele": nom_modele,
            "chiffre_reel": chiffre_reel,
            "origine_chiffre_reel": origine_chiffre_reel,
        },
        "detection": {
            "diagnostics_sauvegardes": diagnostics_actifs,
            "configuration": asdict(detection.CONFIG_PAR_DEFAUT),
            "seuil_otsu": resultat_detection.seuil_otsu,
            "seuil_faible": resultat_detection.seuil_faible,
            "seuil_fort": resultat_detection.seuil_fort,
            "taille_fond": resultat_detection.taille_fond,
            "taille_regroupement": resultat_detection.taille_regroupement,
            "nombre_rectangles": len(resultat_detection.rectangles),
        },
        "pretraitement": {
            "taille_entree": [28, 28, 1],
            "valeurs_modele": "niveaux de gris normalises entre 0 et 1",
            "matrice_qc": "seuil 128, valeurs entieres 0 ou 1",
        },
    }
    chemin.write_text(
        json.dumps(parametres, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def creer_readme(dossier_execution, journal_global):
    """Explique l'arborescence au premier regard."""
    return "\n".join(
        [
            "CONTENU DE CETTE EXECUTION",
            "==========================",
            "00_source/parametres.json",
            "    Chemins des entrees, date, chiffre reel et seuils de detection.",
            "01_detection/",
            "    Etapes visuelles permettant de verifier la detection.",
            "02_pretraitement/chiffre_NNN/",
            "    Recadrage, entree 28 x 28, matrice 0/1 et planche QC.",
            "03_resultats/resultat_annote.jpg",
            "    Photo avec prediction et confiance.",
            "03_resultats/resultats_terrain.csv",
            "    Une ligne detaillee par chiffre detecte.",
            "03_resultats/resume.txt",
            "    Resume humainement lisible du test.",
            "",
            f"Dossier courant : {dossier_execution.resolve()}",
            f"Journal global  : {journal_global.resolve()}",
        ]
    )


def main(argv=None):
    """Lance la detection, la reconnaissance et toutes les sauvegardes."""
    args = analyser_arguments(argv)
    verifier_configuration(args)
    date_test = datetime.now().astimezone()
    chiffre_reel, origine_chiffre_reel = interpreter_chiffre_reel(
        args.chiffre_reel,
        args.image,
    )
    nom_modele = args.nom_modele or args.modele.parent.name or MODEL_NAME
    diagnostics_actifs = SAUVEGARDER_DIAGNOSTICS and not args.sans_diagnostics

    image = detection.charger_image(args.image)
    modele = charger_modele(args.modele)
    resultat_detection = detection.detecter_chiffres(image)
    preparations = preparer_chiffres(
        image,
        resultat_detection.rectangles,
    )
    predictions, probabilites_lot = reconnaitre_chiffres(
        image,
        resultat_detection.rectangles,
        modele,
        preparations=preparations,
        retourner_probabilites=True,
    )

    dossiers = creer_dossier_execution(
        args.sortie,
        args.nom_experience,
        args.image,
        date_test,
    )
    dossier_execution = dossiers["execution"]

    if diagnostics_actifs:
        detection.sauvegarder_diagnostics(
            resultat_detection,
            image,
            dossiers["detection"],
        )

    artefacts = sauvegarder_controles_qualite(
        dossiers["pretraitement"],
        dossier_execution,
        image,
        predictions,
        preparations,
        chiffre_reel,
        origine_chiffre_reel,
        nom_modele,
    )
    image_encadree = dessiner_rectangles(
        image,
        predictions,
        nom_modele=nom_modele,
        nom_image=args.image.name,
    )
    chemin_image_annotee = dossiers["resultats"] / "resultat_annote.jpg"
    sauvegarder_image(chemin_image_annotee, image_encadree)

    lignes_csv = construire_lignes_csv(
        predictions,
        probabilites_lot,
        artefacts,
        date_test,
        dossier_execution,
        args.image,
        args.modele,
        nom_modele,
        args.nom_experience,
        chiffre_reel,
        origine_chiffre_reel,
    )
    nombre_classes = (
        probabilites_lot.shape[1]
        if probabilites_lot.ndim == 2 and probabilites_lot.shape[1]
        else 10
    )
    chemin_csv = dossiers["resultats"] / "resultats_terrain.csv"
    sauvegarder_csv(chemin_csv, lignes_csv, nombre_classes=nombre_classes)

    journal_global = (
        args.sortie
        / nettoyer_nom_dossier(args.nom_experience)
        / "journal_global_tests_terrain.csv"
    )
    sauvegarder_csv(
        journal_global,
        lignes_csv,
        nombre_classes=nombre_classes,
        ajouter=True,
    )
    sauvegarder_parametres(
        dossiers["source"] / "parametres.json",
        args,
        nom_modele,
        date_test,
        chiffre_reel,
        origine_chiffre_reel,
        resultat_detection,
        dossier_execution,
        diagnostics_actifs,
    )
    resume = creer_resume(
        args,
        nom_modele,
        date_test,
        resultat_detection,
        predictions,
        chiffre_reel,
        origine_chiffre_reel,
        dossier_execution,
    )
    (dossiers["resultats"] / "resume.txt").write_text(
        resume + "\n",
        encoding="utf-8",
    )
    (dossier_execution / "README.txt").write_text(
        creer_readme(dossier_execution, journal_global) + "\n",
        encoding="utf-8",
    )

    print(resume)
    print(f"\nDossier complet du test: {dossier_execution}")
    print(f"Journal cumulatif: {journal_global}")

    if args.afficher:
        cv2.imshow("Image originale", image)
        cv2.imshow("Chiffres encadres", image_encadree)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
