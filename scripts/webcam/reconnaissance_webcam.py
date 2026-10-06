"""Reconnaissance de chiffres manuscrits en direct avec la webcam.

Le flux reste local : aucune image n'est envoyée sur Internet. Le détecteur
repère les zones sombres, les transforme en entrées 28 x 28, puis le modèle
Keras prédit tous les chiffres détectés en un seul lot.

Lancement :
    .venv/bin/python scripts/webcam/reconnaissance_webcam.py

Touches :
    q ou Échap : quitter
    s          : sauvegarder l'image annotée courante
    r          : démarrer ou arrêter l'enregistrement vidéo annoté

Cliquer sur « Modele » dans l'image pour choisir l'un des six modèles.
"""

from __future__ import annotations

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401


import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from reconnaissance_chiffres import config as env_config

import cv2
import numpy as np

from reconnaissance_chiffres import detection as detection
from scripts.evaluation import test_condition_reelle as terrain


CAMERA_PAR_DEFAUT = 0
LARGEUR_PAR_DEFAUT = 1280
HAUTEUR_PAR_DEFAUT = 720
TRAITER_CHAQUE_PAR_DEFAUT = 2
SEUIL_CONFIANCE_PAR_DEFAUT = 0.70
NOM_FENETRE = "Reconnaissance de chiffres en direct"
SORTIE_PAR_DEFAUT = env_config.SORTIES_WEBCAM
DOSSIER_MODELES = env_config.PROJECT_ROOT / "modeles" / "actifs"
LARGEUR_MENU = 330
HAUTEUR_LIGNE = 32
HAUT_MENU = 115


def entier_positif(valeur: str) -> int:
    nombre = int(valeur)
    if nombre <= 0:
        raise argparse.ArgumentTypeError("la valeur doit être supérieure à zéro")
    return nombre


def entier_non_negatif(valeur: str) -> int:
    nombre = int(valeur)
    if nombre < 0:
        raise argparse.ArgumentTypeError("la valeur doit être positive ou nulle")
    return nombre


def probabilite(valeur: str) -> float:
    nombre = float(valeur)
    if not 0 <= nombre <= 1:
        raise argparse.ArgumentTypeError("la valeur doit être comprise entre 0 et 1")
    return nombre


def analyser_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Ouvrir une webcam, détecter les chiffres manuscrits et afficher "
            "les prédictions du modèle en direct."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=entier_non_negatif,
        default=CAMERA_PAR_DEFAUT,
        help="indice de la caméra ; essayer 1 si une caméra externe est utilisée",
    )
    parser.add_argument(
        "--modele",
        type=Path,
        default=terrain.MODEL_PATH,
        help="fichier Keras à utiliser",
    )
    parser.add_argument(
        "--nom-modele",
        default=None,
        help="nom affiché ; par défaut, nom du dossier du modèle",
    )
    parser.add_argument(
        "--largeur",
        type=entier_positif,
        default=LARGEUR_PAR_DEFAUT,
        help="largeur demandée à la webcam",
    )
    parser.add_argument(
        "--hauteur",
        type=entier_positif,
        default=HAUTEUR_PAR_DEFAUT,
        help="hauteur demandée à la webcam",
    )
    parser.add_argument(
        "--traiter-chaque",
        type=entier_positif,
        default=TRAITER_CHAQUE_PAR_DEFAUT,
        metavar="N",
        help="effectuer la détection toutes les N images pour alléger le calcul",
    )
    parser.add_argument(
        "--seuil-confiance",
        type=probabilite,
        default=SEUIL_CONFIANCE_PAR_DEFAUT,
        help="seuil au-dessus duquel un résultat est dessiné en vert",
    )
    parser.add_argument(
        "--sortie",
        type=Path,
        default=SORTIE_PAR_DEFAUT,
        help="dossier utilisé pour sauvegarder les captures et les vidéos",
    )
    return parser.parse_args(argv)


def analyser_frame(image: np.ndarray, modele):
    """Détecte et reconnaît tous les chiffres visibles dans une image BGR."""
    resultat_detection = detection.detecter_chiffres(image)
    preparations = terrain.preparer_chiffres(
        image,
        resultat_detection.rectangles,
    )
    predictions, probabilites = terrain.reconnaitre_chiffres(
        image,
        resultat_detection.rectangles,
        modele,
        preparations=preparations,
        retourner_probabilites=True,
    )
    return predictions, probabilites


def dessiner_predictions(
    image: np.ndarray,
    predictions,
    seuil_confiance: float,
) -> np.ndarray:
    """Dessine les rectangles sans masquer les prédictions peu confiantes."""
    resultat = image.copy()
    hauteur_image, largeur_image = resultat.shape[:2]

    for rectangle, chiffre, confiance in predictions:
        x, y, largeur, hauteur = rectangle
        couleur = (
            (40, 210, 40)
            if confiance >= seuil_confiance
            else (0, 165, 255)
        )
        petite_dimension = min(largeur, hauteur)
        marge = max(3, round(petite_dimension * 0.06))
        epaisseur = max(2, round(petite_dimension / 45))
        x1 = max(0, x - marge)
        y1 = max(0, y - marge)
        x2 = min(largeur_image - 1, x + largeur + marge)
        y2 = min(hauteur_image - 1, y + hauteur + marge)
        cv2.rectangle(resultat, (x1, y1), (x2, y2), couleur, epaisseur)

        texte = f"{chiffre}  {terrain.formater_confiance(confiance)}"
        echelle = max(0.45, min(0.9, petite_dimension / 85))
        (largeur_texte, hauteur_texte), base = cv2.getTextSize(
            texte,
            cv2.FONT_HERSHEY_SIMPLEX,
            echelle,
            epaisseur,
        )
        y_texte = max(hauteur_texte + base + 4, y1 - 6)
        cv2.rectangle(
            resultat,
            (x1, y_texte - hauteur_texte - base - 4),
            (min(largeur_image - 1, x1 + largeur_texte + 6), y_texte + base),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            resultat,
            texte,
            (x1 + 3, y_texte),
            cv2.FONT_HERSHEY_SIMPLEX,
            echelle,
            couleur,
            epaisseur,
            cv2.LINE_AA,
        )

    return resultat


def ajouter_bandeau(
    image: np.ndarray,
    nom_modele: str,
    nombre_predictions: int,
    duree_analyse_ms: float,
    fps: float,
    seuil_confiance: float,
    message: str | None = None,
    enregistrement_video: bool = False,
) -> np.ndarray:
    """Ajoute les informations utiles sans modifier l'image analysée."""
    resultat = image.copy()
    largeur = resultat.shape[1]
    hauteur_bandeau = 82 if message is None else 108
    cv2.rectangle(resultat, (0, 0), (largeur, hauteur_bandeau), (25, 25, 25), -1)
    cv2.putText(
        resultat,
        (
            f"Modele: {nom_modele} | chiffres: {nombre_predictions} | "
            f"analyse: {duree_analyse_ms:.0f} ms | affichage: {fps:.1f} FPS"
        ),
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        resultat,
        (
            f"Vert >= {seuil_confiance:.0%} | orange = incertain | "
            "Q/Echap: quitter | S: capture | R: video"
        ),
        (14, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    if enregistrement_video:
        cv2.circle(resultat, (largeur - 76, 25), 9, (0, 0, 255), -1)
        cv2.putText(
            resultat,
            "REC",
            (largeur - 59, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if message:
        cv2.putText(
            resultat,
            message,
            (14, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return resultat


def sauvegarder_capture(
    image: np.ndarray,
    dossier_racine: Path,
    nom_modele: str,
    date_session: datetime,
    numero: int,
) -> Path:
    dossier = (
        dossier_racine
        / terrain.nettoyer_nom_dossier(nom_modele)
        / date_session.strftime("%Y-%m-%d_%H-%M-%S")
    )
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"capture_{numero:03d}.jpg"
    if not cv2.imwrite(str(chemin), image):
        raise OSError(f"Impossible de sauvegarder la capture: {chemin}")
    return chemin


def creer_enregistreur_video(
    image: np.ndarray,
    dossier_racine: Path,
    nom_modele: str,
    date_session: datetime,
    numero: int,
    fps: float,
):
    """Crée un fichier MP4 adapté aux dimensions exactes de l'affichage."""
    dossier = (
        dossier_racine
        / terrain.nettoyer_nom_dossier(nom_modele)
        / date_session.strftime("%Y-%m-%d_%H-%M-%S")
    )
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"video_{numero:03d}.mp4"
    hauteur, largeur = image.shape[:2]
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    enregistreur = cv2.VideoWriter(
        str(chemin),
        codec,
        fps,
        (largeur, hauteur),
    )
    if not enregistreur.isOpened():
        enregistreur.release()
        raise OSError(f"Impossible de créer la vidéo: {chemin}")
    return enregistreur, chemin


def choisir_fps_video(capture, fps_affichage: float) -> float:
    """Choisit une cadence réaliste, avec repli si la caméra ne la fournit pas."""
    if np.isfinite(fps_affichage) and 1 <= fps_affichage <= 60:
        return float(fps_affichage)
    fps_camera = float(capture.get(cv2.CAP_PROP_FPS))
    if np.isfinite(fps_camera) and 1 <= fps_camera <= 60:
        return fps_camera
    return 30.0


def modeles_disponibles():
    """Liste les modèles complets disponibles dans la recherche."""
    return sorted(DOSSIER_MODELES.glob("*/best_model.keras"))


def charger_modele_pret(chemin):
    modele = terrain.charger_modele(chemin)
    modele.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)
    return modele


def dessiner_menu_modele(image, noms, nom_actuel, ouvert):
    """Dessine un menu cliquable directement sur l'image OpenCV."""
    x = max(0, image.shape[1] - LARGEUR_MENU - 12)
    y = HAUT_MENU
    lignes = [f"Modele: {nom_actuel}  {'^' if ouvert else 'v'}"]
    if ouvert:
        lignes.extend(noms)
    for index, libelle in enumerate(lignes):
        haut = y + index * HAUTEUR_LIGNE
        bas = haut + HAUTEUR_LIGNE
        couleur = (55, 75, 85) if index == 0 else (38, 38, 38)
        cv2.rectangle(image, (x, haut), (x + LARGEUR_MENU, bas), couleur, -1)
        cv2.rectangle(image, (x, haut), (x + LARGEUR_MENU, bas), (180, 180, 180), 1)
        cv2.putText(image, libelle, (x + 9, haut + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)


def clic_menu_modele(etat, noms, largeur_image, evenement, x, y, _flags, _param):
    if evenement != cv2.EVENT_LBUTTONDOWN:
        return
    gauche = max(0, largeur_image - LARGEUR_MENU - 12)
    if not gauche <= x <= gauche + LARGEUR_MENU:
        etat["ouvert"] = False
        return
    if HAUT_MENU <= y < HAUT_MENU + HAUTEUR_LIGNE:
        etat["ouvert"] = not etat["ouvert"]
    elif etat["ouvert"]:
        index = (y - HAUT_MENU) // HAUTEUR_LIGNE - 1
        if 0 <= index < len(noms):
            etat["demande"] = index
        etat["ouvert"] = False


def ouvrir_camera(indice: int, largeur: int, hauteur: int):
    """Ouvre la caméra et produit un message adapté aux permissions macOS."""
    capture = cv2.VideoCapture(indice)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Impossible d'ouvrir la caméra {indice}. Sur macOS, autorisez le "
            "Terminal ou votre éditeur dans Réglages Système > "
            "Confidentialité et sécurité > Caméra."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def webcam_live(args) -> None:
    """Exécute la boucle interactive et libère toujours la caméra à la fin."""
    chemins_modeles = modeles_disponibles()
    noms_modeles = [chemin.parent.name for chemin in chemins_modeles]
    modele = charger_modele_pret(args.modele)
    nom_modele = args.nom_modele or args.modele.parent.name
    chemin_actif = args.modele
    capture = ouvrir_camera(args.camera, args.largeur, args.hauteur)
    date_session = datetime.now().astimezone()
    predictions = []
    duree_analyse_ms = 0.0
    numero_frame = 0
    numero_capture = 0
    numero_video = 0
    enregistreur_video = None
    chemin_video = None
    dernier_temps = time.perf_counter()
    fps_lisse = 0.0
    message = None
    fin_message = 0.0
    etat_menu = {"ouvert": False, "demande": None}
    chargement = None
    nom_en_chargement = None
    chargeur = ThreadPoolExecutor(max_workers=1)

    print(
        f"Caméra {args.camera} ouverte. Q ou Échap pour quitter, "
        "S pour capturer, R pour enregistrer une vidéo. "
        "Cliquer sur Modele pour changer de modèle."
    )
    try:
        cv2.namedWindow(NOM_FENETRE, cv2.WINDOW_AUTOSIZE)
        while True:
            lu, image = capture.read()
            if not lu or image is None:
                raise RuntimeError("La webcam ne fournit plus d'image.")
            numero_frame += 1
            if numero_frame == 1:
                cv2.setMouseCallback(
                    NOM_FENETRE,
                    lambda evenement, x, y, flags, param: clic_menu_modele(
                        etat_menu, noms_modeles, image.shape[1],
                        evenement, x, y, flags, param,
                    ),
                )

            if chargement is not None and chargement.done():
                try:
                    nouveau_modele = chargement.result()
                except Exception as exc:
                    message = f"Chargement impossible: {nom_en_chargement}"
                    print(f"{message}: {exc}")
                else:
                    modele = nouveau_modele
                    nom_modele = nom_en_chargement
                    chemin_actif = chemin_en_chargement
                    predictions = []
                    message = f"Modele actif: {nom_modele}"
                    print(message)
                fin_message = time.perf_counter() + 3
                chargement = None

            demande = etat_menu.get("demande")
            if demande is not None and chargement is None:
                etat_menu["demande"] = None
                chemin = chemins_modeles[demande]
                if chemin != chemin_actif:
                    nom_en_chargement = chemin.parent.name
                    chemin_en_chargement = chemin
                    chargement = chargeur.submit(charger_modele_pret, chemin)
                    message = f"Chargement: {nom_en_chargement}"
                    fin_message = time.perf_counter() + 3600

            if numero_frame == 1 or numero_frame % args.traiter_chaque == 0:
                debut_analyse = time.perf_counter()
                predictions, _ = analyser_frame(image, modele)
                duree_analyse_ms = (time.perf_counter() - debut_analyse) * 1000

            image_annotee = dessiner_predictions(
                image,
                predictions,
                args.seuil_confiance,
            )
            maintenant = time.perf_counter()
            fps_instantane = 1.0 / max(maintenant - dernier_temps, 1e-9)
            dernier_temps = maintenant
            fps_lisse = (
                fps_instantane
                if fps_lisse == 0
                else 0.90 * fps_lisse + 0.10 * fps_instantane
            )
            if message is not None and maintenant >= fin_message:
                message = None
            image_affichee = ajouter_bandeau(
                image_annotee,
                nom_modele,
                len(predictions),
                duree_analyse_ms,
                fps_lisse,
                args.seuil_confiance,
                message=message,
                enregistrement_video=enregistreur_video is not None,
            )
            if enregistreur_video is not None:
                enregistreur_video.write(image_affichee)
            image_fenetre = image_affichee.copy()
            dessiner_menu_modele(image_fenetre, noms_modeles, nom_modele, etat_menu["ouvert"])
            cv2.imshow(NOM_FENETRE, image_fenetre)

            touche = cv2.waitKey(1) & 0xFF
            if touche in (ord("q"), ord("Q"), 27):
                break
            if touche in (ord("s"), ord("S")):
                numero_capture += 1
                chemin = sauvegarder_capture(
                    image_affichee,
                    args.sortie,
                    nom_modele,
                    date_session,
                    numero_capture,
                )
                message = f"Capture sauvegardee: {chemin.name}"
                fin_message = time.perf_counter() + 2.5
                print(f"Capture sauvegardée: {chemin}")
            if touche in (ord("r"), ord("R")):
                if enregistreur_video is None:
                    numero_video += 1
                    fps_video = choisir_fps_video(capture, fps_lisse)
                    enregistreur_video, chemin_video = creer_enregistreur_video(
                        image_affichee,
                        args.sortie,
                        nom_modele,
                        date_session,
                        numero_video,
                        fps_video,
                    )
                    message = f"Enregistrement demarre: {chemin_video.name}"
                    fin_message = time.perf_counter() + 2.5
                    print(
                        f"Enregistrement démarré ({fps_video:.1f} FPS): "
                        f"{chemin_video}"
                    )
                else:
                    enregistreur_video.release()
                    enregistreur_video = None
                    message = f"Video sauvegardee: {chemin_video.name}"
                    fin_message = time.perf_counter() + 2.5
                    print(f"Vidéo sauvegardée: {chemin_video}")

            try:
                fenetre_visible = cv2.getWindowProperty(
                    NOM_FENETRE,
                    cv2.WND_PROP_VISIBLE,
                )
            except cv2.error:
                fenetre_visible = 1
            if fenetre_visible < 1:
                break
    finally:
        chargeur.shutdown(wait=True)
        if enregistreur_video is not None:
            enregistreur_video.release()
            print(f"Vidéo sauvegardée: {chemin_video}")
        capture.release()
        cv2.destroyAllWindows()
        print("Webcam arrêtée et libérée.")


def main(argv=None) -> int:
    args = analyser_arguments(argv)
    webcam_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
