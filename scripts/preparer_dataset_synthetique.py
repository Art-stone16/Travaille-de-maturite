"""Prepare et controle un dataset synthetique de caracteres en 28 x 28.

Le script n'appelle aucun service d'IA et n'entraine aucun modele. Il sait :

* importer des images ou des matrices produites ailleurs (Claude, dessin, etc.);
* generer un petit baseline procedural clairement identifie comme tel;
* augmenter un dataset deja prepare;
* valider et visualiser un dataset sans le melanger a MNIST.

Chaque dataset cree possede son propre manifeste et des dossiers separes pour les
matrices, les apercus PNG et les rapports de qualite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import env_config

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENTREE_PAR_DEFAUT = env_config.PROJECT_ROOT / "donnees" / "synthetiques_a_importer"
SORTIE_PAR_DEFAUT = env_config.PROJECT_ROOT / "donnees" / "datasets_synthetiques"
FORMATS_IMAGE = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FORMATS_MATRICE = {".npy", ".npz", ".csv", ".txt", ".json"}
FORMATS_ACCEPTES = FORMATS_IMAGE | FORMATS_MATRICE
CHAMPS_MANIFESTE = (
    "sample_id",
    "label",
    "split",
    "origine",
    "source_type",
    "source_path",
    "source_index",
    "image_path",
    "matrix_path",
    "sha256",
    "quality_status",
    "quality_notes",
    "transformations",
)


@dataclass
class ElementBrut:
    donnees: np.ndarray
    etiquette: str | None
    split: str | None
    chemin_source: Path
    index_source: str
    type_source: str
    metadonnees_source: dict[str, Any] | None = None


@dataclass
class ResultatQualite:
    statut: str
    notes: list[str]
    statistiques: dict[str, float | int]


def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Importe, genere, augmente, valide ou visualise des caracteres "
            "synthetiques 28 x 28 sans entrainer de modele."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sous_commandes = parser.add_subparsers(dest="commande", required=True)

    importer = sous_commandes.add_parser(
        "importer",
        help="Importer des images/matrices et creer un dataset controle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    importer.add_argument(
        "--entree",
        type=Path,
        action="append",
        default=[],
        help=(
            "Fichier ou dossier a importer. L'option peut etre repetee. "
            f"Par defaut: {ENTREE_PAR_DEFAUT}"
        ),
    )
    _ajouter_arguments_creation(importer)
    importer.add_argument(
        "--etiquette",
        help=(
            "Etiquette forcee pour toutes les entrees. Sans cette option, elle "
            "vient du JSON, du premier sous-dossier ou du prefixe du fichier."
        ),
    )
    importer.add_argument(
        "--origine",
        default="ia_externe",
        help="Provenance explicite inscrite dans le manifeste (ex. claude).",
    )
    importer.add_argument(
        "--split",
        choices=("train", "validation", "test", "non_attribue"),
        default="non_attribue",
        help="Partition par defaut; un champ split valide du JSON est prioritaire.",
    )
    importer.add_argument(
        "--polarite",
        choices=("auto", "blanc-sur-noir", "noir-sur-blanc"),
        default="auto",
        help="Polarite des entrees avant conversion au format MNIST.",
    )
    importer.add_argument(
        "--binaire",
        action="store_true",
        help="Force tous les pixels finaux a 0 ou 1 apres redimensionnement.",
    )

    baseline = sous_commandes.add_parser(
        "generer-baseline",
        help="Creer un baseline procedural local, sans IA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ajouter_arguments_creation(baseline)
    baseline.add_argument(
        "--classes",
        nargs="+",
        default=[str(i) for i in range(10)],
        help="Caracteres a dessiner; les etiquettes de plusieurs caracteres sont permises.",
    )
    baseline.add_argument(
        "--par-classe",
        type=int,
        default=20,
        help="Nombre d'echantillons proceduraux par classe.",
    )
    baseline.add_argument("--seed", type=int, default=42, help="Graine aleatoire.")
    baseline.add_argument(
        "--binaire",
        action="store_true",
        help="Force les pixels finaux a 0 ou 1.",
    )

    augmenter = sous_commandes.add_parser(
        "augmenter",
        help="Creer un dataset derive par augmentations documentees.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    augmenter.add_argument("--dataset-source", type=Path, required=True)
    _ajouter_arguments_creation(augmenter)
    augmenter.add_argument(
        "--copies-par-image", type=int, default=2, help="Variantes par image source."
    )
    augmenter.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test", "non_attribue"),
        default=["train", "non_attribue"],
        help=(
            "Partitions sources a augmenter. Le test et la validation sont "
            "exclus par defaut pour eviter une fuite experimentale."
        ),
    )
    augmenter.add_argument(
        "--inclure-originaux",
        action="store_true",
        help="Copie aussi les matrices originales dans le nouveau dataset.",
    )
    augmenter.add_argument(
        "--rotation-max", type=float, default=12.0, help="Rotation absolue maximale en degres."
    )
    augmenter.add_argument(
        "--translation-max", type=float, default=2.0, help="Translation absolue maximale en pixels."
    )
    augmenter.add_argument(
        "--bruit-max", type=float, default=0.04, help="Ecart-type maximal du bruit gaussien."
    )
    augmenter.add_argument("--seed", type=int, default=42, help="Graine aleatoire.")
    augmenter.add_argument(
        "--binaire",
        action="store_true",
        help="Force les pixels finaux a 0 ou 1.",
    )

    valider = sous_commandes.add_parser(
        "valider",
        help="Verifier le manifeste et toutes les matrices d'un dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    valider.add_argument("--dataset", type=Path, required=True)
    valider.add_argument(
        "--rapport",
        type=Path,
        help="Chemin JSON du rapport; par defaut dans dataset/rapports/.",
    )

    visualiser = sous_commandes.add_parser(
        "visualiser",
        help="Regenerer une grille d'apercu depuis le manifeste.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    visualiser.add_argument("--dataset", type=Path, required=True)
    visualiser.add_argument("--sortie", type=Path, help="Chemin du PNG produit.")
    visualiser.add_argument(
        "--par-classe", type=int, default=8, help="Nombre maximal d'images par classe."
    )
    visualiser.add_argument(
        "--classes-max", type=int, default=20, help="Nombre maximal de classes dans une grille."
    )
    visualiser.add_argument("--dpi", type=int, default=160)

    return parser


def _ajouter_arguments_creation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sortie",
        type=Path,
        default=SORTIE_PAR_DEFAUT,
        help="Racine des datasets prepares (distincte des entrees brutes).",
    )
    parser.add_argument(
        "--nom-dataset",
        required=True,
        help="Nom du nouveau sous-dossier; un dataset existant n'est jamais ecrase.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="But ou contexte de l'experience, conserve dans le manifeste.",
    )
    parser.add_argument(
        "--apercus-par-classe",
        type=int,
        default=8,
        help="Nombre d'echantillons visibles par classe dans la grille automatique.",
    )


def slug(texte: str) -> str:
    texte_original = str(texte).strip()
    simplifie = re.sub(r"[^A-Za-z0-9._-]+", "_", texte_original).strip("._-")
    if not simplifie:
        simplifie = "classe"
    if simplifie != texte_original:
        suffixe = hashlib.sha1(texte_original.encode("utf-8")).hexdigest()[:8]
        simplifie = f"{simplifie}_{suffixe}"
    return simplifie[:80]


def creer_dossier_dataset(base: Path, nom: str) -> Path:
    base = base.expanduser().resolve()
    dossier = base / slug(nom)
    try:
        dossier.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Le dataset existe deja: {dossier}. Choisissez un autre --nom-dataset."
        ) from exc
    for sous_dossier in ("images", "matrices", "apercus", "rapports"):
        (dossier / sous_dossier).mkdir(parents=True, exist_ok=True)
    return dossier


def _chemin_est_dans(enfant: Path, parent: Path) -> bool:
    try:
        enfant.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def verifier_separation_entrees_sortie(entrees: Iterable[Path], sortie: Path) -> None:
    sortie = sortie.resolve()
    for entree in entrees:
        entree = entree.resolve()
        racine_entree = entree if entree.is_dir() else entree.parent
        if _chemin_est_dans(sortie, racine_entree):
            raise ValueError(
                "Le dossier de sortie ne doit pas etre place dans une entree brute: "
                f"sortie={sortie}, entree={racine_entree}."
            )


def decouvrir_fichiers(entrees: Iterable[Path]) -> list[tuple[Path, str | None]]:
    trouves: list[tuple[Path, str | None]] = []
    vus: set[Path] = set()
    for entree_brute in entrees:
        entree = entree_brute.expanduser().resolve()
        if not entree.exists():
            raise FileNotFoundError(f"Entree introuvable: {entree}")
        if entree.is_file():
            candidats = [(entree, None)]
        else:
            candidats = []
            for chemin in sorted(entree.rglob("*")):
                if not chemin.is_file() or chemin.suffix.lower() not in FORMATS_ACCEPTES:
                    continue
                relatif = chemin.relative_to(entree)
                etiquette_dossier = (
                    relatif.parts[0] if len(relatif.parts) > 1 else None
                )
                if etiquette_dossier is None and slug(entree.name) == entree.name:
                    if re.fullmatch(r"[A-Za-z0-9]{1,16}", entree.name):
                        etiquette_dossier = entree.name
                candidats.append((chemin, etiquette_dossier))
        for chemin, etiquette in candidats:
            if chemin.suffix.lower() not in FORMATS_ACCEPTES or chemin in vus:
                continue
            vus.add(chemin)
            trouves.append((chemin, etiquette))
    if not trouves:
        raise ValueError(
            "Aucun fichier compatible trouve. Formats acceptes: "
            + ", ".join(sorted(FORMATS_ACCEPTES))
        )
    return trouves


def inferer_etiquette_fichier(chemin: Path) -> str | None:
    correspondance = re.match(
        r"^(?:(?:digit|chiffre|caractere)[_-]?)?([^_-]+)[_-]",
        chemin.stem,
        flags=re.IGNORECASE,
    )
    return correspondance.group(1).strip() if correspondance else None


def _charger_texte_matrice(chemin: Path) -> np.ndarray:
    texte = chemin.read_text(encoding="utf-8").strip()
    if not texte:
        raise ValueError("fichier vide")
    delimiteur = "," if "," in texte.splitlines()[0] else None
    return np.loadtxt(chemin, delimiter=delimiteur)


def _extraire_json(
    objet: Any,
    chemin: Path,
) -> Iterator[
    tuple[np.ndarray, str | None, str | None, str, dict[str, Any]]
]:
    if isinstance(objet, dict) and "samples" in objet:
        echantillons = objet["samples"]
        metadonnees_racine = {
            str(cle): valeur
            for cle, valeur in objet.items()
            if cle != "samples"
        }
    elif isinstance(objet, list) and len(objet) == 28 and all(
        isinstance(ligne, list) for ligne in objet
    ):
        echantillons = [{"pixels": objet}]
        metadonnees_racine = {}
    elif isinstance(objet, list):
        echantillons = objet
        metadonnees_racine = {}
    elif isinstance(objet, dict):
        echantillons = [objet]
        metadonnees_racine = {}
    else:
        raise ValueError("JSON attendu: matrice, echantillon ou objet avec 'samples'")

    if not isinstance(echantillons, list):
        raise ValueError("Le champ JSON 'samples' doit etre une liste")
    for index, echantillon in enumerate(echantillons):
        if not isinstance(echantillon, dict) or "pixels" not in echantillon:
            raise ValueError(f"Echantillon JSON {index}: champ 'pixels' absent")
        etiquette = echantillon.get("label")
        split = echantillon.get("split")
        identifiant = str(echantillon.get("id", index))
        metadonnees = {
            str(cle): valeur
            for cle, valeur in echantillon.items()
            if cle not in {"pixels", "label", "split", "id"}
        }
        if metadonnees_racine:
            metadonnees["metadonnees_racine"] = metadonnees_racine
        yield np.asarray(echantillon["pixels"]), (
            None if etiquette is None else str(etiquette)
        ), (None if split is None else str(split)), identifiant, metadonnees


def charger_elements_fichier(
    chemin: Path,
    etiquette_suggeree: str | None,
) -> Iterator[ElementBrut]:
    extension = chemin.suffix.lower()
    etiquette_fichier = inferer_etiquette_fichier(chemin)
    etiquette_par_defaut = etiquette_suggeree or etiquette_fichier

    if extension in FORMATS_IMAGE:
        image = cv2.imread(str(chemin), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("image illisible")
        yield ElementBrut(
            image,
            etiquette_par_defaut,
            None,
            chemin,
            "0",
            "image_importee",
        )
        return

    if extension == ".npy":
        tableau = np.load(chemin, allow_pickle=False)
        tableaux = tableau[None, ...] if tableau.ndim == 2 else tableau
        if tableaux.ndim != 3:
            raise ValueError(".npy attendu: (28,28) ou (N,28,28)")
        for index, matrice in enumerate(tableaux):
            yield ElementBrut(
                matrice,
                etiquette_par_defaut,
                None,
                chemin,
                str(index),
                "matrice_importee",
            )
        return

    if extension == ".npz":
        with np.load(chemin, allow_pickle=False) as archive:
            for cle in archive.files:
                tableau = np.asarray(archive[cle])
                tableaux = tableau[None, ...] if tableau.ndim == 2 else tableau
                if tableaux.ndim != 3:
                    raise ValueError(
                        f"Cle NPZ {cle!r}: attendu (28,28) ou (N,28,28)"
                    )
                etiquette = etiquette_par_defaut or cle
                for index, matrice in enumerate(tableaux):
                    yield ElementBrut(
                        matrice,
                        etiquette,
                        None,
                        chemin,
                        f"{cle}:{index}",
                        "matrice_importee",
                    )
        return

    if extension in {".csv", ".txt"}:
        yield ElementBrut(
            _charger_texte_matrice(chemin),
            etiquette_par_defaut,
            None,
            chemin,
            "0",
            "matrice_importee",
        )
        return

    if extension == ".json":
        objet = json.loads(chemin.read_text(encoding="utf-8"))
        for matrice, etiquette, split, index, metadonnees in _extraire_json(
            objet, chemin
        ):
            yield ElementBrut(
                matrice,
                etiquette or etiquette_par_defaut,
                split,
                chemin,
                index,
                "matrice_ia_importee",
                metadonnees,
            )
        return

    raise ValueError(f"format non pris en charge: {extension}")


def _normaliser_valeurs(tableau: np.ndarray) -> np.ndarray:
    if not np.issubdtype(tableau.dtype, np.number):
        raise ValueError("pixels non numeriques")
    resultat = tableau.astype(np.float32)
    if not np.isfinite(resultat).all():
        raise ValueError("NaN ou valeur infinie")
    minimum = float(resultat.min())
    maximum = float(resultat.max())
    if minimum < 0:
        raise ValueError(f"pixel negatif ({minimum})")
    if maximum > 1:
        if maximum <= 255:
            resultat /= 255.0
        else:
            raise ValueError(f"pixel superieur a 255 ({maximum})")
    return np.clip(resultat, 0, 1)


def _convertir_gris(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"forme d'image non prise en charge: {image.shape}")


def _adapter_polarite(matrice: np.ndarray, polarite: str) -> np.ndarray:
    if polarite == "noir-sur-blanc":
        return 1.0 - matrice
    if polarite == "blanc-sur-noir":
        return matrice
    bord = np.concatenate(
        (matrice[0], matrice[-1], matrice[1:-1, 0], matrice[1:-1, -1])
    )
    if float(np.median(bord)) > 0.5:
        return 1.0 - matrice
    return matrice


def _recadrer_image_vers_28(matrice: np.ndarray) -> np.ndarray:
    uint8 = np.rint(matrice * 255).astype(np.uint8)
    seuil, masque = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if seuil <= 0 and uint8.max() > 0:
        masque = np.where(uint8 > 0, 255, 0).astype(np.uint8)
    points = cv2.findNonZero(masque)
    if points is None:
        raise ValueError("aucun trait detecte apres normalisation")
    x, y, largeur, hauteur = cv2.boundingRect(points)
    contenu = matrice[y : y + hauteur, x : x + largeur]
    cote = max(largeur, hauteur)
    marge = max(2, round(cote * 0.20))
    canevas = np.zeros((cote + 2 * marge, cote + 2 * marge), dtype=np.float32)
    y0 = marge + (cote - hauteur) // 2
    x0 = marge + (cote - largeur) // 2
    canevas[y0 : y0 + hauteur, x0 : x0 + largeur] = contenu
    interpolation = cv2.INTER_AREA if canevas.shape[0] > 28 else cv2.INTER_CUBIC
    return np.clip(cv2.resize(canevas, (28, 28), interpolation=interpolation), 0, 1)


def preparer_element(
    element: ElementBrut,
    polarite: str,
    binaire: bool,
) -> np.ndarray:
    if element.type_source == "image_importee":
        gris = _convertir_gris(element.donnees)
        matrice = _normaliser_valeurs(gris)
        matrice = _adapter_polarite(matrice, polarite)
        matrice = _recadrer_image_vers_28(matrice)
    else:
        matrice = np.squeeze(np.asarray(element.donnees))
        if matrice.shape != (28, 28):
            raise ValueError(
                f"matrice attendue en 28 x 28; forme recue: {matrice.shape}"
            )
        matrice = _adapter_polarite(_normaliser_valeurs(matrice), polarite)
    if binaire:
        matrice = (matrice >= 0.5).astype(np.float32)
    return np.asarray(np.clip(matrice, 0, 1), dtype=np.float32)


def evaluer_qualite(matrice: np.ndarray) -> ResultatQualite:
    notes: list[str] = []
    if matrice.shape != (28, 28):
        return ResultatQualite("invalide", [f"forme={matrice.shape}"], {})
    if not np.isfinite(matrice).all():
        return ResultatQualite("invalide", ["NaN ou infini"], {})
    minimum = float(matrice.min())
    maximum = float(matrice.max())
    moyenne = float(matrice.mean())
    actifs = int(np.count_nonzero(matrice >= 0.2))
    proportion_active = actifs / matrice.size
    bord = np.concatenate(
        (matrice[0], matrice[-1], matrice[1:-1, 0], matrice[1:-1, -1])
    )
    moyenne_bord = float(bord.mean())
    if minimum < 0 or maximum > 1:
        return ResultatQualite(
            "invalide",
            [f"plage hors de [0,1]: {minimum:.4f}..{maximum:.4f}"],
            {},
        )
    if maximum < 0.05 or actifs == 0:
        return ResultatQualite("invalide", ["matrice vide ou presque vide"], {})
    if proportion_active < 0.005:
        notes.append("tres peu de pixels actifs")
    if proportion_active > 0.65:
        notes.append("plus de 65 % des pixels sont actifs")
    if moyenne_bord > 0.15:
        notes.append("encre importante sur le bord")
    statut = "avertissement" if notes else "valide"
    return ResultatQualite(
        statut,
        notes,
        {
            "minimum": minimum,
            "maximum": maximum,
            "moyenne": moyenne,
            "pixels_actifs": actifs,
            "proportion_active": proportion_active,
            "moyenne_bord": moyenne_bord,
        },
    )


def empreinte_matrice(matrice: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrice).tobytes()).hexdigest()


def sauvegarder_echantillon(
    dossier: Path,
    numero: int,
    etiquette: str,
    split: str,
    origine: str,
    type_source: str,
    chemin_source: str,
    index_source: str,
    matrice: np.ndarray,
    qualite: ResultatQualite,
    transformations: dict[str, Any] | None = None,
) -> dict[str, str]:
    identifiant = f"s{numero:06d}"
    dossier_classe = slug(etiquette)
    chemin_image = Path("images") / dossier_classe / f"{identifiant}.png"
    chemin_matrice = Path("matrices") / dossier_classe / f"{identifiant}.npy"
    (dossier / chemin_image).parent.mkdir(parents=True, exist_ok=True)
    (dossier / chemin_matrice).parent.mkdir(parents=True, exist_ok=True)
    np.save(dossier / chemin_matrice, matrice.astype(np.float32))
    succes = cv2.imwrite(
        str(dossier / chemin_image), np.rint(matrice * 255).astype(np.uint8)
    )
    if not succes:
        raise OSError(f"Impossible d'ecrire {dossier / chemin_image}")
    return {
        "sample_id": identifiant,
        "label": etiquette,
        "split": split,
        "origine": origine,
        "source_type": type_source,
        "source_path": chemin_source,
        "source_index": index_source,
        "image_path": chemin_image.as_posix(),
        "matrix_path": chemin_matrice.as_posix(),
        "sha256": empreinte_matrice(matrice),
        "quality_status": qualite.statut,
        "quality_notes": " | ".join(qualite.notes),
        "transformations": json.dumps(
            transformations or {}, ensure_ascii=False, sort_keys=True
        ),
    }


def lire_manifeste_csv(dossier: Path) -> list[dict[str, str]]:
    chemin = dossier / "manifest.csv"
    if not chemin.is_file():
        raise FileNotFoundError(f"Manifeste CSV introuvable: {chemin}")
    with chemin.open(newline="", encoding="utf-8") as fichier:
        lignes = list(csv.DictReader(fichier))
    if not lignes:
        raise ValueError(f"Manifeste vide: {chemin}")
    absents = set(CHAMPS_MANIFESTE) - set(lignes[0])
    if absents:
        raise ValueError("Colonnes absentes du manifeste: " + ", ".join(sorted(absents)))
    return lignes


def sauvegarder_manifestes(
    dossier: Path,
    lignes: list[dict[str, str]],
    metadonnees: dict[str, Any],
    rejets: list[dict[str, str]],
) -> None:
    with (dossier / "manifest.csv").open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=CHAMPS_MANIFESTE)
        writer.writeheader()
        writer.writerows(lignes)
    document = {
        "format": "dataset_caracteres_28x28_v1",
        "cree_le": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_dir": str(dossier),
        "nombre_echantillons": len(lignes),
        "classes": sorted({ligne["label"] for ligne in lignes}),
        "repartition_classes": compter(lignes, "label"),
        "repartition_splits": compter(lignes, "split"),
        "repartition_origines": compter(lignes, "origine"),
        "metadonnees": metadonnees,
        "fichiers": {
            "table_echantillons": "manifest.csv",
            "matrices": "matrices/<classe>/<sample_id>.npy",
            "images": "images/<classe>/<sample_id>.png",
            "apercu": "apercus/grille_par_classe.png" if lignes else None,
            "rapport_initial": (
                "rapports/validation_initiale.json"
                if lignes
                else "rapports/echec_import.json"
            ),
            "rejets_import": "rapports/rejets_import.csv" if rejets else None,
        },
    }
    with (dossier / "manifest.json").open("w", encoding="utf-8") as fichier:
        json.dump(document, fichier, ensure_ascii=False, indent=2)
    if rejets:
        with (dossier / "rapports" / "rejets_import.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fichier:
            writer = csv.DictWriter(
                fichier,
                fieldnames=("source_path", "source_index", "raison"),
            )
            writer.writeheader()
            writer.writerows(rejets)


def compter(lignes: Iterable[dict[str, str]], champ: str) -> dict[str, int]:
    resultat: dict[str, int] = {}
    for ligne in lignes:
        valeur = ligne[champ]
        resultat[valeur] = resultat.get(valeur, 0) + 1
    return dict(sorted(resultat.items()))


def charger_matrice_manifestee(dossier: Path, ligne: dict[str, str]) -> np.ndarray:
    chemin_relatif = Path(ligne["matrix_path"])
    if chemin_relatif.is_absolute() or ".." in chemin_relatif.parts:
        raise ValueError(
            f"Chemin de matrice non sur dans {ligne.get('sample_id', '?')}: {chemin_relatif}"
        )
    chemin = dossier / chemin_relatif
    if not chemin.is_file():
        raise FileNotFoundError(f"Matrice absente: {chemin}")
    return np.load(chemin, allow_pickle=False)


def valider_dataset(dossier: Path) -> dict[str, Any]:
    dossier = dossier.expanduser().resolve()
    lignes = lire_manifeste_csv(dossier)
    erreurs = []
    avertissements = []
    empreintes: dict[str, str] = {}
    for ligne in lignes:
        identifiant = ligne["sample_id"]
        try:
            matrice = charger_matrice_manifestee(dossier, ligne)
            qualite = evaluer_qualite(matrice)
            if qualite.statut == "invalide":
                erreurs.append({"sample_id": identifiant, "notes": qualite.notes})
            elif qualite.notes:
                avertissements.append(
                    {"sample_id": identifiant, "notes": qualite.notes}
                )
            empreinte = empreinte_matrice(np.asarray(matrice, dtype=np.float32))
            if empreinte != ligne["sha256"]:
                erreurs.append(
                    {"sample_id": identifiant, "notes": ["empreinte SHA-256 differente"]}
                )
            if empreinte in empreintes:
                avertissements.append(
                    {
                        "sample_id": identifiant,
                        "notes": [f"doublon exact de {empreintes[empreinte]}"],
                    }
                )
            else:
                empreintes[empreinte] = identifiant
        except (FileNotFoundError, OSError, ValueError) as exc:
            erreurs.append({"sample_id": identifiant, "notes": [str(exc)]})
    return {
        "dataset": str(dossier),
        "verifie_le": datetime.now().astimezone().isoformat(timespec="seconds"),
        "nombre_echantillons": len(lignes),
        "nombre_erreurs": len(erreurs),
        "nombre_avertissements": len(avertissements),
        "valide": not erreurs,
        "erreurs": erreurs,
        "avertissements": avertissements,
        "repartition_classes": compter(lignes, "label"),
        "repartition_splits": compter(lignes, "split"),
    }


def ecrire_rapport_validation(rapport: dict[str, Any], chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as fichier:
        json.dump(rapport, fichier, ensure_ascii=False, indent=2)


def creer_apercu(
    dossier: Path,
    lignes: list[dict[str, str]],
    chemin_sortie: Path,
    par_classe: int,
    classes_max: int = 20,
    dpi: int = 160,
) -> Path:
    if par_classe <= 0 or classes_max <= 0 or dpi <= 0:
        raise ValueError("Les dimensions de l'apercu et le DPI doivent etre positifs.")
    groupes: dict[str, list[dict[str, str]]] = {}
    for ligne in lignes:
        groupes.setdefault(ligne["label"], []).append(ligne)
    etiquettes = sorted(groupes, key=lambda valeur: (len(valeur), valeur))[:classes_max]
    colonnes = min(par_classe, max(len(groupes[e]) for e in etiquettes))
    figure, axes = plt.subplots(
        len(etiquettes),
        colonnes,
        figsize=(1.5 * colonnes + 1.5, 1.45 * len(etiquettes) + 1.0),
        squeeze=False,
        constrained_layout=True,
    )
    for ligne_index, etiquette in enumerate(etiquettes):
        selection = groupes[etiquette][:colonnes]
        for colonne in range(colonnes):
            axe = axes[ligne_index, colonne]
            axe.set_xticks([])
            axe.set_yticks([])
            if colonne < len(selection):
                matrice = charger_matrice_manifestee(dossier, selection[colonne])
                axe.imshow(matrice, cmap="gray", vmin=0, vmax=1)
                axe.set_title(selection[colonne]["sample_id"], fontsize=7)
            else:
                axe.axis("off")
        axes[ligne_index, 0].set_ylabel(
            f"Classe {etiquette}\n(n={len(groupes[etiquette])})",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
        )
    classes_omises = max(0, len(groupes) - len(etiquettes))
    sous_titre = (
        f"{len(lignes)} echantillons, {len(groupes)} classes"
        + (f", {classes_omises} classe(s) omise(s)" if classes_omises else "")
    )
    figure.suptitle(
        "Apercu du dataset synthetique 28 x 28\n" + sous_titre,
        fontsize=13,
    )
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin_sortie, dpi=dpi, facecolor="white")
    plt.close(figure)
    return chemin_sortie


def ecrire_readme_dataset(dossier: Path, metadonnees: dict[str, Any]) -> None:
    texte = f"""# Dataset synthetique 28 x 28

Ce dossier a ete cree par `scripts/preparer_dataset_synthetique.py`.

Description : {metadonnees.get('description') or 'non renseignee'}

## Organisation

- `manifest.json` : resume, provenance et repartitions du dataset.
- `manifest.csv` : une ligne par echantillon; fichier a utiliser pour les analyses.
- `matrices/<classe>/` : matrices NumPy `float32` de forme 28 x 28, valeurs 0..1.
- `images/<classe>/` : apercus PNG correspondants.
- `apercus/grille_par_classe.png` : controle visuel rapide.
- `rapports/validation_initiale.json` : controles de forme, plage et empreinte.
- `rapports/rejets_import.csv` : sources refusees et raisons, si necessaire.

Les entrees brutes ne sont ni modifiees ni deplacees. `origine`, `source_path`,
`source_type` et `transformations` permettent de distinguer les donnees IA,
procedurales, augmentees et originales. Aucun entrainement n'est lance ici.
"""
    (dossier / "README.md").write_text(texte, encoding="utf-8")


def ecrire_readme_echec_import(dossier: Path) -> None:
    texte = """# Import synthétique échoué

Aucun échantillon 28 x 28 valide n'a été importé. Ce dossier est conservé afin
que l'échec reste traçable :

- `manifest.json` : paramètres et provenance de la tentative ;
- `manifest.csv` : en-tête du manifeste, sans échantillon ;
- `rapports/echec_import.json` : état final de la tentative ;
- `rapports/rejets_import.csv` : fichier, index et raison de chaque rejet.

Corrigez ou remplacez les entrées brutes, puis relancez avec un nouveau nom de
dataset. Aucune entrée brute n'a été modifiée.
"""
    (dossier / "README.md").write_text(texte, encoding="utf-8")


def finaliser_dataset(
    dossier: Path,
    lignes: list[dict[str, str]],
    metadonnees: dict[str, Any],
    rejets: list[dict[str, str]],
    apercus_par_classe: int,
) -> None:
    if not lignes:
        raise ValueError("Aucun echantillon valide n'a ete produit.")
    sauvegarder_manifestes(dossier, lignes, metadonnees, rejets)
    creer_apercu(
        dossier,
        lignes,
        dossier / "apercus" / "grille_par_classe.png",
        par_classe=apercus_par_classe,
    )
    rapport = valider_dataset(dossier)
    ecrire_rapport_validation(
        rapport,
        dossier / "rapports" / "validation_initiale.json",
    )
    ecrire_readme_dataset(dossier, metadonnees)
    if not rapport["valide"]:
        raise ValueError(
            "Le dataset a ete cree mais a echoue a la validation interne; "
            f"voir {dossier / 'rapports' / 'validation_initiale.json'}."
        )


def commande_importer(args: argparse.Namespace) -> Path:
    entrees = args.entree or [ENTREE_PAR_DEFAUT]
    fichiers = decouvrir_fichiers(entrees)
    futur_dossier = args.sortie.expanduser().resolve() / slug(args.nom_dataset)
    verifier_separation_entrees_sortie(entrees, futur_dossier)
    dossier = creer_dossier_dataset(args.sortie, args.nom_dataset)
    lignes: list[dict[str, str]] = []
    rejets: list[dict[str, str]] = []

    for chemin, etiquette_dossier in fichiers:
        try:
            elements = list(charger_elements_fichier(chemin, etiquette_dossier))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejets.append(
                {"source_path": str(chemin), "source_index": "*", "raison": str(exc)}
            )
            continue
        if not elements:
            rejets.append(
                {
                    "source_path": str(chemin),
                    "source_index": "*",
                    "raison": "aucun echantillon trouve dans le fichier",
                }
            )
            continue
        for element in elements:
            etiquette = (args.etiquette or element.etiquette or "").strip()
            split = element.split or args.split
            if split not in {"train", "validation", "test", "non_attribue"}:
                rejets.append(
                    {
                        "source_path": str(chemin),
                        "source_index": element.index_source,
                        "raison": f"split non reconnu: {split!r}",
                    }
                )
                continue
            if not etiquette:
                rejets.append(
                    {
                        "source_path": str(chemin),
                        "source_index": element.index_source,
                        "raison": (
                            "etiquette absente; utiliser --etiquette, un sous-dossier "
                            "de classe ou le champ JSON label"
                        ),
                    }
                )
                continue
            try:
                matrice = preparer_element(element, args.polarite, args.binaire)
                qualite = evaluer_qualite(matrice)
                if qualite.statut == "invalide":
                    raise ValueError("; ".join(qualite.notes))
                lignes.append(
                    sauvegarder_echantillon(
                        dossier,
                        len(lignes) + 1,
                        etiquette,
                        split,
                        args.origine,
                        element.type_source,
                        str(element.chemin_source),
                        element.index_source,
                        matrice,
                        qualite,
                        {
                            "polarite": args.polarite,
                            "binarisation": bool(args.binaire),
                            "image_recadree": element.type_source == "image_importee",
                            "metadonnees_source": element.metadonnees_source or {},
                        },
                    )
                )
            except (OSError, ValueError) as exc:
                rejets.append(
                    {
                        "source_path": str(chemin),
                        "source_index": element.index_source,
                        "raison": str(exc),
                    }
                )

    metadonnees = {
        "commande": "importer",
        "description": args.description,
        "origine": args.origine,
        "entrees": [str(Path(e).expanduser().resolve()) for e in entrees],
        "polarite": args.polarite,
        "binaire": bool(args.binaire),
        "nombre_fichiers_sources": len(fichiers),
        "nombre_rejets": len(rejets),
    }
    if not lignes:
        sauvegarder_manifestes(dossier, [], metadonnees, rejets)
        ecrire_readme_echec_import(dossier)
        ecrire_rapport_validation(
            {
                "dataset": str(dossier),
                "valide": False,
                "nombre_echantillons": 0,
                "nombre_rejets": len(rejets),
                "erreur": "Aucun echantillon valide n'a pu etre importe.",
            },
            dossier / "rapports" / "echec_import.json",
        )
        raise ValueError(
            "Aucun echantillon valide n'a pu etre importe. Les causes sont "
            f"repertoriees dans {dossier / 'rapports' / 'rejets_import.csv'}."
        )
    finaliser_dataset(dossier, lignes, metadonnees, rejets, args.apercus_par_classe)
    return dossier


def _dessiner_procedural(
    texte: str,
    generateur: np.random.Generator,
    binaire: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    polices = (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    )
    police = int(generateur.choice(polices))
    echelle = float(generateur.uniform(1.15, 1.9))
    epaisseur = int(generateur.integers(1, 5))
    canevas = np.zeros((72, 72), dtype=np.uint8)
    (largeur, hauteur), base = cv2.getTextSize(texte, police, echelle, epaisseur)
    x = max(1, (72 - largeur) // 2 + int(generateur.integers(-5, 6)))
    y = max(hauteur + 1, (72 + hauteur) // 2 + int(generateur.integers(-5, 6)))
    cv2.putText(
        canevas,
        texte,
        (x, y),
        police,
        echelle,
        255,
        epaisseur,
        cv2.LINE_AA,
    )
    angle = float(generateur.uniform(-16, 16))
    transformation = cv2.getRotationMatrix2D((36, 36), angle, 1.0)
    canevas = cv2.warpAffine(
        canevas,
        transformation,
        (72, 72),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    morphologie = int(generateur.integers(-1, 2))
    if morphologie:
        noyau = np.ones((2, 2), np.uint8)
        operation = cv2.dilate if morphologie > 0 else cv2.erode
        canevas = operation(canevas, noyau, iterations=1)
    matrice = _recadrer_image_vers_28(canevas.astype(np.float32) / 255)
    flou = float(generateur.uniform(0, 0.65))
    if flou > 0.15:
        matrice = cv2.GaussianBlur(matrice, (0, 0), flou)
    niveau_bruit = float(generateur.uniform(0, 0.025))
    if niveau_bruit:
        bruit = generateur.normal(0, niveau_bruit, matrice.shape).astype(np.float32)
        matrice = np.clip(matrice + bruit * (matrice > 0.02), 0, 1)
    if binaire:
        matrice = (matrice >= 0.5).astype(np.float32)
    parametres = {
        "police_opencv": police,
        "echelle": echelle,
        "epaisseur": epaisseur,
        "rotation_deg": angle,
        "morphologie": morphologie,
        "sigma_flou": flou,
        "sigma_bruit": niveau_bruit,
        "binarisation": binaire,
    }
    return np.asarray(matrice, dtype=np.float32), parametres


def commande_baseline(args: argparse.Namespace) -> Path:
    if args.par_classe <= 0:
        raise ValueError("--par-classe doit etre strictement positif.")
    classes = [str(valeur).strip() for valeur in args.classes]
    if not all(classes):
        raise ValueError("Les etiquettes de classes ne peuvent pas etre vides.")
    dossier = creer_dossier_dataset(args.sortie, args.nom_dataset)
    generateur = np.random.default_rng(args.seed)
    lignes = []
    for etiquette in classes:
        for index in range(args.par_classe):
            matrice, parametres = _dessiner_procedural(
                etiquette, generateur, args.binaire
            )
            qualite = evaluer_qualite(matrice)
            if qualite.statut == "invalide":
                raise ValueError(
                    f"Generation invalide pour {etiquette!r}: {qualite.notes}"
                )
            lignes.append(
                sauvegarder_echantillon(
                    dossier,
                    len(lignes) + 1,
                    etiquette,
                    "train",
                    "baseline_procedural_opencv",
                    "procedural_baseline",
                    "",
                    str(index),
                    matrice,
                    qualite,
                    parametres,
                )
            )
    metadonnees = {
        "commande": "generer-baseline",
        "description": args.description,
        "classes": classes,
        "par_classe": args.par_classe,
        "seed": args.seed,
        "binaire": bool(args.binaire),
        "avertissement": (
            "Baseline typographique procedural: ce n'est ni de l'ecriture "
            "humaine ni une generation par IA."
        ),
    }
    finaliser_dataset(dossier, lignes, metadonnees, [], args.apercus_par_classe)
    return dossier


def _augmenter_matrice(
    matrice: np.ndarray,
    generateur: np.random.Generator,
    rotation_max: float,
    translation_max: float,
    bruit_max: float,
    binaire: bool,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    angle = float(generateur.uniform(-rotation_max, rotation_max))
    dx = float(generateur.uniform(-translation_max, translation_max))
    dy = float(generateur.uniform(-translation_max, translation_max))
    transformation = cv2.getRotationMatrix2D((13.5, 13.5), angle, 1.0)
    transformation[:, 2] += (dx, dy)
    resultat = cv2.warpAffine(
        matrice,
        transformation,
        (28, 28),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    morphologie = int(generateur.integers(-1, 2))
    if morphologie:
        noyau = np.ones((2, 2), dtype=np.uint8)
        resultat_uint8 = np.rint(resultat * 255).astype(np.uint8)
        operation = cv2.dilate if morphologie > 0 else cv2.erode
        resultat = operation(resultat_uint8, noyau, iterations=1).astype(np.float32) / 255
    sigma_bruit = float(generateur.uniform(0, bruit_max))
    if sigma_bruit:
        bruit = generateur.normal(0, sigma_bruit, resultat.shape).astype(np.float32)
        resultat = np.clip(resultat + bruit * (resultat > 0.02), 0, 1)
    if binaire:
        resultat = (resultat >= 0.5).astype(np.float32)
    return np.asarray(resultat, dtype=np.float32), {
        "rotation_deg": angle,
        "translation_x": dx,
        "translation_y": dy,
        "morphologie": morphologie,
        "sigma_bruit": sigma_bruit,
        "binarisation": binaire,
    }


def _verifier_borne_positive(nom: str, valeur: float) -> None:
    if valeur < 0 or not math.isfinite(valeur):
        raise ValueError(f"{nom} doit etre un nombre fini positif ou nul.")


def commande_augmenter(args: argparse.Namespace) -> Path:
    if args.copies_par_image <= 0:
        raise ValueError("--copies-par-image doit etre strictement positif.")
    _verifier_borne_positive("--rotation-max", args.rotation_max)
    _verifier_borne_positive("--translation-max", args.translation_max)
    _verifier_borne_positive("--bruit-max", args.bruit_max)
    dataset_source = args.dataset_source.expanduser().resolve()
    lignes_sources = lire_manifeste_csv(dataset_source)
    lignes_sources = [
        ligne for ligne in lignes_sources if ligne["split"] in set(args.splits)
    ]
    if not lignes_sources:
        raise ValueError(
            "Aucun echantillon ne correspond aux partitions demandees avec --splits."
        )
    futur_dossier = args.sortie.expanduser().resolve() / slug(args.nom_dataset)
    if _chemin_est_dans(futur_dossier, dataset_source):
        raise ValueError("Le dataset derive ne doit pas etre cree dans le dataset source.")
    dossier = creer_dossier_dataset(args.sortie, args.nom_dataset)
    generateur = np.random.default_rng(args.seed)
    lignes = []
    for ligne_source in lignes_sources:
        source = np.asarray(
            charger_matrice_manifestee(dataset_source, ligne_source), dtype=np.float32
        )
        if args.inclure_originaux:
            qualite = evaluer_qualite(source)
            lignes.append(
                sauvegarder_echantillon(
                    dossier,
                    len(lignes) + 1,
                    ligne_source["label"],
                    ligne_source["split"],
                    ligne_source["origine"],
                    "original_copie",
                    str(dataset_source / ligne_source["matrix_path"]),
                    ligne_source["sample_id"],
                    source,
                    qualite,
                    {"copie_sans_modification": True},
                )
            )
        for copie in range(args.copies_par_image):
            matrice, parametres = _augmenter_matrice(
                source,
                generateur,
                args.rotation_max,
                args.translation_max,
                args.bruit_max,
                args.binaire,
            )
            qualite = evaluer_qualite(matrice)
            if qualite.statut == "invalide":
                raise ValueError(
                    f"Augmentation invalide de {ligne_source['sample_id']}: {qualite.notes}"
                )
            parametres["copie_numero"] = copie + 1
            lignes.append(
                sauvegarder_echantillon(
                    dossier,
                    len(lignes) + 1,
                    ligne_source["label"],
                    ligne_source["split"],
                    f"augmentation_de:{ligne_source['origine']}",
                    "augmentation_classique",
                    str(dataset_source / ligne_source["matrix_path"]),
                    ligne_source["sample_id"],
                    matrice,
                    qualite,
                    parametres,
                )
            )
    metadonnees = {
        "commande": "augmenter",
        "description": args.description,
        "dataset_source": str(dataset_source),
        "copies_par_image": args.copies_par_image,
        "splits_sources": args.splits,
        "inclure_originaux": bool(args.inclure_originaux),
        "rotation_max": args.rotation_max,
        "translation_max": args.translation_max,
        "bruit_max": args.bruit_max,
        "seed": args.seed,
        "binaire": bool(args.binaire),
    }
    finaliser_dataset(dossier, lignes, metadonnees, [], args.apercus_par_classe)
    return dossier


def commande_valider(args: argparse.Namespace) -> Path:
    dossier = args.dataset.expanduser().resolve()
    rapport = valider_dataset(dossier)
    chemin = args.rapport
    if chemin is None:
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin = dossier / "rapports" / f"validation_{horodatage}.json"
    else:
        chemin = chemin.expanduser().resolve()
    ecrire_rapport_validation(rapport, chemin)
    print(
        f"Validation: {rapport['nombre_erreurs']} erreur(s), "
        f"{rapport['nombre_avertissements']} avertissement(s)."
    )
    if not rapport["valide"]:
        raise ValueError(f"Dataset invalide; consulter {chemin}")
    return chemin


def commande_visualiser(args: argparse.Namespace) -> Path:
    dossier = args.dataset.expanduser().resolve()
    lignes = lire_manifeste_csv(dossier)
    if args.sortie is None:
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin = dossier / "apercus" / f"grille_{horodatage}.png"
    else:
        chemin = args.sortie.expanduser().resolve()
    return creer_apercu(
        dossier,
        lignes,
        chemin,
        args.par_classe,
        args.classes_max,
        args.dpi,
    )


def main() -> int:
    parser = construire_parser()
    args = parser.parse_args()
    dossier_partiel: Path | None = None
    try:
        if hasattr(args, "apercus_par_classe") and args.apercus_par_classe <= 0:
            raise ValueError("--apercus-par-classe doit etre strictement positif.")
        if args.commande == "importer":
            resultat = commande_importer(args)
            dossier_partiel = resultat
            print(f"Dataset importe: {resultat}")
            print(f"Index des echantillons: {resultat / 'manifest.csv'}")
        elif args.commande == "generer-baseline":
            resultat = commande_baseline(args)
            dossier_partiel = resultat
            print(f"Baseline procedural cree: {resultat}")
            print("Aucun modele d'IA n'a ete appele.")
        elif args.commande == "augmenter":
            resultat = commande_augmenter(args)
            dossier_partiel = resultat
            print(f"Dataset augmente cree: {resultat}")
        elif args.commande == "valider":
            resultat = commande_valider(args)
            print(f"Rapport de validation: {resultat}")
        else:
            resultat = commande_visualiser(args)
            print(f"Apercu sauvegarde: {resultat}")
        return 0
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        if dossier_partiel is not None:
            print(f"Dossier potentiellement partiel: {dossier_partiel}")
        parser.exit(2, f"Erreur: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
