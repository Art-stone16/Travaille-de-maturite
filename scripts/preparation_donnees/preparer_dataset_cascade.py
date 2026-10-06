"""Extraire et contrôler un dataset 28 x 28 depuis les feuilles Cascade Top-N.

Le script ne reprend jamais les prédictions d'un modèle comme étiquettes. La
classe vient du nom de la photo (par exemple CTN_7.jpg -> classe 7).

Workflow :

1. ``extraire`` détecte les zones, les transforme comme les entrées du modèle
   et crée un manifeste dont chaque décision vaut ``a_verifier``.
2. Après contrôle des planches, remplacer les décisions par ``inclure`` ou
   ``exclure`` dans ``manifest.csv``.
3. ``finaliser`` crée les tableaux NumPy uniquement lorsque toutes les lignes
   ont reçu une décision explicite.
"""

from __future__ import annotations

# Permet aussi le lancement direct depuis n'importe quel répertoire.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts import _bootstrap  # noqa: F401


import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from reconnaissance_chiffres import config as env_config

import cv2
import numpy as np

from reconnaissance_chiffres import detection as detection


ENTREE_PAR_DEFAUT = env_config.DONNEES_CASCADE_TOP_N
SORTIE_PAR_DEFAUT = (
    env_config.PROJECT_ROOT
    / "donnees"
    / "preparees" / "cascade"
    / "cascade_top_n_v1"
)
FORMATS_IMAGE = {".jpg", ".jpeg", ".png"}
DECISIONS = {"a_verifier", "inclure", "exclure"}
SPLITS = {"train", "validation", "test"}
CHAMPS_MANIFESTE = (
    "sample_id",
    "label",
    "source_image",
    "source_variant",
    "source_index",
    "x",
    "y",
    "largeur",
    "hauteur",
    "decision",
    "notes_validation",
    "split",
    "image_28_path",
    "matrix_path",
    "recadrage_path",
)


def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prépare un dataset MNIST-compatible depuis les feuilles "
            "Cascade Top-N."
        )
    )
    sous_commandes = parser.add_subparsers(dest="commande", required=True)

    extraire = sous_commandes.add_parser(
        "extraire",
        help="Détecter et exporter les candidats sans les valider.",
    )
    extraire.add_argument("--entree", type=Path, default=ENTREE_PAR_DEFAUT)
    extraire.add_argument("--sortie", type=Path, default=SORTIE_PAR_DEFAUT)
    extraire.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="train",
        help="Partition proposée dans le manifeste.",
    )

    finaliser = sous_commandes.add_parser(
        "finaliser",
        help="Créer les tableaux NumPy après validation du manifeste.",
    )
    finaliser.add_argument("--dataset", type=Path, default=SORTIE_PAR_DEFAUT)

    decider = sous_commandes.add_parser(
        "decider",
        help="Inscrire des décisions de contrôle dans le manifeste.",
    )
    decider.add_argument("--dataset", type=Path, default=SORTIE_PAR_DEFAUT)
    decider.add_argument(
        "--accepter-reste",
        action="store_true",
        help="Marquer 'inclure' toutes les lignes qui ne sont pas exclues.",
    )
    decider.add_argument(
        "--exclure",
        nargs="*",
        default=[],
        metavar="SAMPLE_ID",
        help="Identifiants à marquer 'exclure'.",
    )
    decider.add_argument(
        "--note-exclusion",
        default="Exclusion après contrôle visuel.",
        help="Note inscrite pour chaque exclusion.",
    )

    resume = sous_commandes.add_parser(
        "resume",
        help="Afficher l'état des décisions du manifeste.",
    )
    resume.add_argument("--dataset", type=Path, default=SORTIE_PAR_DEFAUT)
    return parser


def extraire_classe_et_variante(chemin: Path) -> tuple[int, str]:
    correspondance = re.fullmatch(
        r"CTN_(\d)(?:\.(\d+))?",
        chemin.stem,
        flags=re.IGNORECASE,
    )
    if correspondance is None:
        raise ValueError(
            f"Nom de feuille non reconnu: {chemin.name}. "
            "Format attendu: CTN_0.jpg à CTN_9.jpg."
        )
    etiquette = int(correspondance.group(1))
    variante = correspondance.group(2) or "principale"
    return etiquette, variante


def lister_images(dossier: Path) -> list[Path]:
    if not dossier.is_dir():
        raise FileNotFoundError(f"Dossier d'entrée introuvable: {dossier}")
    images = sorted(
        (
            chemin
            for chemin in dossier.iterdir()
            if chemin.is_file() and chemin.suffix.lower() in FORMATS_IMAGE
        ),
        key=lambda chemin: extraire_classe_et_variante(chemin),
    )
    if not images:
        raise ValueError(f"Aucune feuille Cascade Top-N dans {dossier}")
    etiquettes = {extraire_classe_et_variante(chemin)[0] for chemin in images}
    manquantes = sorted(set(range(10)) - etiquettes)
    if manquantes:
        raise ValueError(
            "Classes absentes des feuilles Cascade Top-N: "
            + ", ".join(map(str, manquantes))
        )
    return images


def verifier_sortie_vide(dossier: Path) -> None:
    if dossier.exists() and any(dossier.iterdir()):
        raise FileExistsError(
            f"Le dossier de sortie n'est pas vide: {dossier}\n"
            "Choisir un autre --sortie pour préserver le dataset existant."
        )


def ecrire_image(chemin: Path, image: np.ndarray) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(chemin), image):
        raise OSError(f"Impossible d'écrire l'image: {chemin}")


def creer_planche(
    dossier: Path,
    source: str,
    lignes: list[dict[str, str]],
    colonnes: int = 10,
) -> Path:
    largeur_case = 112
    hauteur_case = 142
    marge_titre = 64
    lignes_grille = math.ceil(len(lignes) / colonnes)
    planche = np.full(
        (
            marge_titre + lignes_grille * hauteur_case,
            colonnes * largeur_case,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        planche,
        f"{source} - {len(lignes)} candidats - tous a verifier",
        (14, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )

    for position, ligne in enumerate(lignes):
        rangee, colonne = divmod(position, colonnes)
        x0 = colonne * largeur_case
        y0 = marge_titre + rangee * hauteur_case
        image_28 = cv2.imread(
            str(dossier / ligne["image_28_path"]),
            cv2.IMREAD_GRAYSCALE,
        )
        if image_28 is None:
            raise FileNotFoundError(ligne["image_28_path"])
        apercu = cv2.resize(image_28, (96, 96), interpolation=cv2.INTER_NEAREST)
        apercu = cv2.cvtColor(apercu, cv2.COLOR_GRAY2BGR)
        planche[y0 + 6 : y0 + 102, x0 + 8 : x0 + 104] = apercu
        cv2.rectangle(
            planche,
            (x0 + 7, y0 + 5),
            (x0 + 104, y0 + 103),
            (80, 80, 80),
            1,
        )
        cv2.putText(
            planche,
            ligne["sample_id"],
            (x0 + 8, y0 + 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    chemin = Path("planches_controle") / f"{source}.png"
    ecrire_image(dossier / chemin, planche)
    return chemin


def ecrire_manifeste(dossier: Path, lignes: list[dict[str, str]]) -> None:
    with (dossier / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fichier:
        writer = csv.DictWriter(fichier, fieldnames=CHAMPS_MANIFESTE)
        writer.writeheader()
        writer.writerows(lignes)


def lire_manifeste(dossier: Path) -> list[dict[str, str]]:
    chemin = dossier / "manifest.csv"
    if not chemin.is_file():
        raise FileNotFoundError(f"Manifeste introuvable: {chemin}")
    with chemin.open(newline="", encoding="utf-8") as fichier:
        lignes = list(csv.DictReader(fichier))
    if not lignes:
        raise ValueError(f"Manifeste vide: {chemin}")
    absents = set(CHAMPS_MANIFESTE) - set(lignes[0])
    if absents:
        raise ValueError(
            "Colonnes manquantes dans le manifeste: "
            + ", ".join(sorted(absents))
        )
    return lignes


def ecrire_guide(dossier: Path) -> None:
    guide = """# Validation du dataset Cascade Top-N

1. Ouvrir les images de `planches_controle/`.
2. Repérer les faux positifs, les chiffres coupés ou les cases qui ne
   correspondent pas à la classe annoncée.
3. Dans `manifest.csv`, remplacer `a_verifier` par :
   - `inclure` pour un chiffre correct ;
   - `exclure` pour une zone incorrecte.
4. Ajouter une explication dans `notes_validation` pour chaque exclusion.
5. Lancer :

```bash
.venv/bin/python scripts/preparation_donnees/preparer_dataset_cascade.py finaliser
```

La finalisation est refusée tant qu'une décision vaut `a_verifier`. Les
prédictions du réseau ne sont jamais utilisées comme étiquettes.

## Fichiers finaux

Après finalisation, `dataset_numpy/` contient les tableaux `x`, `y` et les
identifiants pour chaque split. `train_modele_principal.py` charge les tableaux
du split `train`, les ajoute à MNIST puis mélange les deux sources avant le
prélèvement de la validation. Le jeu de test MNIST reste inchangé.
"""
    (dossier / "LISEZ_MOI.md").write_text(guide, encoding="utf-8")


def extraire_dataset(entree: Path, sortie: Path, split: str) -> None:
    entree = entree.expanduser().resolve()
    sortie = sortie.expanduser().resolve()
    images = lister_images(entree)
    verifier_sortie_vide(sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    manifeste: list[dict[str, str]] = []
    resume_sources = []
    compteur_global = 1

    for chemin_source in images:
        etiquette, variante = extraire_classe_et_variante(chemin_source)
        image = detection.charger_image(chemin_source)
        resultat = detection.detecter_chiffres(image)
        source_id = chemin_source.stem
        lignes_source = []

        detection.sauvegarder_diagnostics(
            resultat,
            image,
            sortie / "diagnostics" / source_id,
        )

        for index, rectangle in enumerate(resultat.rectangles, start=1):
            preparation = detection.preparer_chiffre_avec_details(
                image,
                rectangle,
            )
            sample_id = f"ctn_{compteur_global:05d}"
            x, y, largeur, hauteur = rectangle
            chemin_image = (
                Path("images_28x28") / str(etiquette) / f"{sample_id}.png"
            )
            chemin_matrice = (
                Path("matrices") / str(etiquette) / f"{sample_id}.npy"
            )
            chemin_recadrage = (
                Path("recadrages") / source_id / f"{sample_id}.png"
            )
            ecrire_image(
                sortie / chemin_image,
                preparation.image_28_niveaux_gris,
            )
            ecrire_image(
                sortie / chemin_recadrage,
                preparation.recadrage_original,
            )
            (sortie / chemin_matrice).parent.mkdir(parents=True, exist_ok=True)
            np.save(
                sortie / chemin_matrice,
                preparation.tenseur_modele[0].astype(np.float32),
            )
            ligne = {
                "sample_id": sample_id,
                "label": str(etiquette),
                "source_image": chemin_source.name,
                "source_variant": variante,
                "source_index": str(index),
                "x": str(x),
                "y": str(y),
                "largeur": str(largeur),
                "hauteur": str(hauteur),
                "decision": "a_verifier",
                "notes_validation": "",
                "split": split,
                "image_28_path": chemin_image.as_posix(),
                "matrix_path": chemin_matrice.as_posix(),
                "recadrage_path": chemin_recadrage.as_posix(),
            }
            manifeste.append(ligne)
            lignes_source.append(ligne)
            compteur_global += 1

        planche = creer_planche(sortie, source_id, lignes_source)
        resume_sources.append(
            {
                "source_image": chemin_source.name,
                "label": etiquette,
                "variante": variante,
                "nombre_candidats": len(lignes_source),
                "planche_controle": planche.as_posix(),
            }
        )
        print(
            f"{chemin_source.name}: {len(lignes_source)} candidats "
            f"étiquetés {etiquette} (à vérifier)"
        )

    ecrire_manifeste(sortie, manifeste)
    ecrire_guide(sortie)
    metadata = {
        "format": "dataset_cascade_28x28_v1",
        "cree_le": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entree": str(entree),
        "sortie": str(sortie),
        "nombre_candidats": len(manifeste),
        "decision_initiale": "a_verifier",
        "etiquetage": "classe déduite du nom CTN_<chiffre>",
        "prediction_modele_utilisee": False,
        "sources": resume_sources,
    }
    (sortie / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(manifeste)} candidats exportés dans {sortie}")
    print(f"Contrôle requis: {sortie / 'LISEZ_MOI.md'}")


def compter_etat(lignes: list[dict[str, str]]) -> dict[str, object]:
    decisions = Counter(ligne["decision"].strip().lower() for ligne in lignes)
    classes = Counter(ligne["label"] for ligne in lignes)
    sources = Counter(ligne["source_image"] for ligne in lignes)
    return {
        "nombre_lignes": len(lignes),
        "decisions": dict(sorted(decisions.items())),
        "classes": dict(sorted(classes.items(), key=lambda item: int(item[0]))),
        "sources": dict(sorted(sources.items())),
    }


def afficher_resume(dossier: Path) -> None:
    lignes = lire_manifeste(dossier.expanduser().resolve())
    print(json.dumps(compter_etat(lignes), ensure_ascii=False, indent=2))


def appliquer_decisions(
    dossier: Path,
    accepter_reste: bool,
    exclusions: list[str],
    note_exclusion: str,
) -> None:
    dossier = dossier.expanduser().resolve()
    lignes = lire_manifeste(dossier)
    ids_connus = {ligne["sample_id"] for ligne in lignes}
    exclusions_uniques = set(exclusions)
    ids_inconnus = sorted(exclusions_uniques - ids_connus)
    if ids_inconnus:
        raise ValueError(
            "Identifiants inconnus: " + ", ".join(ids_inconnus)
        )
    if exclusions_uniques and not note_exclusion.strip():
        raise ValueError("Une note est obligatoire pour les exclusions.")
    if not accepter_reste and not exclusions_uniques:
        raise ValueError("Aucune décision demandée.")

    for ligne in lignes:
        sample_id = ligne["sample_id"]
        if sample_id in exclusions_uniques:
            ligne["decision"] = "exclure"
            ligne["notes_validation"] = note_exclusion.strip()
        elif accepter_reste:
            ligne["decision"] = "inclure"
            ligne["notes_validation"] = "Contrôle visuel de la planche effectué."

    ecrire_manifeste(dossier, lignes)
    print(json.dumps(compter_etat(lignes), ensure_ascii=False, indent=2))


def finaliser_dataset(dossier: Path) -> None:
    dossier = dossier.expanduser().resolve()
    lignes = lire_manifeste(dossier)
    erreurs = []

    for numero, ligne in enumerate(lignes, start=2):
        decision = ligne["decision"].strip().lower()
        split = ligne["split"].strip().lower()
        if decision not in DECISIONS:
            erreurs.append(f"ligne {numero}: décision inconnue {decision!r}")
        if decision == "a_verifier":
            erreurs.append(f"ligne {numero}: décision encore à vérifier")
        if split not in SPLITS:
            erreurs.append(f"ligne {numero}: split inconnu {split!r}")
        if decision == "exclure" and not ligne["notes_validation"].strip():
            erreurs.append(
                f"ligne {numero}: une exclusion doit être expliquée dans "
                "notes_validation"
            )

    if erreurs:
        apercu = "\n".join(f"- {erreur}" for erreur in erreurs[:20])
        suffixe = "" if len(erreurs) <= 20 else f"\n- … {len(erreurs) - 20} autres"
        raise ValueError(
            "Le manifeste n'est pas prêt pour la finalisation:\n"
            + apercu
            + suffixe
        )

    incluses = [
        ligne for ligne in lignes if ligne["decision"].strip().lower() == "inclure"
    ]
    if not incluses:
        raise ValueError("Aucun échantillon marqué 'inclure'.")

    dossier_numpy = dossier / "dataset_numpy"
    dossier_numpy.mkdir(parents=True, exist_ok=True)
    resume_splits = {}

    for split in sorted(SPLITS):
        lignes_split = [
            ligne for ligne in incluses if ligne["split"].strip().lower() == split
        ]
        if not lignes_split:
            continue
        matrices = []
        etiquettes = []
        identifiants = []
        for ligne in lignes_split:
            chemin = dossier / ligne["matrix_path"]
            matrice = np.load(chemin, allow_pickle=False)
            if matrice.shape != (28, 28, 1):
                raise ValueError(
                    f"Forme invalide pour {ligne['sample_id']}: {matrice.shape}"
                )
            if not np.isfinite(matrice).all() or matrice.min() < 0 or matrice.max() > 1:
                raise ValueError(
                    f"Valeurs invalides pour {ligne['sample_id']}"
                )
            matrices.append(matrice.astype(np.float32))
            etiquettes.append(int(ligne["label"]))
            identifiants.append(ligne["sample_id"])
        x = np.stack(matrices, axis=0)
        y = np.asarray(etiquettes, dtype=np.uint8)
        ids = np.asarray(identifiants)
        np.save(dossier_numpy / f"x_{split}_cascade.npy", x)
        np.save(dossier_numpy / f"y_{split}_cascade.npy", y)
        np.save(dossier_numpy / f"ids_{split}_cascade.npy", ids)
        resume_splits[split] = {
            "nombre": len(lignes_split),
            "forme_x": list(x.shape),
            "forme_y": list(y.shape),
            "repartition_classes": dict(
                sorted(Counter(map(int, y)).items())
            ),
        }

    resume_final = {
        "format": "dataset_cascade_numpy_v1",
        "finalise_le": datetime.now().astimezone().isoformat(timespec="seconds"),
        "nombre_inclus": len(incluses),
        "nombre_exclus": len(lignes) - len(incluses),
        "splits": resume_splits,
    }
    (dossier_numpy / "resume.json").write_text(
        json.dumps(resume_final, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(resume_final, ensure_ascii=False, indent=2))
    print(f"\nDataset final: {dossier_numpy}")


def main() -> None:
    args = construire_parser().parse_args()
    if args.commande == "extraire":
        extraire_dataset(args.entree, args.sortie, args.split)
    elif args.commande == "decider":
        appliquer_decisions(
            args.dataset,
            args.accepter_reste,
            args.exclure,
            args.note_exclusion,
        )
    elif args.commande == "finaliser":
        finaliser_dataset(args.dataset)
    elif args.commande == "resume":
        afficher_resume(args.dataset)


if __name__ == "__main__":
    main()
