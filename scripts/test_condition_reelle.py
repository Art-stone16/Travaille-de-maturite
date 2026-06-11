import argparse
import unicodedata
from pathlib import Path

import env_config

import cv2
import keras
import numpy as np


MODEL_PATH = env_config.PROJECT_ROOT / "modeles" / "modeles_valides" / "Best_COLOR_MAP" / "best_model.keras"
IMAGE_PATH = env_config.PROJECT_ROOT / "donnees" / "image_reelle.jpg"
OUTPUT_PATH = env_config.PROJECT_ROOT / "sorties" / "tests_condition_reelle" / "hypothese_condition_reel.png"


def normalise_path(path: Path) -> Path:
    """Retrouve aussi les fichiers avec accents encodes differemment sur macOS."""
    if path.exists():
        return path

    parent = path.parent if str(path.parent) != "." else Path(".")
    wanted = unicodedata.normalize("NFC", path.name)
    for candidate in parent.iterdir():
        if unicodedata.normalize("NFC", candidate.name) == wanted:
            return candidate

    raise FileNotFoundError(f"Fichier introuvable: {path}")


def prepare_digit(threshold_image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(
        threshold_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [c for c in contours if cv2.contourArea(c) > 200]

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        digit = threshold_image[y : y + h, x : x + w]
    else:
        x, y, w, h = 0, 0, threshold_image.shape[1], threshold_image.shape[0]
        digit = threshold_image

    side = max(w, h)
    margin = max(4, int(side * 0.20))
    square = np.zeros((side + 2 * margin, side + 2 * margin), dtype=np.uint8)
    x_offset = margin + (side - w) // 2
    y_offset = margin + (side - h) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = digit

    digit_28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    digit_28 = digit_28.astype("float32") / 255.0
    digit_28 = np.expand_dims(digit_28, axis=(0, -1))

    return digit_28, (x, y, w, h)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teste un modele MNIST sur une image prise en condition reelle."
    )
    parser.add_argument("--image", type=Path, default=IMAGE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    image_path = normalise_path(args.image)
    model_path = normalise_path(args.model)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV n'arrive pas a lire l'image: {image_path}")

    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (7, 7), 0)
    seuil = cv2.adaptiveThreshold(
        flou,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )

    digit_28, box = prepare_digit(seuil)
    model = keras.models.load_model(model_path)
    probabilities = model.predict(digit_28, verbose=0)[0]
    hypothesis = int(np.argmax(probabilities))
    confidence = float(probabilities[hypothesis])

    print(f"Image: {image_path}")
    print(f"Modele: {model_path}")
    print(f"Hypothese principale: {hypothesis} ({confidence:.1%})")
    print("Probabilites par chiffre:")
    for digit, probability in enumerate(probabilities):
        print(f"  {digit}: {probability:.1%}")

    x, y, w, h = box
    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)
    cv2.putText(
        annotated,
        f"Hypothese: {hypothesis} ({confidence:.1%})",
        (max(0, x), max(45, y - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)
    print(f"Image de controle sauvegardee: {args.output}")


if __name__ == "__main__":
    main()
