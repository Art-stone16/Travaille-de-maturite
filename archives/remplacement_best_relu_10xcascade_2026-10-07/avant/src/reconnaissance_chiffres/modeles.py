"""Chargement des modèles avec l'adaptation des anciennes archives Keras."""

import json
import zipfile


def charger_modele(model_path, *, compile=True):
    """Charge le modèle et conserve l'adaptation des paramètres Keras."""
    import keras

    if not model_path.exists():
        raise ValueError(f"Modele introuvable: {model_path}")

    try:
        return keras.models.load_model(model_path, compile=compile)
    except TypeError as exc:
        if "input_axes" not in str(exc):
            raise

    with zipfile.ZipFile(model_path) as archive:
        configuration = json.loads(archive.read("config.json"))

    def retirer_axes_incompatibles(valeur):
        if isinstance(valeur, dict):
            valeur.pop("input_axes", None)
            valeur.pop("output_axes", None)
            for enfant in valeur.values():
                retirer_axes_incompatibles(enfant)
        elif isinstance(valeur, list):
            for enfant in valeur:
                retirer_axes_incompatibles(enfant)

    retirer_axes_incompatibles(configuration)
    modele = keras.models.model_from_json(json.dumps(configuration))
    modele.load_weights(model_path)
    return modele
