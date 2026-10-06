"""Tests des calculs statistiques de l'évaluation par classe."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import _bootstrap  # noqa: E402, F401
from scripts.evaluation import matrice_confusion as evaluation  # noqa: E402


def observations_depuis_matrice(matrice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vrais: list[int] = []
    predits: list[int] = []
    for classe_reelle in range(10):
        for classe_predite in range(10):
            effectif = int(matrice[classe_reelle, classe_predite])
            vrais.extend([classe_reelle] * effectif)
            predits.extend([classe_predite] * effectif)
    return np.asarray(vrais), np.asarray(predits)


class MetriquesParClasseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrice = np.eye(10, dtype=np.int64) * 8
        self.matrice[0, 0] = 8
        self.matrice[0, 1] = 2
        self.matrice[1, 0] = 1
        self.matrice[1, 1] = 9

    def test_decomposition_et_metriques_connues(self) -> None:
        metriques = evaluation.calculer_metriques_par_classe(self.matrice)
        classe_zero = metriques.loc[metriques["classe"] == 0].iloc[0]
        total = int(self.matrice.sum())

        self.assertEqual(classe_zero["tp"], 8)
        self.assertEqual(classe_zero["fn"], 2)
        self.assertEqual(classe_zero["fp"], 1)
        self.assertEqual(
            classe_zero["tp"]
            + classe_zero["fn"]
            + classe_zero["fp"]
            + classe_zero["tn"],
            total,
        )
        self.assertAlmostEqual(classe_zero["sensibilite"], 0.8)
        self.assertAlmostEqual(classe_zero["precision"], 8 / 9)
        self.assertLessEqual(classe_zero["sensibilite_ic95_bas"], 0.8)
        self.assertGreaterEqual(classe_zero["sensibilite_ic95_haut"], 0.8)

    def test_accord_avec_sklearn(self) -> None:
        vrais, predits = observations_depuis_matrice(self.matrice)
        metriques = evaluation.calculer_metriques_par_classe(self.matrice)
        matrices_binaires = multilabel_confusion_matrix(
            vrais,
            predits,
            labels=np.arange(10),
        )
        rapport = classification_report(
            vrais,
            predits,
            labels=np.arange(10),
            output_dict=True,
            zero_division=0,
        )

        for classe in range(10):
            ligne = metriques.iloc[classe]
            tn, fp, fn, tp = matrices_binaires[classe].ravel()
            self.assertEqual((ligne.tn, ligne.fp, ligne.fn, ligne.tp), (tn, fp, fn, tp))
            self.assertAlmostEqual(ligne.precision, rapport[str(classe)]["precision"])
            self.assertAlmostEqual(ligne.sensibilite, rapport[str(classe)]["recall"])
            self.assertAlmostEqual(ligne.f1, rapport[str(classe)]["f1-score"])

    def test_classe_absente_devient_nan(self) -> None:
        matrice = self.matrice.copy()
        matrice[9, :] = 0
        with warnings.catch_warnings(record=True) as avertissements:
            warnings.simplefilter("always")
            metriques = evaluation.calculer_metriques_par_classe(matrice)
        self.assertTrue(math.isnan(metriques.iloc[9]["sensibilite"]))
        self.assertTrue(avertissements)

    def test_classe_presente_jamais_predite_a_un_f1_nul(self) -> None:
        matrice = np.eye(10, dtype=np.int64) * 5
        matrice[0, 0] = 0
        matrice[0, 1] = 5
        metriques = evaluation.calculer_metriques_par_classe(matrice)
        classe_zero = metriques.iloc[0]
        self.assertEqual(classe_zero["sensibilite"], 0)
        self.assertEqual(classe_zero["precision"], 0)
        self.assertEqual(classe_zero["f1"], 0)

    def test_normalisation_par_ligne(self) -> None:
        normalisee = evaluation.normaliser_matrice(self.matrice, axe=1)
        np.testing.assert_allclose(np.nansum(normalisee, axis=1), np.ones(10))
        self.assertAlmostEqual(normalisee[0, 0], 0.8)

    def test_resume_global_coherent(self) -> None:
        metriques = evaluation.calculer_metriques_par_classe(self.matrice)
        evaluation.verifier_coherence(self.matrice, metriques)
        resume = evaluation.calculer_resume(self.matrice, metriques).iloc[0]
        attendu = float(np.trace(self.matrice) / self.matrice.sum())
        self.assertAlmostEqual(resume["accuracy_globale"], attendu)
        self.assertAlmostEqual(resume["rappel_pondere"], attendu)

    def test_wilson_aux_extremes_et_sans_denominateur(self) -> None:
        vide_bas, vide_haut = evaluation.intervalle_wilson(0, 0)
        self.assertTrue(math.isnan(vide_bas))
        self.assertTrue(math.isnan(vide_haut))
        bas_zero, haut_zero = evaluation.intervalle_wilson(0, 10)
        bas_un, haut_un = evaluation.intervalle_wilson(10, 10)
        self.assertEqual(bas_zero, 0)
        self.assertGreater(haut_zero, 0)
        self.assertLess(bas_un, 1)
        self.assertAlmostEqual(haut_un, 1)


class ChargementMnistTests(unittest.TestCase):
    def test_refuse_les_etiquettes_non_entieres(self) -> None:
        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "mnist_invalide.npz"
            np.savez(
                chemin,
                x_test=np.zeros((2, 28, 28), dtype=np.uint8),
                y_test=np.asarray([0.0, 1.9]),
            )
            with self.assertRaisesRegex(ValueError, "entiers exacts"):
                evaluation.charger_mnist_test(chemin, None)

    def test_conserve_les_indices_du_sous_echantillon(self) -> None:
        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "mnist_valide.npz"
            images = np.zeros((40, 28, 28), dtype=np.uint8)
            etiquettes = np.arange(40, dtype=np.int64) % 10
            np.savez(chemin, x_test=images, y_test=etiquettes)
            _, selection, indices, _, empreinte, preparation = evaluation.charger_mnist_test(
                chemin,
                20,
            )
            self.assertEqual(len(indices), 20)
            self.assertEqual(len(np.unique(indices)), 20)
            np.testing.assert_array_equal(selection, etiquettes[indices])
            self.assertEqual(len(empreinte), 64)
            self.assertTrue(preparation["sous_echantillonnage_applique"])
            self.assertFalse(preparation["division_255_appliquee"])

    def test_limite_superieure_ne_declenche_pas_de_tirage(self) -> None:
        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "mnist_complet.npz"
            np.savez(
                chemin,
                x_test=np.zeros((10, 28, 28), dtype=np.uint8),
                y_test=np.arange(10, dtype=np.int64),
            )
            _, _, indices, _, _, preparation = evaluation.charger_mnist_test(
                chemin,
                20,
            )
            np.testing.assert_array_equal(indices, np.arange(10))
            self.assertFalse(preparation["sous_echantillonnage_applique"])
            self.assertEqual(preparation["methode_selection"], "tous les exemples")


if __name__ == "__main__":
    unittest.main()
