"""Garanties d'alignement, de répétition et d'isolation de la validation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import _bootstrap  # noqa: F401
from reconnaissance_chiffres.entrainement_cascade import preparer_donnees, repeter_cascade


class InjectionCascadeTests(unittest.TestCase):
    def test_dix_occurrences_exactes_et_labels(self):
        x = np.zeros((3, 28, 28, 1), dtype=np.float32)
        x[:, 5, 8, 0] = [0.1, 0.5, 1.0]
        y = np.array([7, 2, 9], dtype=np.uint8)
        images, labels = repeter_cascade(x, y, 10)
        self.assertEqual(images.shape, (30, 28, 28, 1))
        for image, label in zip(x, y):
            correspondances = np.all(images == image, axis=(1, 2, 3))
            self.assertEqual(int(correspondances.sum()), 10)
            np.testing.assert_array_equal(labels[correspondances], np.full(10, label))

    def test_cascade_train_seulement_et_validation_identique_2x_10x(self):
        mnist = np.zeros((100, 28, 28, 1), dtype=np.float32)
        mnist[:, 0, 0, 0] = np.arange(100) / 100
        labels = np.arange(100) % 10
        cascade = np.ones((2, 28, 28, 1), dtype=np.float32)
        cascade[1, 3, 4, 0] = 0.3
        y_cascade = np.array([4, 6])
        deux = preparer_donnees(mnist, labels, cascade, y_cascade, 2)
        dix = preparer_donnees(mnist, labels, cascade, y_cascade, 10)
        for a, b in zip(deux[2:6], dix[2:6]):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(set(dix[4]) & set(dix[5]))
        self.assertEqual(len(dix[0]), 85 + 20)
        self.assertEqual(len(dix[2]), 15)
        for image, label in zip(cascade, y_cascade):
            selection = np.all(dix[0] == image, axis=(1, 2, 3))
            self.assertEqual(int(selection.sum()), 10)
            np.testing.assert_array_equal(dix[1][selection], np.full(10, label))
            self.assertFalse(np.any(np.all(dix[2] == image, axis=(1, 2, 3))))
        for image, label in zip(dix[2], dix[3]):
            source = np.flatnonzero(np.all(mnist == image, axis=(1, 2, 3)))
            self.assertEqual(label, labels[source[0]])

    def test_rejet_labels_desalignes_ou_invalides(self):
        x = np.zeros((2, 28, 28, 1), dtype=np.float32)
        for y in (np.array([1]), np.array([1, 10]), np.array([1.0, 2.0])):
            with self.assertRaises(ValueError):
                repeter_cascade(x, y, 10)


if __name__ == "__main__":
    unittest.main()
