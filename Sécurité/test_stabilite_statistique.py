"""Tests sans entraînement du protocole de stabilité statistique."""

from __future__ import annotations

import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import test_stabilite as stabilite  # noqa: E402


def ligne_brute(
    config_id: str,
    seed: int,
    accuracy: float | None,
    status: str = "success",
) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in stabilite.RAW_COLUMNS}
    row.update(
        {
            "protocol_id": "protocole-test",
            "experiment_name": "synthetique",
            "run_id": f"{config_id}_{seed}",
            "config_id": config_id,
            "model_mode": "cnn_2conv_parametrique",
            "filter_1": 4 if config_id == "reference" else 8,
            "filter_2": 8,
            "kernel_1": 5,
            "kernel_2": 4,
            "activation_conv": "softmax",
            "dropout": 0.3,
            "seed": seed,
            "repetition": seed - 41,
            "runs_requested_for_config": 3,
            "split_seed": 2026,
            "epochs_requested": 4,
            "epochs_completed": 2 if status == "success" else None,
            "best_epoch": 2 if status == "success" else None,
            "batch_size": 128,
            "learning_rate": 0.001,
            "train_samples": 100,
            "validation_samples": 20,
            "test_samples": 30,
            "model_parameters": 1_000,
            "best_train_accuracy": accuracy,
            "best_val_accuracy": (
                accuracy - 0.001 if accuracy is not None else None
            ),
            "best_val_loss": 0.1 if status == "success" else None,
            "test_accuracy": accuracy,
            "test_loss": 1.0 - accuracy if accuracy is not None else None,
            "duration_seconds": 1.0,
            "status": status,
            "error_type": None if status == "success" else "SyntheticError",
            "error_message": None if status == "success" else "erreur de test",
            "completed_at_utc": "2026-08-10T00:00:00+00:00",
        }
    )
    return row


class StabiliteStatistiqueTests(unittest.TestCase):
    def test_configuration_par_defaut_preserve_architecture_historique(self) -> None:
        parser = stabilite.build_parser()
        args = parser.parse_args(["--dry-run"])
        configurations = stabilite.build_configurations(
            args,
            {"mode": "cnn_2conv_parametrique", "resolved_path": None},
        )
        self.assertEqual(len(configurations), 1)
        configuration = configurations[0]
        self.assertEqual(configuration["filter_1"], 4)
        self.assertEqual(configuration["filter_2"], 8)
        self.assertEqual(configuration["kernel_1"], 5)
        self.assertEqual(configuration["kernel_2"], 4)
        self.assertEqual(configuration["activation_conv"], "softmax")
        self.assertEqual(configuration["dropout"], 0.3)

    def test_identifiants_float_et_runs_sans_collision(self) -> None:
        first = 0.123456781
        second = 0.123456782
        self.assertNotEqual(stabilite.float_token(first), stabilite.float_token(second))

        parser = stabilite.build_parser()
        args = parser.parse_args(
            ["--dry-run", "--dropouts", f"{first},{second}", "--seeds", "42"]
        )
        configurations = stabilite.build_configurations(
            args, {"mode": "cnn_2conv_parametrique", "resolved_path": None}
        )
        runs = stabilite.planned_runs("a" * 64, configurations, (42,))
        self.assertEqual(len({row["config_id"] for row in configurations}), 2)
        self.assertEqual(len({row["run_id"] for row in runs}), 2)

    def test_protocol_inclut_script_et_versions(self) -> None:
        parser = stabilite.build_parser()
        args = parser.parse_args(["--dry-run", "--seeds", "42"])
        dataset = {
            "identity": "dataset-synthetique",
            "sha256": "d" * 64,
            "normalization": {
                "policy": "test",
                "mode": "deja_0_1",
                "divisor": 1.0,
            },
        }
        model = {
            "mode": "cnn_2conv_parametrique",
            "resolved_path": None,
            "sha256": None,
        }
        configurations = stabilite.build_configurations(args, model)
        protocol = stabilite.protocol_payload(args, dataset, model, configurations)
        implementation = protocol["implementation"]
        self.assertEqual(
            implementation["script_sha256"],
            stabilite.hash_file(Path(stabilite.__file__).resolve()),
        )
        self.assertEqual(protocol["dataset"]["sha256"], "d" * 64)
        self.assertEqual(protocol["protocol_schema"], "stabilite_mnist_v3")
        self.assertTrue(
            {"keras", "tensorflow", "numpy", "pandas", "scipy", "scikit-learn"}
            <= set(implementation["libraries"])
        )

    def test_intervalle_student_et_cas_un_seul_run(self) -> None:
        summary = stabilite.student_summary([0.98, 0.99, 1.0], 0.95)
        self.assertAlmostEqual(summary["mean"], 0.99)
        self.assertAlmostEqual(summary["std"], 0.01)
        self.assertAlmostEqual(summary["sem"], 0.01 / math.sqrt(3))
        self.assertAlmostEqual(
            summary["ci_high"] - summary["mean"], 0.0248413771, places=7
        )
        single = stabilite.student_summary([0.99], 0.95)
        self.assertTrue(math.isnan(single["std"]))
        self.assertTrue(math.isnan(single["ci_low"]))

    def test_historique_utilise_nan_apres_early_stopping(self) -> None:
        run = {"run_id": "run", "config_id": "config", "seed": 42}
        rows = stabilite.history_rows(
            "protocole-test",
            run,
            4,
            {
                "loss": [0.4, 0.2],
                "accuracy": [0.8, 0.9],
                "val_loss": [0.5, 0.3],
                "val_accuracy": [0.75, 0.88],
            },
        )
        self.assertTrue(rows[1]["reached"])
        self.assertFalse(rows[2]["reached"])
        self.assertTrue(math.isnan(rows[2]["val_accuracy"]))

    def test_metriques_non_finies_ne_peuvent_pas_devenir_un_succes(self) -> None:
        history = {
            "loss": [0.4, 0.2],
            "accuracy": [0.8, 0.9],
            "val_loss": [0.5, 0.3],
            "val_accuracy": [0.75, 0.88],
        }
        validated = stabilite.validate_success_metrics(
            history, {"loss": 0.1, "accuracy": 0.99}, epochs_requested=4
        )
        self.assertEqual(validated["epochs_completed"], 2)
        with self.assertRaisesRegex(FloatingPointError, "NaN ou inf"):
            stabilite.validate_success_metrics(
                history, {"loss": 0.1, "accuracy": math.nan}, epochs_requested=4
            )
        invalid_history = {**history, "val_accuracy": [0.75, 1.01]}
        with self.assertRaisesRegex(ValueError, "hors plage"):
            stabilite.validate_success_metrics(
                invalid_history,
                {"loss": 0.1, "accuracy": 0.99},
                epochs_requested=4,
            )

    def test_integrite_refuse_protocole_etranger_et_doublons(self) -> None:
        mixed_rows = [
            ligne_brute("reference", 42, 0.99),
            ligne_brute("candidate", 43, 0.98),
        ]
        mixed_rows[1]["protocol_id"] = "autre-protocole"
        mixed = pd.DataFrame(mixed_rows, columns=stabilite.RAW_COLUMNS)
        with self.assertRaisesRegex(ValueError, "protocol_id"):
            stabilite.aggregate_results(mixed, 0.95, 2, "protocole-test")

        duplicate = pd.DataFrame(
            [
                ligne_brute("reference", 42, 0.99),
                ligne_brute("reference", 42, 0.99),
            ],
            columns=stabilite.RAW_COLUMNS,
        )
        with self.assertRaisesRegex(ValueError, "dupliqu"):
            stabilite.paired_comparisons(
                duplicate, "reference", 0.95, "protocole-test"
            )
        raw = pd.DataFrame(
            [ligne_brute("reference", 42, 0.99)],
            columns=stabilite.RAW_COLUMNS,
        )
        run = {"run_id": "reference_42", "config_id": "reference", "seed": 42}
        history_rows = stabilite.history_rows(
            "protocole-test",
            run,
            2,
            {
                "loss": [0.2],
                "accuracy": [0.9],
                "val_loss": [0.3],
                "val_accuracy": [0.88],
            },
        )
        duplicated_history = pd.DataFrame(
            [history_rows[0], history_rows[0]], columns=stabilite.HISTORY_COLUMNS
        )
        with self.assertRaisesRegex(ValueError, "époque"):
            stabilite.validate_result_frames(
                raw, duplicated_history, "protocole-test"
            )

    def test_csv_success_avec_nan_est_refuse(self) -> None:
        raw = pd.DataFrame(
            [ligne_brute("reference", 42, math.nan)],
            columns=stabilite.RAW_COLUMNS,
        )
        with self.assertRaisesRegex(ValueError, "success.*test_accuracy"):
            stabilite.aggregate_results(raw, 0.95, 1, "protocole-test")

    def test_code_retour_non_nul_sans_aucun_succes(self) -> None:
        errors = pd.DataFrame(
            [ligne_brute("echec", 42, None, status="error")],
            columns=stabilite.RAW_COLUMNS,
        )
        success = pd.DataFrame(
            [ligne_brute("reference", 42, 0.99)],
            columns=stabilite.RAW_COLUMNS,
        )
        self.assertEqual(stabilite.result_exit_code(errors), 1)
        self.assertEqual(stabilite.result_exit_code(success), 0)

    def test_agregation_conserve_erreurs_et_comparaison_est_pairee(self) -> None:
        rows = []
        reference = [0.980, 0.990, 1.000]
        candidate = [0.985, 0.992, 0.998]
        for seed, accuracy in zip((42, 43, 44), reference):
            rows.append(ligne_brute("reference", seed, accuracy))
        for seed, accuracy in zip((42, 43, 44), candidate):
            rows.append(ligne_brute("candidate", seed, accuracy))
        rows.append(ligne_brute("echec", 42, None, status="error"))
        raw = pd.DataFrame(rows, columns=stabilite.RAW_COLUMNS)

        aggregated = stabilite.aggregate_results(
            raw,
            confidence_level=0.95,
            expected_seed_count=3,
            protocol_id="protocole-test",
        )
        comparisons = stabilite.paired_comparisons(
            raw,
            reference_config="reference",
            confidence_level=0.95,
            protocol_id="protocole-test",
        )

        failed = aggregated[aggregated["config_id"] == "echec"].iloc[0]
        paired = comparisons[comparisons["candidate_config_id"] == "candidate"].iloc[0]
        self.assertEqual(failed["runs_successful"], 0)
        self.assertEqual(failed["runs_failed"], 1)
        self.assertTrue(math.isnan(failed["test_accuracy_mean"]))
        self.assertEqual(paired["paired_seeds"], 3)
        self.assertAlmostEqual(
            paired["delta_test_accuracy_mean"],
            sum(c - r for c, r in zip(candidate, reference)) / 3,
        )

    def test_reprise_refuse_un_succes_sans_historique_complet(self) -> None:
        raw = pd.DataFrame(
            [ligne_brute("reference", 42, 0.99)], columns=stabilite.RAW_COLUMNS
        )
        run = {"run_id": "reference_42", "config_id": "reference", "seed": 42}
        complete = pd.DataFrame(
            stabilite.history_rows(
                "protocole-test",
                run,
                4,
                {
                    "loss": [0.4, 0.2],
                    "accuracy": [0.8, 0.9],
                    "val_loss": [0.5, 0.3],
                    "val_accuracy": [0.75, 0.88],
                },
            ),
            columns=stabilite.HISTORY_COLUMNS,
        )
        incomplete = complete.iloc[1:].copy()
        self.assertEqual(
            stabilite.resumable_success_ids(raw, complete), {"reference_42"}
        )
        self.assertEqual(stabilite.resumable_success_ids(raw, incomplete), set())

    def test_validation_du_modele_gabarit(self) -> None:
        class Couche:
            def __init__(self, activation: str) -> None:
                self.activation = activation

            def get_config(self) -> dict[str, str]:
                return {"activation": self.activation}

        class Modele:
            def __init__(
                self,
                input_shape: tuple[object, ...] = (None, 28, 28, 1),
                output_shape: tuple[object, ...] = (None, 10),
                activation: str = "softmax",
            ) -> None:
                self.input_shape = input_shape
                self.output_shape = output_shape
                self.layers = [Couche(activation)]

        stabilite.validate_template_model(Modele())
        with self.assertRaisesRegex(ValueError, "sortie"):
            stabilite.validate_template_model(Modele(output_shape=(None, 1)))
        with self.assertRaisesRegex(ValueError, "softmax"):
            stabilite.validate_template_model(Modele(activation="linear"))

    def test_normalisation_npz_est_conditionnelle_et_tracee(self) -> None:
        labels = np.arange(10, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized_path = root / "normalise.npz"
            raw_path = root / "brut.npz"
            mixed_path = root / "mixte.npz"
            normalized = np.zeros((10, 28, 28), dtype=np.float32)
            normalized[:, 0, 0] = 1.0
            raw = np.zeros((10, 28, 28), dtype=np.uint8)
            raw[:, 0, 0] = 255
            np.savez(
                normalized_path,
                x_train=normalized,
                y_train=labels,
                x_test=normalized,
                y_test=labels,
            )
            np.savez(
                raw_path,
                x_train=raw,
                y_train=labels,
                x_test=raw,
                y_test=labels,
            )
            np.savez(
                mixed_path,
                x_train=normalized,
                y_train=labels,
                x_test=raw,
                y_test=labels,
            )
            self.assertEqual(
                stabilite.inspect_npz_normalization(normalized_path)["divisor"],
                1.0,
            )
            self.assertEqual(
                stabilite.inspect_npz_normalization(raw_path)["divisor"], 255.0
            )
            with self.assertRaisesRegex(ValueError, "même échelle"):
                stabilite.inspect_npz_normalization(mixed_path)

    def test_split_existant_est_relu_valide_et_non_recalcule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.npz"
            train_labels = np.tile(np.arange(10, dtype=np.uint8), 10)
            test_labels = np.tile(np.arange(10, dtype=np.uint8), 2)
            train_images = np.zeros((100, 28, 28), dtype=np.uint8)
            test_images = np.zeros((20, 28, 28), dtype=np.uint8)
            train_images[:, 0, 0] = 255
            test_images[:, 0, 0] = 255
            np.savez(
                dataset_path,
                x_train=train_images,
                y_train=train_labels,
                x_test=test_images,
                y_test=test_labels,
            )
            parser = stabilite.build_parser()
            args = parser.parse_args(
                [
                    "--dry-run",
                    "--dataset",
                    str(dataset_path),
                    "--validation-split",
                    "0.2",
                    "--seeds",
                    "42",
                ]
            )
            descriptor = stabilite.resolve_dataset_descriptor(parser, args)
            paths = stabilite.StabilityPaths(root / "experience")
            paths.create()
            protocol_id = "p" * 64
            first = stabilite.load_and_split_dataset(
                args, descriptor, paths, protocol_id
            )
            with mock.patch.object(
                stabilite,
                "stratified_subset_indices",
                side_effect=AssertionError("le split ne doit pas être recalculé"),
            ):
                second = stabilite.load_and_split_dataset(
                    args, descriptor, paths, protocol_id
                )
            self.assertTrue(np.array_equal(first[0], second[0]))
            with self.assertRaisesRegex(ValueError, "autre protocol_id"):
                stabilite.load_split_indices(paths.split_indices, "q" * 64)

    def test_reprise_refuse_resultats_sans_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = stabilite.StabilityPaths(Path(temporary) / "experience")
            paths.create()
            protocol = {"schema": "synthetique"}
            protocol_id = stabilite.make_protocol_id(protocol)
            stabilite.write_json_atomic(
                paths.configuration,
                {"protocol_id": protocol_id, "protocol": protocol},
            )
            row = ligne_brute("reference", 42, 0.99)
            row["protocol_id"] = protocol_id
            pd.DataFrame([row], columns=stabilite.RAW_COLUMNS).to_csv(
                paths.raw_csv, index=False
            )
            parser = stabilite.build_parser()
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    stabilite.validate_existing_experiment(
                        parser, paths, protocol_id
                    )


if __name__ == "__main__":
    unittest.main()
