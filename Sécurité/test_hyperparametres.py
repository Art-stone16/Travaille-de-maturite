"""Tests sans entraînement du protocole et des statistiques hyperparamètres."""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyser_hyperparametres as analyse  # noqa: E402
import generer_color_map as grille  # noqa: E402


class ProtocoleHyperparametresTests(unittest.TestCase):
    def test_catalogue_ne_liste_que_les_surfaces_existantes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(
                grille,
                "OUTPUT_ROOT",
                Path(temporary_directory),
            ):
                paths = grille.ExperimentPaths("catalogue_relu_seulement")
            paths.surface_details.mkdir(parents=True)
            paths.surface_relu.touch()

            grille.write_output_catalog(paths)
            catalogue = pd.read_csv(paths.catalog)

        surfaces = catalogue.loc[
            catalogue["type"] == "figure PNG détaillée", "path"
        ].tolist()
        self.assertEqual(
            surfaces,
            ["graphiques/surfaces_3d_detaillees/activation_relu.png"],
        )

    def test_surfaces_3d_creent_une_vue_detaillee_par_activation(self) -> None:
        aggregated = pd.DataFrame(
            {
                "conv_activation": ["relu", "relu", "softmax", "softmax"],
                "dropout": [0.2, 0.4, 0.2, 0.4],
            }
        )
        output_directory = Path(
            "/tmp/experience/graphiques/surfaces_3d_detaillees"
        )

        with mock.patch.object(grille, "_render_surfaces_3d") as renderer:
            grille.plot_surfaces_3d(aggregated, output_directory, show=False)

        self.assertEqual(renderer.call_count, 2)
        relu_detail, softmax_detail = renderer.call_args_list
        self.assertEqual(
            relu_detail.args[1],
            output_directory / "activation_relu.png",
        )
        self.assertEqual(relu_detail.kwargs["detail_activation"], "relu")
        self.assertEqual(
            softmax_detail.args[1],
            output_directory / "activation_softmax.png",
        )
        self.assertEqual(softmax_detail.kwargs["detail_activation"], "softmax")

    def test_activations_reelles_du_projet_sont_acceptees(self) -> None:
        self.assertEqual(
            grille.comma_separated_activations("relu,softmax,relu"),
            ("relu", "softmax"),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            grille.comma_separated_activations("activation_inconnue")

    def test_protocol_id_change_avec_activation_et_graines(self) -> None:
        parser = grille.build_parser()
        args = parser.parse_args(["--dry-run"])
        args.mnist_path_resolved = None
        original = grille.make_protocol_id(args, (42, 43, 44))
        args.activations = ("relu", "softmax")
        changed_activation = grille.make_protocol_id(args, (42, 43, 44))
        changed_seeds = grille.make_protocol_id(args, (42, 43, 45))
        self.assertNotEqual(original, changed_activation)
        self.assertNotEqual(changed_activation, changed_seeds)

        args.mnist_sha256 = "a" * 64
        first_dataset = grille.make_protocol_id(args, (42, 43, 45))
        args.mnist_sha256 = "b" * 64
        second_dataset = grille.make_protocol_id(args, (42, 43, 45))
        self.assertNotEqual(first_dataset, second_dataset)

        with mock.patch.object(grille, "implementation_sha256", return_value="1" * 64):
            first_code = grille.make_protocol_id(args, (42, 43, 45))
        with mock.patch.object(grille, "implementation_sha256", return_value="2" * 64):
            second_code = grille.make_protocol_id(args, (42, 43, 45))
        self.assertNotEqual(first_code, second_code)

    def test_run_id_preserve_legacy_et_distingue_dropouts_proches(self) -> None:
        historical = grille.make_run_id(4, 8, 0.4, 42)
        close_value = math.nextafter(0.4, 1.0)
        closer_again = math.nextafter(close_value, 1.0)
        close_identifier = grille.make_run_id(4, 8, close_value, 42)
        closer_identifier = grille.make_run_id(4, 8, closer_again, 42)

        self.assertEqual(historical, "f1_4__f2_8__dropout_0p4__seed_42")
        self.assertNotEqual(historical, close_identifier)
        self.assertNotEqual(close_identifier, closer_identifier)
        self.assertIn("dropout_v2_", close_identifier)
        self.assertEqual(analyse.numeric_identifier_token(0.4), "0p4")
        self.assertNotEqual(
            analyse.numeric_identifier_token(0.4),
            analyse.numeric_identifier_token(close_value),
        )

    def test_csv_brut_sans_configuration_est_refuse_dans_les_deux_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_path = root / "donnees" / "resultats_bruts.csv"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text("run_id,status\nx,success\n", encoding="utf-8")
            paths = SimpleNamespace(
                raw_csv=raw_path,
                configuration=root / "configuration.json",
            )
            parser = argparse.ArgumentParser(prog="test-protocole")
            for plot_only in (False, True):
                with self.subTest(plot_only=plot_only):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            grille.validate_existing_experiment(
                                parser,
                                SimpleNamespace(plot_only=plot_only),
                                paths,
                                (42,),
                                "sha256:" + "a" * 64,
                            )

    def test_valeurs_non_finies_et_graines_hors_bornes_sont_refusees(self) -> None:
        for value in ("nan", "inf", "-inf"):
            with self.subTest(dropout=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    grille.comma_separated_floats(value)

        parser = grille.build_parser()
        args = parser.parse_args(["--learning-rate", "nan"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                grille.validate_args(parser, args)
        with self.assertRaises(argparse.ArgumentTypeError):
            grille.comma_separated_seeds(str(grille.MAX_RANDOM_SEED + 1))

        args = parser.parse_args(
            ["--seed-base", str(grille.MAX_RANDOM_SEED), "--repetitions", "2"]
        )
        with self.assertRaises(ValueError):
            grille.resolve_seeds(args)

    def test_reprise_ignore_un_succes_aux_metriques_incoherentes(self) -> None:
        protocol_id = "sha256:" + "c" * 64
        run = {
            "protocol_id": protocol_id,
            "run_id": grille.make_run_id(4, 8, 0.4, 42),
            "conv_activation": "relu",
            "output_activation": "softmax",
            "filter_1": 4,
            "filter_2": 8,
            "dropout": 0.4,
            "repetition": 1,
            "seed": 42,
        }
        row = {
            "experiment_name": "test",
            **run,
            "repetitions_requested": 1,
            "epochs_requested": 20,
            "epochs_completed": 10,
            "best_epoch": 8,
            "batch_size": 128,
            "validation_split": 0.15,
            "learning_rate": 0.001,
            "train_samples": 51_000,
            "validation_samples": 9_000,
            "test_samples": 10_000,
            "model_parameters": 2250,
            "best_val_accuracy": 0.989,
            "best_val_loss": 0.03,
            "test_accuracy": 0.99,
            "test_loss": 0.03,
            "duration_seconds": 20.0,
            "status": "success",
            "error_message": None,
            "completed_at_utc": "2026-08-10T00:00:00+00:00",
        }
        valid = pd.DataFrame([row])[grille.RAW_COLUMNS]
        self.assertEqual(
            grille.resumable_success_ids(valid, [run], protocol_id),
            {run["run_id"]},
        )
        invalid = valid.copy()
        invalid.loc[0, "test_accuracy"] = math.nan
        self.assertEqual(
            grille.resumable_success_ids(invalid, [run], protocol_id),
            set(),
        )

    def test_aggregation_conserve_cellule_tout_echec_et_calcule_ic95(self) -> None:
        protocol_id = "sha256:" + "b" * 64
        rows = []
        for filter_1, status_values in ((4, ("success", "success", "success")), (128, ("error", "error", "error"))):
            for repetition, (seed, status) in enumerate(
                zip((42, 43, 44), status_values), start=1
            ):
                success = status == "success"
                rows.append(
                    {
                        "experiment_name": "test",
                        "protocol_id": protocol_id,
                        "run_id": grille.make_run_id(filter_1, 8, 0.4, seed),
                        "conv_activation": "relu",
                        "output_activation": "softmax",
                        "filter_1": filter_1,
                        "filter_2": 8,
                        "dropout": 0.4,
                        "repetition": repetition,
                        "seed": seed,
                        "repetitions_requested": 3,
                        "epochs_requested": 20,
                        "epochs_completed": 10 if success else None,
                        "best_epoch": 8 if success else None,
                        "batch_size": 128,
                        "validation_split": 0.15,
                        "learning_rate": 0.001,
                        "train_samples": 51_000,
                        "validation_samples": 9_000,
                        "test_samples": 10_000,
                        "model_parameters": 2_250 if success else None,
                        "best_val_accuracy": 0.989 if success else None,
                        "best_val_loss": 0.03 if success else None,
                        "test_accuracy": (0.99 + (seed - 43) * 0.001) if success else None,
                        "test_loss": 0.03 if success else None,
                        "duration_seconds": 20.0,
                        "status": status,
                        "error_message": None if success else "synthetic OOM",
                        "completed_at_utc": "2026-08-10T00:00:00+00:00",
                    }
                )
        raw = pd.DataFrame(rows)[grille.RAW_COLUMNS]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "resultats_agreges.csv"
            aggregated = grille.aggregate_results(
                raw,
                SimpleNamespace(aggregated_csv=path),
            )
        failed = aggregated[aggregated["filter_1"] == 128].iloc[0]
        successful = aggregated[aggregated["filter_1"] == 4].iloc[0]
        self.assertEqual(failed["runs_completed"], 0)
        self.assertEqual(failed["runs_failed"], 3)
        self.assertTrue(failed["attempts_complete"])
        self.assertFalse(failed["is_complete"])
        self.assertTrue(math.isnan(failed["test_accuracy_mean"]))
        self.assertAlmostEqual(successful["test_accuracy_sem"], 0.001 / math.sqrt(3))
        self.assertLess(successful["test_accuracy_ci95_low"], 0.99)
        self.assertGreater(successful["test_accuracy_ci95_high"], 0.99)

    def test_table_analytique_utilise_le_grain_configuration(self) -> None:
        aggregated = pd.DataFrame(
            [
                {
                    "protocol_id": "legacy-v1",
                    "conv_activation": "relu",
                    "output_activation": "softmax",
                    "filter_1": 4,
                    "filter_2": 8,
                    "dropout": 0.4,
                    "runs_completed": 3,
                    "runs_failed": 0,
                    "model_parameters": 2250,
                    "test_accuracy_mean": 0.99,
                    "test_accuracy_std": 0.001,
                }
            ]
        )
        raw = pd.DataFrame(columns=grille.RAW_COLUMNS)
        analytical = analyse.build_analytical_table(raw, aggregated)
        self.assertEqual(len(analytical), 1)
        self.assertEqual(analytical.iloc[0]["analysis_grain"], "une ligne par configuration")
        self.assertAlmostEqual(analytical.iloc[0]["log2_filter_1"], 2.0)

    def test_protocoles_mixtes_sont_refuses_avant_correlation(self) -> None:
        analytical = pd.DataFrame(
            {
                "protocol_id": ["protocole-a", "protocole-b"],
                "conv_activation": ["relu", "relu"],
                "log2_filter_1": [2.0, 3.0],
                "log2_filter_2": [3.0, 4.0],
                "dropout": [0.2, 0.4],
                "test_accuracy_mean": [0.98, 0.99],
            }
        )
        with self.assertRaisesRegex(ValueError, "plusieurs protocol_id"):
            analyse.correlation_tables(analytical)

    def test_png_preexistant_est_catalogue_comme_ancien_si_rendu_echoue(self) -> None:
        aggregated = pd.DataFrame(
            [
                {
                    "protocol_id": "legacy-v1",
                    "conv_activation": "relu",
                    "output_activation": "softmax",
                    "filter_1": 4,
                    "filter_2": 8,
                    "dropout": 0.4,
                    "runs_completed": 1,
                    "runs_failed": 0,
                    "model_parameters": 2250,
                    "test_accuracy_mean": 0.99,
                    "test_accuracy_std": math.nan,
                    "duration_seconds_mean": 1.0,
                }
            ]
        )
        raw = pd.DataFrame(columns=grille.RAW_COLUMNS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stale_path = (
                root
                / "graphiques"
                / "correlations"
                / "scatter_accuracy_hyperparametres.png"
            )
            stale_path.parent.mkdir(parents=True)
            stale_path.write_bytes(b"ancienne figure")

            paths = analyse.generate_analysis_outputs(raw, aggregated, root)
            catalog = pd.read_csv(paths["catalog"])
            row = catalog[catalog["key"] == "scatter_relationships"].iloc[0]
            self.assertEqual(row["status"], "ancien_non_actualise")
            self.assertEqual(stale_path.read_bytes(), b"ancienne figure")

    def test_jitter_ne_depasse_pas_dix_pourcent_du_pas_minimal(self) -> None:
        width = analyse.dynamic_jitter_width([0.0, 0.025, 0.1])
        offsets = analyse.deterministic_jitter(7, width)
        self.assertAlmostEqual(width, 0.0025)
        self.assertLessEqual(max(abs(float(value)) for value in offsets), width)
        self.assertEqual(analyse.dynamic_jitter_width([0.4, 0.4]), 0.0)

    def test_zero_succes_produit_un_code_retour_non_nul(self) -> None:
        all_failed = pd.DataFrame({"runs_completed": [0, 0, 0]})
        one_success = pd.DataFrame({"runs_completed": [0, 1, 0]})
        self.assertEqual(
            grille.completion_exit_code(all_failed),
            grille.NO_SUCCESS_EXIT_CODE,
        )
        self.assertEqual(grille.completion_exit_code(one_success), 0)

    def test_plot_only_lit_id_enregistre_et_retourne_deux_si_tout_echoue(self) -> None:
        def write_experiment(name: str, success: bool) -> str:
            protocol_id = "sha256:" + (("d" if success else "e") * 64)
            paths = grille.ExperimentPaths(name)
            paths.create()
            paths.configuration.write_text(
                '{"schema_version": 2, "protocol_id": "'
                + protocol_id
                + '", "experiment": {"display_name": "test"}}\n',
                encoding="utf-8",
            )
            row = {
                "experiment_name": name,
                "protocol_id": protocol_id,
                "run_id": grille.make_run_id(4, 8, 0.4, 42),
                "conv_activation": "relu",
                "output_activation": "softmax",
                "filter_1": 4,
                "filter_2": 8,
                "dropout": 0.4,
                "repetition": 1,
                "seed": 42,
                "repetitions_requested": 1,
                "epochs_requested": 2,
                "epochs_completed": 2 if success else None,
                "best_epoch": 1 if success else None,
                "batch_size": 128,
                "validation_split": 0.15,
                "learning_rate": 0.001,
                "train_samples": 100,
                "validation_samples": 20,
                "test_samples": 20,
                "model_parameters": 2250 if success else None,
                "best_val_accuracy": 0.98 if success else None,
                "best_val_loss": 0.04 if success else None,
                "test_accuracy": 0.97 if success else None,
                "test_loss": 0.05 if success else None,
                "duration_seconds": 1.0,
                "status": "success" if success else "error",
                "error_message": None if success else "synthetic OOM",
                "completed_at_utc": "2026-08-10T00:00:00+00:00",
            }
            pd.DataFrame([row])[grille.RAW_COLUMNS].to_csv(
                paths.raw_csv,
                index=False,
            )
            return protocol_id

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            with mock.patch.object(grille, "OUTPUT_ROOT", temporary_root):
                success_protocol = write_experiment(
                    "plot_success",
                    True,
                )
                failed_protocol = write_experiment(
                    "plot_failed",
                    False,
                )
                with (
                    mock.patch.object(
                        grille,
                        "make_protocol_id",
                        side_effect=AssertionError("ne doit pas être recalculé"),
                    ),
                    mock.patch.object(grille, "plot_heatmaps"),
                    mock.patch.object(grille, "plot_scatter_3d"),
                    mock.patch.object(grille, "plot_surfaces_3d"),
                    mock.patch.object(analyse, "generate_analysis_outputs"),
                    contextlib.redirect_stdout(io.StringIO()) as stdout,
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        grille.main(
                            ["--nom-experience", "plot_success", "--plot-only"]
                        ),
                        0,
                    )
                    self.assertIn(success_protocol, stdout.getvalue())
                    stdout.seek(0)
                    stdout.truncate(0)
                    self.assertEqual(
                        grille.main(
                            ["--nom-experience", "plot_failed", "--plot-only"]
                        ),
                        grille.NO_SUCCESS_EXIT_CODE,
                    )
                    self.assertIn(failed_protocol, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
