#!/usr/bin/env python3
"""Analyser les relations hyperparamètres-métriques sans entraîner de modèle.

Le grain principal est une configuration unique
(``activation, filter_1, filter_2, dropout``), après agrégation des graines.
Les corrélations sont descriptives : elles ne prouvent pas un effet causal et
peuvent masquer des interactions entre hyperparamètres.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import env_config


MIN_SCATTER_POINTS = 8
DEFAULT_ACTIVATION = "relu"
DEFAULT_OUTPUT_ACTIVATION = "softmax"
ACTIVATION_COLORS = {
    "relu": "#356AA0",
    "softmax": "#D28A28",
}
ACTIVATION_MARKERS = {
    "relu": "o",
    "softmax": "^",
}
FALLBACK_COLORS = ("#6C7A89", "#9A6FB0", "#A56A43", "#667A3E")
FALLBACK_MARKERS = ("s", "D", "P", "X")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_dataframe_atomic(dataframe: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def require_single_protocol(*dataframes: Any) -> str | None:
    """Garantir qu'aucune analyse ne fusionne deux protocoles scientifiques."""
    identifiers: set[str] = set()
    blank_identifiers = False
    for dataframe in dataframes:
        if dataframe is None or "protocol_id" not in dataframe.columns:
            continue
        for value in dataframe["protocol_id"].tolist():
            if value is None:
                blank_identifiers = True
                continue
            token = str(value).strip()
            if not token or token.lower() == "nan":
                blank_identifiers = True
            else:
                identifiers.add(token)
    if blank_identifiers:
        raise ValueError("protocol_id vide ou manquant dans les données d'analyse")
    if len(identifiers) > 1:
        raise ValueError(
            "analyse refusée : plusieurs protocol_id seraient mélangés ("
            + ", ".join(sorted(identifiers))
            + "). Analysez chaque protocole dans un dossier séparé."
        )
    return next(iter(identifiers), None)


def numeric_identifier_token(value: float) -> str:
    """Token décimal à aller-retour exact pour distinguer des valeurs proches."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("valeur non finie interdite dans un identifiant")
    parsed = 0.0 if parsed == 0 else parsed
    short_text = format(parsed, ".8g")
    if float(short_text) == parsed:
        return short_text.replace(".", "p")
    return "v2_" + format(parsed, ".17g").replace(".", "p")


def normalize_inputs(raw_results: Any, aggregated: Any) -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    raw = raw_results.copy()
    configurations = aggregated.copy()
    for dataframe in (raw, configurations):
        if "conv_activation" not in dataframe.columns:
            dataframe["conv_activation"] = DEFAULT_ACTIVATION
        dataframe["conv_activation"] = dataframe["conv_activation"].fillna(
            DEFAULT_ACTIVATION
        ).astype(str)
        if "output_activation" not in dataframe.columns:
            dataframe["output_activation"] = DEFAULT_OUTPUT_ACTIVATION
        dataframe["output_activation"] = dataframe["output_activation"].fillna(
            DEFAULT_OUTPUT_ACTIVATION
        ).astype(str)
        if "protocol_id" not in dataframe.columns:
            dataframe["protocol_id"] = "legacy-v1"
        dataframe["protocol_id"] = dataframe["protocol_id"].fillna(
            "legacy-v1"
        ).astype(str)

    require_single_protocol(raw, configurations)

    numeric_raw = (
        "filter_1",
        "filter_2",
        "dropout",
        "seed",
        "epochs_completed",
        "model_parameters",
        "best_val_accuracy",
        "best_val_loss",
        "test_accuracy",
        "test_loss",
        "duration_seconds",
    )
    numeric_aggregated = (
        "filter_1",
        "filter_2",
        "dropout",
        "runs_completed",
        "runs_failed",
        "model_parameters",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_accuracy_sem",
        "test_accuracy_ci95_low",
        "test_accuracy_ci95_high",
        "test_loss_mean",
        "test_loss_std",
        "test_loss_sem",
        "test_loss_ci95_low",
        "test_loss_ci95_high",
        "duration_seconds_mean",
        "duration_seconds_std",
        "duration_seconds_sem",
        "duration_seconds_ci95_low",
        "duration_seconds_ci95_high",
        "epochs_completed_mean",
        "epochs_completed_std",
        "epochs_completed_sem",
        "epochs_completed_ci95_low",
        "epochs_completed_ci95_high",
        "best_val_accuracy_mean",
    )
    for column in numeric_raw:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
            raw[column] = raw[column].where(np.isfinite(raw[column]), math.nan)
    for column in numeric_aggregated:
        if column not in configurations.columns:
            configurations[column] = math.nan
        configurations[column] = pd.to_numeric(
            configurations[column], errors="coerce"
        )
        configurations[column] = configurations[column].where(
            np.isfinite(configurations[column]),
            math.nan,
        )
    return raw, configurations


def activation_styles(activations: Iterable[str]) -> dict[str, tuple[str, str]]:
    styles: dict[str, tuple[str, str]] = {}
    fallback_index = 0
    for activation in sorted(set(str(value) for value in activations)):
        if activation in ACTIVATION_COLORS:
            styles[activation] = (
                ACTIVATION_COLORS[activation],
                ACTIVATION_MARKERS[activation],
            )
        else:
            styles[activation] = (
                FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)],
                FALLBACK_MARKERS[fallback_index % len(FALLBACK_MARKERS)],
            )
            fallback_index += 1
    return styles


def build_analytical_table(raw_results: Any, aggregated: Any) -> Any:
    import numpy as np
    import pandas as pd

    raw, configurations = normalize_inputs(raw_results, aggregated)
    group_columns = [
        "protocol_id",
        "conv_activation",
        "output_activation",
        "filter_1",
        "filter_2",
        "dropout",
    ]
    successful = (
        raw[raw["status"] == "success"].copy()
        if "status" in raw.columns
        else raw.copy()
    )
    if not successful.empty:
        supplements = successful.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        ).agg(
            model_parameters_from_raw=("model_parameters", "max"),
            epochs_completed_mean_from_raw=("epochs_completed", "mean"),
            epochs_completed_std_from_raw=("epochs_completed", "std"),
            duration_seconds_mean_from_raw=("duration_seconds", "mean"),
            duration_seconds_std_from_raw=("duration_seconds", "std"),
        )
        configurations = configurations.merge(
            supplements,
            on=group_columns,
            how="left",
        )
        for target, source in (
            ("model_parameters", "model_parameters_from_raw"),
            ("epochs_completed_mean", "epochs_completed_mean_from_raw"),
            ("epochs_completed_std", "epochs_completed_std_from_raw"),
            ("duration_seconds_mean", "duration_seconds_mean_from_raw"),
            ("duration_seconds_std", "duration_seconds_std_from_raw"),
        ):
            configurations[target] = configurations[target].fillna(
                configurations[source]
            )
        configurations = configurations.drop(
            columns=[
                column
                for column in configurations.columns
                if column.endswith("_from_raw")
            ]
        )

    configurations["configuration_id"] = configurations.apply(
        lambda row: (
            f"{str(row['protocol_id']).replace('sha256:', '')[:10]}__"
            f"act_{row['conv_activation']}__f1_{int(row['filter_1'])}__"
            f"f2_{int(row['filter_2'])}__dropout_"
            f"{numeric_identifier_token(row['dropout'])}"
        ),
        axis=1,
    )
    configurations["log2_filter_1"] = np.log2(configurations["filter_1"])
    configurations["log2_filter_2"] = np.log2(configurations["filter_2"])
    positive_parameters = configurations["model_parameters"].where(
        configurations["model_parameters"] > 0
    )
    configurations["log10_model_parameters"] = np.log10(positive_parameters)
    if "best_val_accuracy_mean" in configurations.columns:
        configurations["validation_test_gap"] = (
            configurations["best_val_accuracy_mean"]
            - configurations["test_accuracy_mean"]
        )
    configurations["analysis_grain"] = "une ligne par configuration"
    preferred_columns = [
        "configuration_id",
        "protocol_id",
        "conv_activation",
        "output_activation",
        "filter_1",
        "filter_2",
        "dropout",
        "log2_filter_1",
        "log2_filter_2",
        "model_parameters",
        "log10_model_parameters",
        "runs_completed",
        "runs_failed",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_accuracy_sem",
        "test_accuracy_ci95_low",
        "test_accuracy_ci95_high",
        "test_loss_mean",
        "test_loss_std",
        "test_loss_sem",
        "test_loss_ci95_low",
        "test_loss_ci95_high",
        "best_val_accuracy_mean",
        "validation_test_gap",
        "duration_seconds_mean",
        "duration_seconds_std",
        "duration_seconds_sem",
        "duration_seconds_ci95_low",
        "duration_seconds_ci95_high",
        "epochs_completed_mean",
        "epochs_completed_std",
        "epochs_completed_sem",
        "epochs_completed_ci95_low",
        "epochs_completed_ci95_high",
        "analysis_grain",
    ]
    remaining = [
        column for column in configurations.columns if column not in preferred_columns
    ]
    return configurations[
        [column for column in preferred_columns if column in configurations.columns]
        + remaining
    ].sort_values(
        ["conv_activation", "filter_1", "filter_2", "dropout"],
        kind="stable",
    ).reset_index(drop=True)


CORRELATION_VARIABLES = (
    "log2_filter_1",
    "log2_filter_2",
    "dropout",
    "log10_model_parameters",
    "test_accuracy_mean",
    "test_loss_mean",
    "duration_seconds_mean",
    "epochs_completed_mean",
    "test_accuracy_std",
)


def correlation_tables(analytical: Any) -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    protocol_id = require_single_protocol(analytical)
    variables = [
        column
        for column in CORRELATION_VARIABLES
        if column in analytical.columns
        and analytical[column].notna().sum() >= 3
        and analytical[column].nunique(dropna=True) >= 2
    ]
    scopes: list[tuple[str, Any]] = [("toutes_activations", analytical)]
    if analytical["conv_activation"].nunique(dropna=True) > 1:
        scopes.extend(
            (f"activation_{activation}", subset)
            for activation, subset in analytical.groupby("conv_activation")
            if len(subset) >= MIN_SCATTER_POINTS
        )

    def make_table(method: str) -> Any:
        rows: list[dict[str, Any]] = []
        columns = [
            "protocol_id",
            "scope",
            "method",
            "variable_x",
            "variable_y",
            "coefficient",
            "n_configurations",
            "grain",
            "note",
        ]
        for scope, subset in scopes:
            for left_index, variable_x in enumerate(variables):
                for variable_y in variables[left_index + 1 :]:
                    paired = subset[[variable_x, variable_y]].dropna()
                    if len(paired) < 3:
                        continue
                    x_values = paired[variable_x]
                    y_values = paired[variable_y]
                    if x_values.nunique() < 2 or y_values.nunique() < 2:
                        continue
                    if method == "spearman":
                        coefficient = x_values.rank(method="average").corr(
                            y_values.rank(method="average"), method="pearson"
                        )
                    else:
                        coefficient = x_values.corr(y_values, method="pearson")
                    rows.append(
                        {
                            "protocol_id": protocol_id,
                            "scope": scope,
                            "method": method,
                            "variable_x": variable_x,
                            "variable_y": variable_y,
                            "coefficient": float(coefficient),
                            "n_configurations": int(len(paired)),
                            "grain": "configuration agrégée, jamais répétition brute",
                            "note": (
                                "descriptif uniquement ; interactions et grille "
                                "expérimentale peuvent masquer ou créer une tendance"
                            ),
                        }
                    )
        return pd.DataFrame(rows, columns=columns)

    return make_table("spearman"), make_table("pearson")


def focused_accuracy_limits(values: Any) -> tuple[float, float]:
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("aucune accuracy numérique à analyser")
    minimum = float(finite.min())
    maximum = float(finite.max())
    if math.isclose(minimum, maximum):
        padding = max(0.001, abs(minimum) * 0.002)
    else:
        padding = max((maximum - minimum) * 0.08, 0.0005)
    return max(0.0, minimum - padding), min(1.0, maximum + padding)


def dynamic_jitter_width(values: Any, maximum_fraction: float = 0.10) -> float:
    """Limiter le décalage à une fraction du plus petit espacement observé."""
    import numpy as np

    if (
        not math.isfinite(maximum_fraction)
        or maximum_fraction < 0
        or maximum_fraction > 0.10
    ):
        raise ValueError("maximum_fraction doit être comprise entre 0 et 0,10")
    finite = np.asarray(values, dtype=float)
    finite = np.unique(finite[np.isfinite(finite)])
    if len(finite) < 2:
        return 0.0
    spacings = np.diff(np.sort(finite))
    positive_spacings = spacings[spacings > 0]
    if not len(positive_spacings):
        return 0.0
    return float(positive_spacings.min() * maximum_fraction)


def deterministic_jitter(count: int, width: float = 0.0) -> Any:
    import numpy as np

    if not math.isfinite(width) or width < 0:
        raise ValueError("la largeur du jitter doit être finie et positive ou nulle")
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def add_activation_legend(axis: Any, styles: dict[str, tuple[str, str]]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markerfacecolor=color,
            markeredgecolor="#24313c",
            markersize=7,
            label=activation,
        )
        for activation, (color, marker) in styles.items()
    ]
    axis.legend(
        handles=handles,
        title="Activation des deux Conv2D",
        loc="best",
        frameon=True,
        fontsize=8.5,
        title_fontsize=8.5,
    )


def plot_accuracy_relationships(
    analytical: Any,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    require_single_protocol(analytical)
    valid = analytical.dropna(subset=["test_accuracy_mean"]).copy()
    if len(valid) < MIN_SCATTER_POINTS:
        raise ValueError(
            f"scatter accuracy non généré : {len(valid)} configurations, "
            f"minimum {MIN_SCATTER_POINTS}"
        )
    styles = activation_styles(valid["conv_activation"])
    specifications = (
        ("log2_filter_1", "filter_1", "Filtres de la 1re convolution"),
        ("log2_filter_2", "filter_2", "Filtres de la 2e convolution"),
        ("dropout", "dropout", "Dropout"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    y_min, y_max = focused_accuracy_limits(valid["test_accuracy_mean"])

    for axis, (x_column, level_column, label) in zip(axes, specifications):
        jitter_width = dynamic_jitter_width(valid[x_column])
        for activation, activation_subset in valid.groupby("conv_activation"):
            color, marker = styles[activation]
            for _, level_subset in activation_subset.groupby(level_column):
                ordered = level_subset.sort_values(
                    ["filter_1", "filter_2", "dropout"], kind="stable"
                )
                x_values = ordered[x_column].to_numpy(dtype=float)
                x_values = x_values + deterministic_jitter(
                    len(ordered),
                    jitter_width,
                )
                means = ordered["test_accuracy_mean"].to_numpy(dtype=float)
                lower = ordered["test_accuracy_ci95_low"].to_numpy(dtype=float)
                upper = ordered["test_accuracy_ci95_high"].to_numpy(dtype=float)
                finite_interval = np.isfinite(lower) & np.isfinite(upper)
                if bool(finite_interval.any()):
                    axis.errorbar(
                        x_values[finite_interval],
                        means[finite_interval],
                        yerr=np.vstack(
                            (
                                means[finite_interval] - lower[finite_interval],
                                upper[finite_interval] - means[finite_interval],
                            )
                        ),
                        fmt="none",
                        ecolor=color,
                        elinewidth=0.65,
                        alpha=0.25,
                        capsize=1.5,
                        zorder=1,
                    )
                axis.scatter(
                    x_values,
                    means,
                    s=34,
                    marker=marker,
                    facecolor=color,
                    edgecolor="#24313c",
                    linewidth=0.45,
                    alpha=0.62,
                    zorder=2,
                )

            marginal = activation_subset.groupby(level_column, as_index=False).agg(
                x=(x_column, "first"),
                accuracy=("test_accuracy_mean", "mean"),
            ).sort_values("x")
            axis.plot(
                marginal["x"],
                marginal["accuracy"],
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=6,
                markeredgecolor="#24313c",
                label=activation,
                zorder=3,
            )

        if level_column.startswith("filter"):
            levels = sorted(valid[level_column].dropna().unique())
            axis.set_xticks(
                [math.log2(float(level)) for level in levels],
                labels=[str(int(level)) for level in levels],
            )
        axis.set_xlabel(label)
        axis.set_ylim(y_min, y_max)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        axis.grid(True, color="#dce2e7", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy moyenne sur le jeu de test")
    add_activation_legend(axes[-1], styles)
    figure.suptitle(
        "Relations entre hyperparamètres et accuracy moyenne",
        fontsize=16,
        y=0.99,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"{len(valid)} points au grain configuration | axes filtres en log2 | "
            "traits épais = moyenne marginale | barres fines = IC95 lorsqu'il existe"
        ),
        ha="center",
        fontsize=9.5,
        color="#4a5965",
    )
    figure.text(
        0.5,
        0.02,
        (
            "Échelle d'accuracy focalisée et indiquée en pourcentage. "
            "Décalage horizontal déterministe ≤ 10 % du plus petit espacement ; "
            "relations descriptives, pas causales."
        ),
        ha="center",
        fontsize=9,
        color="#5e6b75",
    )
    figure.tight_layout(rect=(0.02, 0.07, 0.99, 0.89))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def pareto_frontier(dataframe: Any, cost_column: str) -> Any:
    require_single_protocol(dataframe)
    ordered = dataframe.dropna(
        subset=[cost_column, "test_accuracy_mean"]
    ).sort_values([cost_column, "test_accuracy_mean"], ascending=[True, False])
    rows = []
    best_accuracy = -math.inf
    for _, row in ordered.iterrows():
        accuracy = float(row["test_accuracy_mean"])
        if accuracy > best_accuracy:
            rows.append(row)
            best_accuracy = accuracy
    if not rows:
        return ordered.iloc[0:0]
    return ordered.__class__(rows)


def plot_cost_performance(analytical: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    require_single_protocol(analytical)
    valid = analytical.dropna(subset=["test_accuracy_mean"]).copy()
    if len(valid) < MIN_SCATTER_POINTS:
        raise ValueError(
            f"scatter coût-performance non généré : {len(valid)} configurations"
        )
    styles = activation_styles(valid["conv_activation"])
    panels = (
        ("model_parameters", "Nombre de paramètres", True),
        ("duration_seconds_mean", "Durée moyenne d'entraînement (s)", True),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    y_min, y_max = focused_accuracy_limits(valid["test_accuracy_mean"])

    for axis, (cost_column, label, logarithmic) in zip(axes, panels):
        panel_data = valid.dropna(subset=[cost_column])
        panel_data = panel_data[panel_data[cost_column] > 0]
        for activation, subset in panel_data.groupby("conv_activation"):
            color, marker = styles[activation]
            axis.scatter(
                subset[cost_column],
                subset["test_accuracy_mean"],
                s=48,
                marker=marker,
                facecolor=color,
                edgecolor="#24313c",
                linewidth=0.5,
                alpha=0.68,
            )
        frontier = pareto_frontier(panel_data, cost_column)
        if len(frontier) >= 2:
            axis.plot(
                frontier[cost_column],
                frontier["test_accuracy_mean"],
                color="#202a33",
                linewidth=1.5,
                linestyle="--",
                label="frontière de Pareto observée",
            )
        if logarithmic and len(panel_data):
            axis.set_xscale("log")
        axis.set_xlabel(label + (" — échelle log" if logarithmic else ""))
        axis.set_ylim(y_min, y_max)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        axis.grid(True, color="#dce2e7", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    top = valid.dropna(subset=["duration_seconds_mean"]).nlargest(
        min(3, len(valid)),
        "test_accuracy_mean",
    )
    top_labels: list[str] = []
    offsets = ((6, 8), (6, -13), (-13, 8))
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        axes[1].annotate(
            str(rank),
            (row["duration_seconds_mean"], row["test_accuracy_mean"]),
            xytext=offsets[(rank - 1) % len(offsets)],
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="#202a33",
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#5e6b75",
                "linewidth": 0.5,
                "alpha": 0.9,
            },
        )
        top_labels.append(
            f"{rank}. {row['conv_activation']} "
            f"{int(row['filter_1'])}/{int(row['filter_2'])}, "
            f"d={row['dropout']:g}"
        )
    if top_labels:
        axes[1].text(
            0.02,
            0.03,
            "Top accuracy\n" + "\n".join(top_labels),
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="#202a33",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#c9d1d8",
                "alpha": 0.88,
            },
        )
    axes[0].set_ylabel("Accuracy moyenne sur le jeu de test")
    add_activation_legend(axes[0], styles)
    figure.suptitle("Coût mesuré et performance des configurations", fontsize=16)
    figure.text(
        0.5,
        0.02,
        (
            "Un point = une configuration agrégée. La frontière relie les gains "
            "observés sans configuration à la fois moins coûteuse et plus précise."
        ),
        ha="center",
        fontsize=9,
        color="#5e6b75",
    )
    figure.tight_layout(rect=(0.02, 0.07, 0.99, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_accuracy_histograms(raw_results: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    raw, _ = normalize_inputs(raw_results, raw_results.iloc[0:0].copy())
    if "status" in raw.columns:
        raw = raw[raw["status"] == "success"]
    raw = raw.dropna(subset=["test_accuracy", "dropout"])
    if raw.empty:
        raise ValueError("aucune accuracy brute disponible pour les histogrammes")
    activations = sorted(raw["conv_activation"].unique())
    dropouts = sorted(raw["dropout"].unique())
    values = raw["test_accuracy"].to_numpy(dtype=float)
    minimum, maximum = focused_accuracy_limits(values)
    bin_count = min(18, max(7, int(round(math.sqrt(len(values))))))
    bins = np.linspace(minimum, maximum, bin_count + 1)
    figure, axes = plt.subplots(
        len(activations),
        len(dropouts),
        figsize=(3.25 * len(dropouts), 3.1 * len(activations)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    styles = activation_styles(activations)
    for row_index, activation in enumerate(activations):
        color, _ = styles[activation]
        for column_index, dropout in enumerate(dropouts):
            axis = axes[row_index, column_index]
            subset = raw[
                (raw["conv_activation"] == activation)
                & (raw["dropout"] == dropout)
            ]["test_accuracy"].to_numpy(dtype=float)
            if len(subset):
                axis.hist(
                    subset,
                    bins=bins,
                    color=color,
                    edgecolor="#24313c",
                    linewidth=0.6,
                    alpha=0.72,
                )
                axis.axvline(
                    float(np.mean(subset)),
                    color="#202a33",
                    linewidth=1.3,
                    linestyle="--",
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "aucun essai réussi",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#6b7680",
                )
            axis.set_title(
                f"{activation} | dropout={dropout:g} | n={len(subset)}",
                fontsize=9,
            )
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            axis.grid(axis="y", color="#dce2e7", linewidth=0.6, alpha=0.75)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == len(activations) - 1:
                axis.set_xlabel("Accuracy d'un entraînement")
            if column_index == 0:
                axis.set_ylabel("Nombre d'entraînements")

    figure.suptitle("Distribution des accuracies par activation et dropout", fontsize=16)
    figure.text(
        0.5,
        0.015,
        (
            "Une observation = un entraînement (configuration × graine), pas une "
            "image de test ; chaque facette regroupe tous les niveaux de filtres. "
            "Classes et axes identiques entre facettes."
        ),
        ha="center",
        fontsize=9,
        color="#5e6b75",
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.995, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def histogram_bins(values: Any, minimum_bins: int = 7) -> Any:
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    minimum = float(finite.min())
    maximum = float(finite.max())
    if math.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.05, 1e-6)
        minimum -= padding
        maximum += padding
    count = min(16, max(minimum_bins, int(round(math.sqrt(len(finite))))))
    return np.linspace(minimum, maximum, count + 1)


def plot_stability_histograms(analytical: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    require_single_protocol(analytical)
    activations = sorted(analytical["conv_activation"].unique())
    if not activations:
        raise ValueError("aucune activation à représenter")
    specifications = (
        ("test_accuracy_std", "Écart-type de l'accuracy", True),
        ("duration_seconds_mean", "Durée moyenne (secondes)", False),
    )
    global_bins = {
        column: histogram_bins(analytical[column])
        for column, _, _ in specifications
    }
    if all(bins is None for bins in global_bins.values()):
        raise ValueError("dispersion et durée indisponibles")
    styles = activation_styles(activations)
    figure, axes = plt.subplots(
        len(activations),
        2,
        figsize=(11.5, 3.2 * len(activations)),
        squeeze=False,
        sharex="col",
        sharey=False,
    )
    for row_index, activation in enumerate(activations):
        color, _ = styles[activation]
        subset = analytical[analytical["conv_activation"] == activation]
        for column_index, (column, label, percent) in enumerate(specifications):
            axis = axes[row_index, column_index]
            values = subset[column].dropna().to_numpy(dtype=float)
            bins = global_bins[column]
            if len(values) and bins is not None:
                axis.hist(
                    values,
                    bins=bins,
                    color=color,
                    edgecolor="#24313c",
                    linewidth=0.65,
                    alpha=0.72,
                )
            else:
                explanation = (
                    "écart-type indisponible\n(une seule répétition)"
                    if column == "test_accuracy_std"
                    else "durée indisponible"
                )
                axis.text(
                    0.5,
                    0.5,
                    explanation,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#6b7680",
                )
            axis.set_title(f"Activation {activation} | n={len(values)}", fontsize=10)
            axis.set_xlabel(label)
            axis.set_ylabel("Configurations")
            if percent:
                axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=3))
            axis.grid(axis="y", color="#dce2e7", linewidth=0.6, alpha=0.75)
            axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("Distribution de la stabilité et du coût d'entraînement", fontsize=16)
    figure.text(
        0.5,
        0.015,
        "Une observation = une configuration agrégée ; mêmes classes par colonne.",
        ha="center",
        fontsize=9,
        color="#5e6b75",
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.995, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


CORRELATION_LABELS = {
    "log2_filter_1": "log2 f1",
    "log2_filter_2": "log2 f2",
    "dropout": "dropout",
    "log10_model_parameters": "log10 paramètres",
    "test_accuracy_mean": "accuracy",
    "test_loss_mean": "loss",
    "duration_seconds_mean": "durée",
    "epochs_completed_mean": "époques",
    "test_accuracy_std": "écart-type accuracy",
}


def plot_correlation_matrix(analytical: Any, output_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    require_single_protocol(analytical)
    variables = [
        column
        for column in CORRELATION_VARIABLES
        if column in analytical.columns
        and analytical[column].notna().sum() >= 3
        and analytical[column].nunique(dropna=True) >= 2
    ]
    if len(variables) < 2 or len(analytical) < MIN_SCATTER_POINTS:
        raise ValueError("données insuffisantes pour une matrice de corrélation")
    scopes: list[tuple[str, Any]] = [("Toutes activations", analytical)]
    if analytical["conv_activation"].nunique(dropna=True) > 1:
        scopes.extend(
            (f"Activation {activation}", subset)
            for activation, subset in analytical.groupby("conv_activation")
            if len(subset) >= MIN_SCATTER_POINTS
        )
    figure, axes = plt.subplots(
        1,
        len(scopes),
        figsize=(6.1 * len(scopes), 5.8),
        squeeze=False,
    )
    color_map = LinearSegmentedColormap.from_list(
        "blue_white_gold",
        ("#356AA0", "#f7f7f5", "#D28A28"),
    )
    last_image = None
    for axis, (scope_name, subset) in zip(axes.flat, scopes):
        ranked = subset[variables].rank(method="average")
        matrix = ranked.corr(method="pearson").to_numpy(dtype=float)
        last_image = axis.imshow(matrix, cmap=color_map, vmin=-1, vmax=1)
        labels = [CORRELATION_LABELS.get(variable, variable) for variable in variables]
        axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_title(f"{scope_name} | n={len(subset)}", fontsize=11)
        for row_index in range(len(variables)):
            for column_index in range(len(variables)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=(
                        "white"
                        if math.isfinite(value) and abs(value) >= 0.62
                        else "#202a33"
                    ),
                )
        axis.tick_params(labelsize=8)
    if last_image is not None:
        color_axis = figure.add_axes([0.935, 0.27, 0.012, 0.48])
        color_bar = figure.colorbar(last_image, cax=color_axis)
        color_bar.set_label("Coefficient de Spearman")
    figure.suptitle("Corrélations descriptives au grain configuration", fontsize=16)
    figure.text(
        0.5,
        0.015,
        (
            "Filtres transformés en log2, paramètres en log10. Corrélation ≠ causalité ; "
            "les interactions restent visibles dans les cartes et scatter plots."
        ),
        ha="center",
        fontsize=9,
        color="#5e6b75",
    )
    figure.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.22, wspace=0.38)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def generate_analysis_outputs(
    raw_results: Any,
    aggregated: Any,
    experiment_root: Path | str,
    show: bool = False,
) -> dict[str, Path]:
    import pandas as pd

    if not show:
        os.environ.setdefault("MPLBACKEND", "Agg")
    root = Path(experiment_root).resolve()
    data_directory = root / "donnees" / "analyse"
    correlation_directory = root / "graphiques" / "correlations"
    distribution_directory = root / "graphiques" / "distributions"
    paths = {
        "analytical": data_directory / "configurations_analytiques.csv",
        "spearman": data_directory / "correlations_spearman.csv",
        "pearson": data_directory / "correlations_pearson.csv",
        "scatter_relationships": (
            correlation_directory / "scatter_accuracy_hyperparametres.png"
        ),
        "scatter_cost": correlation_directory / "scatter_cout_performance.png",
        "correlation_matrix": (
            correlation_directory / "matrice_correlation_spearman.png"
        ),
        "accuracy_histograms": (
            distribution_directory / "histogrammes_accuracy.png"
        ),
        "stability_histograms": (
            distribution_directory / "histogrammes_stabilite_duree.png"
        ),
        "catalog": data_directory / "catalogue_analyse.csv",
    }
    for directory in (data_directory, correlation_directory, distribution_directory):
        directory.mkdir(parents=True, exist_ok=True)

    analytical = build_analytical_table(raw_results, aggregated)
    spearman, pearson = correlation_tables(analytical)
    write_dataframe_atomic(analytical, paths["analytical"])
    write_dataframe_atomic(spearman, paths["spearman"])
    write_dataframe_atomic(pearson, paths["pearson"])

    renderers = (
        (
            "scatter_relationships",
            lambda: plot_accuracy_relationships(
                analytical, paths["scatter_relationships"], show
            ),
        ),
        (
            "scatter_cost",
            lambda: plot_cost_performance(analytical, paths["scatter_cost"], show),
        ),
        (
            "correlation_matrix",
            lambda: plot_correlation_matrix(
                analytical, paths["correlation_matrix"], show
            ),
        ),
        (
            "accuracy_histograms",
            lambda: plot_accuracy_histograms(
                raw_results, paths["accuracy_histograms"], show
            ),
        ),
        (
            "stability_histograms",
            lambda: plot_stability_histograms(
                analytical, paths["stability_histograms"], show
            ),
        ),
    )
    existed_before_render = {
        key: paths[key].exists()
        for key, _ in renderers
    }
    render_errors: dict[str, str] = {}
    for key, renderer in renderers:
        try:
            renderer()
        except Exception as exc:
            render_errors[key] = str(exc)

    descriptions = {
        "analytical": "Une ligne par configuration et champs dérivés pour l'analyse.",
        "spearman": "Corrélations de rang descriptives, pour un seul protocole et par activation.",
        "pearson": "Corrélations linéaires descriptives après transformations documentées.",
        "scatter_relationships": "Accuracy face aux trois hyperparamètres.",
        "scatter_cost": "Accuracy face aux paramètres et à la durée, avec Pareto.",
        "correlation_matrix": "Matrice de Spearman globale et par activation.",
        "accuracy_histograms": "Distribution des répétitions par activation et dropout.",
        "stability_histograms": "Distribution de l'écart-type et de la durée.",
    }
    catalog_rows = []
    for key, path in paths.items():
        if key == "catalog":
            continue
        if key in render_errors:
            was_stale = existed_before_render.get(key, False) and path.exists()
            status = "ancien_non_actualise" if was_stale else "echec_non_genere"
            message = render_errors[key]
            if was_stale:
                message = (
                    "Le PNG existait avant cette analyse et n'a pas été actualisé : "
                    + message
                )
            generated_at = ""
        else:
            status = "genere"
            message = ""
            generated_at = utc_now()
        catalog_rows.append(
            {
                "key": key,
                "path": str(path.relative_to(root)),
                "type": "figure PNG" if path.suffix == ".png" else "données CSV",
                "description": descriptions[key],
                "exists": path.exists(),
                "status": status,
                "message": message,
                "generated_at_utc": generated_at,
                "catalogued_at_utc": utc_now(),
            }
        )
    write_dataframe_atomic(pd.DataFrame(catalog_rows), paths["catalog"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Génère scatter plots, histogrammes et corrélations descriptives à "
            "partir des CSV d'une expérience, sans entraîner de modèle."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nom-experience",
        default="grille_dense",
        help="nom du dossier sous sorties/01_RECHERCHE_MODELE/01_HYPERPARAMETRES",
    )
    parser.add_argument(
        "--racine-experience",
        type=Path,
        default=None,
        help="chemin explicite de l'expérience ; remplace --nom-experience",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="ouvrir les figures après leur sauvegarde",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")
    if args.racine_experience is None:
        root = env_config.SORTIES_HYPERPARAMETRES / args.nom_experience
    else:
        root = args.racine_experience.expanduser()
        if not root.is_absolute():
            root = env_config.PROJECT_ROOT / root
    root = root.resolve()
    raw_path = root / "donnees" / "resultats_bruts.csv"
    aggregated_path = root / "donnees" / "resultats_agreges.csv"
    if not raw_path.is_file():
        parser.error(f"résultats bruts introuvables : {raw_path}")
    if not aggregated_path.is_file():
        parser.error(f"résultats agrégés introuvables : {aggregated_path}")
    paths = generate_analysis_outputs(
        pd.read_csv(raw_path),
        pd.read_csv(aggregated_path),
        root,
        show=args.show,
    )
    print(f"Table analytique : {paths['analytical']}")
    print(f"Corrélations : {paths['spearman']}")
    print(f"Scatter plots : {paths['scatter_relationships']}")
    print(f"Histogrammes : {paths['accuracy_histograms']}")
    print(f"Catalogue : {paths['catalog']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
