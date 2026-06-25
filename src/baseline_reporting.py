from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .configs import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    RESULTS_TABLES_DIR,
    BaselineExperimentConfig,
    build_train_command,
    resolve_python_bin,
)


BASELINE_GROUP = "clean_baseline"
LIGHT_FINETUNE_GROUP = "light_finetune"


def get_baseline_configs() -> list[BaselineExperimentConfig]:
    return [
        BaselineExperimentConfig.cnn_baseline_config(
            model="resnet18",
            run_suffix="cnn_baseline",
        ),
        BaselineExperimentConfig.cnn_baseline_config(
            model="mobilenet_v3_large",
            run_suffix="cnn_baseline",
        ),
        BaselineExperimentConfig.swin_baseline_config(
            run_suffix="nw4",
        ),
        BaselineExperimentConfig.vit_baseline_config(
            run_suffix="vit_baseline",
        ),
    ]


def get_light_finetune_configs() -> list[BaselineExperimentConfig]:
    return [
        BaselineExperimentConfig.light_finetune_config("resnet18"),
        BaselineExperimentConfig.light_finetune_config("swin_t"),
        BaselineExperimentConfig.light_finetune_config("vit_b_16"),
    ]


def get_selected_configs(
    run_clean_baselines: bool = True,
    run_light_finetuning: bool = False,
) -> list[BaselineExperimentConfig]:
    selected_configs: list[BaselineExperimentConfig] = []
    if run_clean_baselines:
        selected_configs.extend(get_baseline_configs())
    if run_light_finetuning:
        selected_configs.extend(get_light_finetune_configs())
    return selected_configs


def build_experiment_plan() -> pd.DataFrame:
    rows = []
    for group_name, configs in [
        (BASELINE_GROUP, get_baseline_configs()),
        (LIGHT_FINETUNE_GROUP, get_light_finetune_configs()),
    ]:
        for config in configs:
            rows.append(
                {
                    "group": group_name,
                    "run_name": config.run_name,
                    "model": config.model,
                    "privacy_mode": config.privacy_mode,
                    "privacy_intensity": config.privacy_intensity,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "label_smoothing": config.label_smoothing,
                    "num_workers": config.num_workers,
                    "pin_memory": config.pin_memory,
                    "metrics_exists": config.metrics_path.exists(),
                    "checkpoint_exists": config.checkpoint_path.exists(),
                    "metrics_path": str(config.metrics_path),
                }
            )
    return pd.DataFrame(rows)


def build_command_table(
    configs: Iterable[BaselineExperimentConfig],
    train_script: Path | None = None,
    python_bin: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    skip_completed: bool = True,
) -> pd.DataFrame:
    train_script = train_script or PROJECT_ROOT / "train.py"
    python_bin = python_bin or resolve_python_bin()

    rows = []
    for config in configs:
        command = build_train_command(
            config=config,
            train_script=train_script,
            python_bin=python_bin,
            data_root=data_root,
        )
        rows.append(
            {
                "run_name": config.run_name,
                "model": config.model,
                "metrics_exists": config.metrics_path.exists(),
                "checkpoint_exists": config.checkpoint_path.exists(),
                "will_skip_if_completed": bool(skip_completed and config.metrics_path.exists()),
                "command": subprocess.list2cmdline(command),
            }
        )
    return pd.DataFrame(rows)


def stream_training(command: list[str], cwd: Path = PROJECT_ROOT) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    return process.wait()


def run_selected_experiments(
    configs: Iterable[BaselineExperimentConfig],
    train_script: Path | None = None,
    python_bin: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    skip_completed: bool = True,
) -> None:
    train_script = train_script or PROJECT_ROOT / "train.py"
    python_bin = python_bin or resolve_python_bin()
    configs = list(configs)

    if not configs:
        print("No runs selected.")
        return

    start_time = time.time()
    for index, config in enumerate(configs, start=1):
        if skip_completed and config.metrics_path.exists():
            print(f"Skipping completed run {index}/{len(configs)}: {config.run_name}")
            continue

        command = build_train_command(
            config=config,
            train_script=train_script,
            python_bin=python_bin,
            data_root=data_root,
        )
        print(f"\nStarting run {index}/{len(configs)}: {config.run_name}")
        print(subprocess.list2cmdline(command))

        return_code = stream_training(command)
        if return_code != 0:
            raise RuntimeError(
                f"Training failed for {config.run_name} with return code {return_code}."
            )

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nSelected runs finished in {elapsed_minutes:.2f} minutes.")


def load_metrics(config: BaselineExperimentConfig, group_name: str) -> dict[str, object] | None:
    if not config.metrics_path.exists():
        return None

    payload = json.loads(config.metrics_path.read_text(encoding="utf-8"))
    test_metrics = payload.get("test_metrics", {})
    row: dict[str, object] = {
        "group": group_name,
        "run_name": payload.get("run_name", config.run_name),
        "model": payload.get("model", config.model),
        "weights": payload.get("weights", config.weights),
        "privacy_mode": payload.get("privacy_mode", config.privacy_mode),
        "privacy_intensity": payload.get("privacy_intensity", config.privacy_intensity),
        "best_epoch": payload.get("best_epoch"),
        "epochs": config.epochs,
        "batch_size": payload.get("batch_size", config.batch_size),
        "image_size": payload.get("image_size", config.image_size),
        "learning_rate": payload.get("learning_rate", config.learning_rate),
        "weight_decay": payload.get("weight_decay", config.weight_decay),
        "label_smoothing": payload.get("label_smoothing", config.label_smoothing),
        "num_workers": payload.get("num_workers", config.num_workers),
        "metrics_path": str(config.metrics_path),
    }
    row.update(test_metrics)
    return row


def load_completed_metrics() -> pd.DataFrame:
    rows = []
    for group_name, configs in [
        (BASELINE_GROUP, get_baseline_configs()),
        (LIGHT_FINETUNE_GROUP, get_light_finetune_configs()),
    ]:
        for config in configs:
            row = load_metrics(config, group_name)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["group", "f1_macro", "accuracy"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def load_results_summary(
    path: Path = RESULTS_TABLES_DIR / "results_summary.csv",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Results summary not found: {path}. Run make_plots.py first."
        )
    return pd.read_csv(path)


def clean_baseline_summary(results_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = results_summary[results_summary["transformation"].eq("baseline_clean")].copy()
    if baseline.empty:
        return baseline
    return baseline.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)


def plot_clean_baseline_scores(baseline_summary: pd.DataFrame) -> plt.Figure:
    if baseline_summary.empty:
        raise ValueError("No clean baseline rows available to plot.")

    plot_df = baseline_summary.sort_values("macro_f1", ascending=True)
    metrics = [
        ("accuracy", "Accuracy", "#2f4858"),
        ("precision", "Macro precision", "#f6ae2d"),
        ("recall", "Macro recall", "#86bbd8"),
        ("macro_f1", "Macro F1", "#4f772d"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    y_positions = range(len(plot_df))
    bar_height = 0.18
    offsets = [-1.5 * bar_height, -0.5 * bar_height, 0.5 * bar_height, 1.5 * bar_height]

    for offset, (metric, label, color) in zip(offsets, metrics):
        values = plot_df[metric]
        ax.barh(
            [position + offset for position in y_positions],
            values,
            height=bar_height,
            label=label,
            color=color,
        )
        for position, value in zip(y_positions, values):
            ax.text(value + 0.002, position + offset, f"{value:.3f}", va="center", fontsize=8)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlim(max(0.0, plot_df["macro_f1"].min() - 0.04), 1.0)
    ax.set_xlabel("Test score")
    ax.set_title("Clean Baseline Metrics")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_precision_recall_tradeoff(baseline_summary: pd.DataFrame) -> plt.Figure:
    if baseline_summary.empty:
        raise ValueError("No clean baseline rows available to plot.")

    plot_df = baseline_summary.sort_values("macro_f1", ascending=False)
    x_positions = range(len(plot_df))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        [position - bar_width / 2 for position in x_positions],
        plot_df["precision"],
        width=bar_width,
        label="Macro precision",
        color="#2f4858",
    )
    ax.bar(
        [position + bar_width / 2 for position in x_positions],
        plot_df["recall"],
        width=bar_width,
        label="Macro recall",
        color="#f6ae2d",
    )

    for position, precision, recall in zip(x_positions, plot_df["precision"], plot_df["recall"]):
        ax.text(position - bar_width / 2, precision + 0.001, f"{precision:.3f}", ha="center", fontsize=8)
        ax.text(position + bar_width / 2, recall + 0.001, f"{recall:.3f}", ha="center", fontsize=8)

    ax.set_title("Clean Baseline Precision vs Recall")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(plot_df["model"], rotation=20, ha="right")
    ax.set_ylim(max(0.0, min(plot_df["precision"].min(), plot_df["recall"].min()) - 0.04), 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_baseline_vs_light_finetuning(metrics: pd.DataFrame) -> plt.Figure:
    if metrics.empty:
        raise ValueError("No completed metrics available to plot.")

    plot_df = metrics.copy()
    plot_df["label"] = plot_df["model"] + "\n" + plot_df["group"]
    colors = plot_df["group"].map(
        {BASELINE_GROUP: "#2f4858", LIGHT_FINETUNE_GROUP: "#4f772d"}
    ).fillna("#355070")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, metric, title in [
        (axes[0], "accuracy", "Test Accuracy"),
        (axes[1], "f1_macro", "Test Macro F1"),
        (axes[2], "loss", "Test Loss"),
    ]:
        axis.bar(plot_df["label"], plot_df[metric], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylim(max(0.0, plot_df["accuracy"].min() - 0.05), 1.0)
    axes[1].set_ylim(max(0.0, plot_df["f1_macro"].min() - 0.05), 1.0)
    axes[2].set_ylim(0.0, plot_df["loss"].max() * 1.15)

    fig.suptitle("Baselines and Light Fine-tuning", fontsize=15)
    fig.tight_layout()
    return fig
