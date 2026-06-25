from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from ..configs import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    TRAIN_SCRIPT,
    RESULTS_FINAL_TABLES_DIR,
    RESULTS_INTERMEDIATE_TABLES_DIR,
    BaselineExperimentConfig,
    build_train_command,
    resolve_python_bin,
)
from .filters import apply_center_crop, apply_gaussian_blur, apply_mosaic


FIXED_FILTERS = [
    {
        "mode": "blur",
        "intensity": 3.0,
        "label": "Gaussian blur",
        "justification": "sigma=3 gives an effective kernel around 7 px, roughly 10% of a 75 px face.",
    },
    {
        "mode": "mosaic",
        "intensity": 8.0,
        "label": "Mosaic",
        "justification": "8 px blocks are about 10% of a 75 px face width.",
    },
    {
        "mode": "crop",
        "intensity": 0.75,
        "label": "Crop/context removal",
        "justification": "center crop keeps 75% of the image and removes external facial context.",
    },
]

MODELS = ("resnet18", "swin_t", "vit_b_16")
FIXED_DEID_EPOCHS = 10
FIXED_DEID_RUN_SUFFIX = "fixed_deid"
FIXED_DEID_CSV_PATH = RESULTS_INTERMEDIATE_TABLES_DIR / "deid_fixed_comparison.csv"


def fixed_filter_table() -> pd.DataFrame:
    return pd.DataFrame(FIXED_FILTERS)


def get_baseline_configs() -> dict[str, BaselineExperimentConfig]:
    return {
        "resnet18": BaselineExperimentConfig.cnn_baseline_config(
            model="resnet18",
            run_suffix="cnn_baseline",
        ),
        "swin_t": BaselineExperimentConfig.swin_baseline_config(
            run_suffix="nw4",
        ),
        "vit_b_16": BaselineExperimentConfig.vit_baseline_config(
            run_suffix="vit_baseline",
        ),
    }


def get_deid_configs() -> list[BaselineExperimentConfig]:
    return [
        BaselineExperimentConfig.deid_experiment_config(
            model=model,
            privacy_mode=filter_config["mode"],
            privacy_intensity=filter_config["intensity"],
            epochs=FIXED_DEID_EPOCHS,
            run_suffix=FIXED_DEID_RUN_SUFFIX,
        )
        for model in MODELS
        for filter_config in FIXED_FILTERS
    ]


def get_best_val_metrics(payload: dict) -> dict:
    history = payload.get("history", {})
    val_history = history.get("val", []) if isinstance(history, dict) else []
    best_epoch = payload.get("best_epoch")

    if best_epoch is not None and 1 <= int(best_epoch) <= len(val_history):
        return val_history[int(best_epoch) - 1]

    if val_history:
        return max(val_history, key=lambda row: row.get("f1_macro", -1.0))

    return {}


def load_metrics_payload(config: BaselineExperimentConfig) -> dict | None:
    if not config.metrics_path.exists():
        return None
    return json.loads(config.metrics_path.read_text(encoding="utf-8"))


def with_balanced_accuracy(metrics: dict) -> dict:
    metrics = dict(metrics)
    if "balanced_accuracy" not in metrics and "recall_macro" in metrics:
        metrics["balanced_accuracy"] = metrics["recall_macro"]
    return metrics


def load_metrics_row(
    config: BaselineExperimentConfig,
    group_name: str,
) -> dict[str, object] | None:
    payload = load_metrics_payload(config)
    if payload is None:
        return None

    val_metrics = with_balanced_accuracy(get_best_val_metrics(payload))
    test_metrics = with_balanced_accuracy(payload.get("test_metrics", {}))
    row: dict[str, object] = {
        "group": group_name,
        "run_name": payload.get("run_name", config.run_name),
        "model": payload.get("model", config.model),
        "privacy_mode": payload.get("privacy_mode", config.privacy_mode),
        "privacy_intensity": payload.get("privacy_intensity", config.privacy_intensity),
        "best_epoch": payload.get("best_epoch"),
        "epochs": config.epochs,
        "batch_size": payload.get("batch_size", config.batch_size),
        "num_workers": payload.get("num_workers", config.num_workers),
        "metrics_path": str(config.metrics_path),
        "checkpoint_path": str(config.checkpoint_path),
    }

    for name, value in val_metrics.items():
        row[f"val_{name}"] = value
    for name, value in test_metrics.items():
        row[f"test_{name}"] = value

    return row


def build_config_table(configs: Iterable[BaselineExperimentConfig]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_name": config.run_name,
                "model": config.model,
                "privacy_mode": config.privacy_mode,
                "privacy_intensity": config.privacy_intensity,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "metrics_exists": config.metrics_path.exists(),
                "checkpoint_exists": config.checkpoint_path.exists(),
            }
            for config in configs
        ]
    )


def build_command_table(
    configs: Iterable[BaselineExperimentConfig],
    train_script: Path | None = None,
    python_bin: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    skip_completed: bool = True,
) -> pd.DataFrame:
    train_script = train_script or TRAIN_SCRIPT
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
                "privacy_mode": config.privacy_mode,
                "privacy_intensity": config.privacy_intensity,
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


def run_configs(
    configs: Iterable[BaselineExperimentConfig],
    stage_name: str = "fixed de-id",
    train_script: Path | None = None,
    python_bin: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    skip_completed: bool = True,
) -> None:
    train_script = train_script or TRAIN_SCRIPT
    python_bin = python_bin or resolve_python_bin()
    configs = list(configs)

    if not configs:
        print(f"No configs selected for {stage_name}.")
        return

    start_time = time.time()
    for index, config in enumerate(configs, start=1):
        if skip_completed and config.metrics_path.exists():
            print(f"Skipping completed {stage_name} run {index}/{len(configs)}: {config.run_name}")
            continue

        command = build_train_command(
            config=config,
            train_script=train_script,
            python_bin=python_bin,
            data_root=data_root,
        )

        print(f"\nStarting {stage_name} run {index}/{len(configs)}: {config.run_name}")
        print(subprocess.list2cmdline(command))
        return_code = stream_training(command)
        if return_code != 0:
            raise RuntimeError(
                f"Training failed for {config.run_name} with return code {return_code}."
            )

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n{stage_name} finished in {elapsed_minutes:.2f} minutes.")


def load_original_baselines() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    for config in get_baseline_configs().values():
        row = load_metrics_row(config, "original")
        if row is None:
            missing.append(config.run_name)
        else:
            rows.append(row)

    return pd.DataFrame(rows), missing


def load_fixed_deid_metrics() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    for config in get_deid_configs():
        row = load_metrics_row(config, "deid")
        if row is None:
            missing.append(config.run_name)
        else:
            rows.append(row)

    return pd.DataFrame(rows), missing


def build_fixed_deid_comparison(
    baseline_df: pd.DataFrame,
    deid_df: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_df.empty or deid_df.empty:
        return pd.DataFrame()

    baseline_reference = baseline_df[
        [
            "model",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1_macro",
            "test_precision_macro",
            "test_recall_macro",
            "test_loss",
        ]
    ].rename(
        columns={
            "test_accuracy": "original_accuracy",
            "test_balanced_accuracy": "original_balanced_accuracy",
            "test_f1_macro": "original_f1_macro",
            "test_precision_macro": "original_precision_macro",
            "test_recall_macro": "original_recall_macro",
            "test_loss": "original_loss",
        }
    )

    comparison = deid_df.merge(baseline_reference, on="model", how="left")
    comparison["condition"] = comparison["privacy_mode"].map(
        {filter_config["mode"]: filter_config["label"] for filter_config in FIXED_FILTERS}
    )
    comparison["delta_accuracy"] = comparison["test_accuracy"] - comparison["original_accuracy"]
    comparison["delta_balanced_accuracy"] = (
        comparison["test_balanced_accuracy"] - comparison["original_balanced_accuracy"]
    )
    comparison["delta_f1_macro"] = comparison["test_f1_macro"] - comparison["original_f1_macro"]
    comparison["delta_precision_macro"] = (
        comparison["test_precision_macro"] - comparison["original_precision_macro"]
    )
    comparison["delta_recall_macro"] = (
        comparison["test_recall_macro"] - comparison["original_recall_macro"]
    )
    comparison["delta_loss"] = comparison["test_loss"] - comparison["original_loss"]
    return comparison.sort_values(["model", "privacy_mode"]).reset_index(drop=True)


def save_fixed_deid_comparison(
    comparison: pd.DataFrame,
    path: Path = FIXED_DEID_CSV_PATH,
) -> Path | None:
    if comparison.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)
    return path


def find_preview_image(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    for pattern in ("*.jpg", "*.png", "*.jpeg"):
        sample_paths = sorted((data_root / "test").rglob(pattern))
        if sample_paths:
            return sample_paths[0]
    raise FileNotFoundError(f"No preview image found under {data_root / 'test'}")


def apply_fixed_filter(image, mode: str, intensity: float):
    if mode == "blur":
        return apply_gaussian_blur(image, intensity)
    if mode == "mosaic":
        return apply_mosaic(image, intensity)
    if mode == "crop":
        return apply_center_crop(image, intensity)
    raise ValueError(f"Unsupported fixed filter mode: {mode}")


def plot_fixed_filter_preview(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[plt.Figure, Path]:
    sample_image_path = find_preview_image(data_root)
    sample_image = cv2.imread(str(sample_image_path), cv2.IMREAD_GRAYSCALE)
    if sample_image is None:
        raise FileNotFoundError(f"Could not read sample image: {sample_image_path}")

    preview_images = [("Original", sample_image)]
    for filter_config in FIXED_FILTERS:
        filtered = apply_fixed_filter(
            sample_image,
            mode=str(filter_config["mode"]),
            intensity=float(filter_config["intensity"]),
        )
        preview_images.append(
            (f"{filter_config['label']} ({filter_config['intensity']})", filtered)
        )

    fig, axes = plt.subplots(1, len(preview_images), figsize=(15, 4))
    for axis, (title, image) in zip(axes, preview_images):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(f"Fixed de-id preview: {sample_image_path.name}")
    fig.tight_layout()
    return fig, sample_image_path


def plot_fixed_deid_impact(
    comparison: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> plt.Figure:
    if comparison.empty:
        raise ValueError("No fixed de-id comparison available to plot.")
    if baseline_df.empty:
        raise ValueError("No original baseline metrics available to plot.")

    metric_columns = [
        ("test_accuracy", "Accuracy"),
        ("test_balanced_accuracy", "Balanced Accuracy"),
        ("test_f1_macro", "Macro F1"),
    ]
    filter_order = [str(filter_config["mode"]) for filter_config in FIXED_FILTERS]

    fig, axes = plt.subplots(1, len(metric_columns), figsize=(18, 5), sharey=True)
    for axis, (metric, title) in zip(axes, metric_columns):
        for model in MODELS:
            model_df = (
                comparison[comparison["model"] == model]
                .set_index("privacy_mode")
                .reindex(filter_order)
            )
            baseline_value = baseline_df[baseline_df["model"] == model][metric]
            if not baseline_value.empty:
                axis.axhline(baseline_value.iloc[0], linestyle="--", linewidth=1, alpha=0.35)
            axis.plot(filter_order, model_df[metric], marker="o", linewidth=2, label=model)

        axis.set_title(title)
        axis.set_xlabel("Condition")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Fixed De-id Impact Across Models", fontsize=16)
    fig.tight_layout()
    return fig


def plot_macro_f1_drop(comparison: pd.DataFrame) -> plt.Figure:
    if comparison.empty:
        raise ValueError("No fixed de-id comparison available to plot.")

    plot_df = comparison.copy()
    plot_df["label"] = plot_df["model"] + "\n" + plot_df["condition"]
    plot_df = plot_df.sort_values("delta_f1_macro")
    colors = plot_df["privacy_mode"].map(
        {"crop": "#4f772d", "blur": "#f6ae2d", "mosaic": "#b23a48"}
    ).fillna("#355070")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["label"], plot_df["delta_f1_macro"], color=colors)
    ax.axvline(0.0, color="#222222", linewidth=1)
    for index, value in enumerate(plot_df["delta_f1_macro"]):
        ax.text(
            value - 0.005 if value < 0 else value + 0.005,
            index,
            f"{value:+.3f}",
            va="center",
            ha="right" if value < 0 else "left",
            fontsize=8.5,
        )
    ax.set_title("Macro F1 Drop Under Fixed De-identification")
    ax.set_xlabel("Macro F1 delta vs original baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def load_results_summary(path: Path = RESULTS_FINAL_TABLES_DIR / "results_summary.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing results summary: {path}. Run scripts/reporting/make_plots.py first."
        )
    return pd.read_csv(path)
