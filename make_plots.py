from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configs import (
    CLASS_NAMES,
    DEFAULT_DATA_ROOT,
    RESULTS_MODELS_DIR,
    RESULTS_PLOTS_DIR,
    RESULTS_TABLES_DIR,
    ensure_results_dirs,
)
from src.privacy_filters import apply_canny_edges, apply_center_crop, apply_diffusion_noise, apply_gaussian_blur, apply_mosaic


TRANSFORM_LABELS = {
    "none": "baseline_clean",
    "crop": "crop_context_removal",
    "blur": "blur",
    "mosaic": "mosaic",
    "edges": "canny",
    "noise": "noise",
}


def save_with_aliases(fig: plt.Figure, output_path: Path, aliases: list[Path] | None = None) -> Path:
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    for alias in aliases or []:
        fig.savefig(alias, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final result tables and plots from saved experiment metrics."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=RESULTS_MODELS_DIR,
        help="Directory containing *_metrics.json files.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=RESULTS_PLOTS_DIR,
        help="Directory where final plots are saved.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=RESULTS_TABLES_DIR,
        help="Directory where final CSV tables are saved.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root used for visual filter examples.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke/debug runs in final tables.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_smoke_run(payload: dict, path: Path) -> bool:
    run_name = str(payload.get("run_name", path.stem)).lower()
    return "smoke" in run_name or "smokecheck" in run_name


def extract_test_metrics(payload: dict) -> dict:
    if "test_metrics" in payload:
        return payload["test_metrics"]
    if "metrics" in payload:
        return payload["metrics"]
    return {}


def metric_row(payload: dict, path: Path) -> dict[str, object] | None:
    metrics = extract_test_metrics(payload)
    if not metrics:
        return None

    privacy_mode = str(payload.get("privacy_mode", "none"))
    history = payload.get("history", {})
    epochs = len(history.get("train", [])) if isinstance(history, dict) else None

    return {
        "run_name": payload.get("run_name", path.stem.replace("_metrics", "")),
        "model": payload.get("model", payload.get("model_name", "")),
        "weights": payload.get("weights", ""),
        "transformation": TRANSFORM_LABELS.get(privacy_mode, privacy_mode),
        "privacy_mode": privacy_mode,
        "privacy_intensity": float(payload.get("privacy_intensity", 0.0)),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy", metrics.get("recall_macro")),
        "precision": metrics.get("precision_macro"),
        "recall": metrics.get("recall_macro"),
        "macro_f1": metrics.get("f1_macro"),
        "weighted_f1": metrics.get("f1_weighted"),
        "loss": metrics.get("loss"),
        "best_epoch": payload.get("best_epoch"),
        "epochs": epochs,
        "batch_size": payload.get("batch_size"),
        "image_size": payload.get("image_size"),
        "num_workers": payload.get("num_workers"),
        "metrics_path": str(path),
        "checkpoint_path": payload.get("checkpoint_path", str(path).replace("_metrics.json", "_best.pt")),
    }


def build_results_table(metrics_dir: Path, include_smoke: bool) -> pd.DataFrame:
    rows = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        payload = read_json(path)
        if not include_smoke and is_smoke_run(payload, path):
            continue
        row = metric_row(payload, path)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["transformation", "model", "privacy_intensity", "macro_f1"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def save_tables(df: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    final_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "balanced_accuracy",
        "transformation",
        "privacy_mode",
        "privacy_intensity",
        "run_name",
        "best_epoch",
        "epochs",
        "batch_size",
        "image_size",
        "num_workers",
        "metrics_path",
        "checkpoint_path",
    ]
    final_results_path = tables_dir / "final_model_results.csv"
    df[final_columns].to_csv(final_results_path, index=False)
    outputs["final_results"] = final_results_path

    results_summary = create_report_results_summary(df)
    results_summary_path = tables_dir / "results_summary.csv"
    results_summary.to_csv(results_summary_path, index=False)
    outputs["results_summary"] = results_summary_path

    summary = (
        results_summary.groupby("transformation")
        .agg(
            runs=("run_name", "count"),
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    summary_path = tables_dir / "final_results_by_transformation.csv"
    summary.to_csv(summary_path, index=False)
    outputs["summary_by_transformation"] = summary_path

    best_rows = (
        results_summary.sort_values("macro_f1", ascending=False)
        .groupby(["transformation", "model"], as_index=False)
        .head(1)
        .sort_values(["transformation", "macro_f1"], ascending=[True, False])
    )
    best_path = tables_dir / "best_run_per_model_and_transformation.csv"
    best_rows[final_columns].to_csv(best_path, index=False)
    outputs["best_runs"] = best_path

    return outputs


def _empty_summary_row() -> dict[str, object]:
    return {
        "model": "",
        "transformation": "",
        "privacy_mode": "",
        "privacy_intensity": 0.0,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "clean_baseline_macro_f1": np.nan,
        "macro_f1_drop_vs_clean": np.nan,
        "loss": np.nan,
        "run_name": "",
        "best_epoch": np.nan,
        "epochs": np.nan,
        "batch_size": np.nan,
        "image_size": np.nan,
        "num_workers": np.nan,
        "metrics_path": "",
        "checkpoint_path": "",
        "summary_source": "",
    }


def _baseline_reference_from_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    references: dict[str, float] = {}
    for row in rows:
        if row["transformation"] == "baseline_clean":
            references[str(row["model"])] = float(row["macro_f1"])
    return references


def create_report_results_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    baseline_path = Path("results/baseline_comparison.csv")
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        for _, source in baseline_df.iterrows():
            row = _empty_summary_row()
            row.update(
                {
                    "model": source["model"],
                    "transformation": "baseline_clean",
                    "privacy_mode": "none",
                    "privacy_intensity": 0.0,
                    "accuracy": source["accuracy"],
                    "balanced_accuracy": source["recall_macro"],
                    "precision": source["precision_macro"],
                    "recall": source["recall_macro"],
                    "macro_f1": source["f1_macro"],
                    "weighted_f1": source["f1_weighted"],
                    "clean_baseline_macro_f1": source["f1_macro"],
                    "macro_f1_drop_vs_clean": 0.0,
                    "loss": source["loss"],
                    "run_name": source["run_name"],
                    "best_epoch": source["best_epoch"],
                    "epochs": source["epochs"],
                    "batch_size": source["batch_size"],
                    "image_size": source["image_size"],
                    "num_workers": source["num_workers"],
                    "metrics_path": source["metrics_path"],
                    "checkpoint_path": str(source["metrics_path"]).replace("_metrics.json", "_best.pt"),
                    "summary_source": "baseline_comparison.csv",
                }
            )
            rows.append(row)
    else:
        baseline_rows = (
            df[df["privacy_mode"] == "none"]
            .sort_values("macro_f1", ascending=False)
            .groupby("model", as_index=False)
            .head(1)
        )
        for _, source in baseline_rows.iterrows():
            row = source.to_dict()
            row["clean_baseline_macro_f1"] = row["macro_f1"]
            row["macro_f1_drop_vs_clean"] = 0.0
            row["summary_source"] = "metrics_dir_best_clean_fallback"
            rows.append(row)

    baseline_reference = _baseline_reference_from_rows(rows)

    deid_fixed_path = Path("results/deid_fixed_comparison.csv")
    if deid_fixed_path.exists():
        deid_df = pd.read_csv(deid_fixed_path)
        for _, source in deid_df.iterrows():
            row = _empty_summary_row()
            transformation = TRANSFORM_LABELS.get(source["privacy_mode"], source["privacy_mode"])
            clean_f1 = baseline_reference.get(str(source["model"]), source["original_f1_macro"])
            row.update(
                {
                    "model": source["model"],
                    "transformation": transformation,
                    "privacy_mode": source["privacy_mode"],
                    "privacy_intensity": source["privacy_intensity"],
                    "accuracy": source["test_accuracy"],
                    "balanced_accuracy": source["test_balanced_accuracy"],
                    "precision": source["test_precision_macro"],
                    "recall": source["test_recall_macro"],
                    "macro_f1": source["test_f1_macro"],
                    "weighted_f1": source["test_f1_weighted"],
                    "clean_baseline_macro_f1": clean_f1,
                    "macro_f1_drop_vs_clean": source["test_f1_macro"] - clean_f1,
                    "loss": source["test_loss"],
                    "run_name": source["run_name"],
                    "best_epoch": source["best_epoch"],
                    "epochs": source["epochs"],
                    "batch_size": source["batch_size"],
                    "image_size": 224,
                    "num_workers": source["num_workers"],
                    "metrics_path": source["metrics_path"],
                    "checkpoint_path": source["checkpoint_path"],
                    "summary_source": "deid_fixed_comparison.csv",
                }
            )
            rows.append(row)

    supplemental = df[
        (
            (df["privacy_mode"] == "edges")
            | (df["privacy_mode"] == "noise")
        )
        & ~df["run_name"].str.contains("smoke", case=False, na=False)
    ].copy()
    for _, source in supplemental.iterrows():
        clean_f1 = baseline_reference.get(str(source["model"]))
        if clean_f1 is None:
            continue
        row = source.to_dict()
        row["clean_baseline_macro_f1"] = clean_f1
        row["macro_f1_drop_vs_clean"] = row["macro_f1"] - clean_f1
        row["summary_source"] = "metrics_dir_edges_noise"
        rows.append(row)

    columns = list(_empty_summary_row().keys())
    summary = pd.DataFrame(rows)
    return summary[columns].sort_values(
        ["transformation", "model", "privacy_intensity", "macro_f1"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def create_results_summary(df: pd.DataFrame) -> pd.DataFrame:
    baseline_reference = (
        df[df["privacy_mode"] == "none"]
        .sort_values("macro_f1", ascending=False)
        .groupby("model", as_index=False)
        .head(1)[["model", "macro_f1"]]
        .rename(columns={"macro_f1": "clean_baseline_macro_f1"})
    )
    summary = df.merge(baseline_reference, on="model", how="left")
    summary["macro_f1_drop_vs_clean"] = (
        summary["macro_f1"] - summary["clean_baseline_macro_f1"]
    )

    columns = [
        "model",
        "transformation",
        "privacy_mode",
        "privacy_intensity",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "macro_f1",
        "weighted_f1",
        "clean_baseline_macro_f1",
        "macro_f1_drop_vs_clean",
        "loss",
        "run_name",
        "best_epoch",
        "epochs",
        "batch_size",
        "image_size",
        "num_workers",
        "metrics_path",
        "checkpoint_path",
    ]
    return summary[columns].sort_values(
        ["transformation", "model", "privacy_intensity", "macro_f1"],
        ascending=[True, True, True, False],
    )


def plot_baseline_metrics(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    baseline_df = df[df["privacy_mode"] == "none"].copy()
    if baseline_df.empty:
        return None

    best_baselines = (
        baseline_df.sort_values("macro_f1", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .sort_values("macro_f1", ascending=True)
    )
    y = np.arange(len(best_baselines))
    bar_height = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset, column, color, label in [
        (-bar_height, "precision", "#176B87", "Precision"),
        (0.0, "recall", "#C79A37", "Recall"),
        (bar_height, "macro_f1", "#3D7D5A", "Macro-F1"),
    ]:
        ax.barh(y + offset, best_baselines[column], height=bar_height, color=color, label=label)
        for index, value in enumerate(best_baselines[column]):
            ax.text(value + 0.003, index + offset, f"{value:.3f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(best_baselines["model"])
    ax.set_xlim(0.90, 0.97)
    ax.set_xlabel("Score")
    ax.set_title("Best Clean Baseline Metrics by Model")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = plots_dir / "precision_recall_f1_by_model.png"
    return save_with_aliases(fig, output_path)


def plot_macro_f1_drop_by_transformation(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    if df.empty:
        return None

    if "macro_f1_drop_vs_clean" in df.columns:
        summary = df.copy()
    else:
        summary = create_results_summary(df)
    best_per_model_transform = (
        summary.sort_values("macro_f1", ascending=False)
        .groupby(["model", "transformation"], as_index=False)
        .head(1)
    )
    drop_summary = (
        best_per_model_transform[best_per_model_transform["transformation"] != "baseline_clean"]
        .groupby("transformation")
        .agg(
            mean_macro_f1_drop=("macro_f1_drop_vs_clean", "mean"),
            models=("model", "nunique"),
        )
        .sort_values("mean_macro_f1_drop", ascending=True)
    )
    if drop_summary.empty:
        return None

    colors = [
        "#3D7D5A" if value >= -0.01 else "#C79A37" if value >= -0.05 else "#B94A48"
        for value in drop_summary["mean_macro_f1_drop"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(drop_summary.index, drop_summary["mean_macro_f1_drop"], color=colors)
    ax.axvline(0, color="#182027", linewidth=1)
    for index, value in enumerate(drop_summary["mean_macro_f1_drop"]):
        ax.text(
            value - 0.006 if value < 0 else value + 0.006,
            index,
            f"{value:+.3f}",
            va="center",
            ha="right" if value < 0 else "left",
            fontsize=9,
        )
    ax.set_xlabel("Macro-F1 drop vs best clean baseline of the same model")
    ax.set_title("Macro-F1 Drop by Privacy Transformation")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = plots_dir / "macro_f1_drop_by_transformation.png"
    return save_with_aliases(fig, output_path)


def plot_vit_confusion_from_predictions(plots_dir: Path) -> Path | None:
    predictions_path = Path("results/vit_attention_predictions.csv")
    if not predictions_path.exists():
        return None

    predictions = pd.read_csv(predictions_path)
    original_predictions = predictions[predictions["condition"] == "Original"]
    if original_predictions.empty:
        return None

    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.float64)
    for _, row in original_predictions.iterrows():
        matrix[int(row["true_index"]), int(row["pred_index"])] += 1

    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    image = ax.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("ViT-B/16 Confusion Matrix by Emotion")
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            value = normalized[row, col]
            if row == col or value >= 0.03:
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#182027",
                    fontsize=8.5,
                    fontweight="bold" if row == col else "normal",
                )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = plots_dir / "confusion_matrix_by_emotion.png"
    return save_with_aliases(fig, output_path)


def plot_recall_by_emotion_from_predictions(plots_dir: Path) -> Path | None:
    predictions_path = Path("results/vit_attention_predictions.csv")
    if not predictions_path.exists():
        return None

    predictions = pd.read_csv(predictions_path)
    original_predictions = predictions[predictions["condition"] == "Original"]
    if original_predictions.empty:
        return None

    recall = (
        original_predictions.groupby("true_class")["correct"]
        .mean()
        .reindex(CLASS_NAMES)
    )
    recall = recall.dropna()
    if recall.empty:
        return None

    recall = recall.sort_values(ascending=True)
    colors = ["#B94A48" if value < 0.94 else "#3D7D5A" for value in recall]

    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    ax.barh(recall.index, recall.values, color=colors)
    for index, value in enumerate(recall.values):
        ax.text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=10)
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("Recall per emotion")
    ax.set_title("Recall by Emotion for ViT-B/16 on Original Test Images")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = plots_dir / "recall_by_emotion.png"
    return save_with_aliases(fig, output_path)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def find_example_images(data_root: Path, max_rows: int = 4) -> list[tuple[str, Path]]:
    preferred_classes = ["surprise", "happy", "angry", "neutral"]
    examples: list[tuple[str, Path]] = []
    for class_name in preferred_classes:
        class_dir = data_root / "test" / class_name
        if not class_dir.exists():
            continue
        candidates = sorted(class_dir.glob("*.png"))
        if candidates:
            examples.append((class_name, candidates[0]))
        if len(examples) >= max_rows:
            break
    return examples


def plot_filter_examples(data_root: Path, plots_dir: Path) -> Path | None:
    examples = find_example_images(data_root)
    if not examples:
        return None

    transforms = [
        ("Original", lambda image: image),
        ("Crop/context", lambda image: apply_center_crop(image, 0.75)),
        ("Blur", lambda image: apply_gaussian_blur(image, 3.0)),
        ("Mosaic", lambda image: apply_mosaic(image, 8.0)),
        ("Canny", lambda image: apply_canny_edges(image)),
        ("Noise", lambda image: apply_diffusion_noise(image, 100.0)),
    ]

    fig, axes = plt.subplots(len(examples), len(transforms), figsize=(13, 2.4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, (class_name, image_path) in enumerate(examples):
        image = read_gray(image_path)
        for col_index, (title, transform) in enumerate(transforms):
            axes[row_index, col_index].imshow(transform(image), cmap="gray", vmin=0, vmax=255)
            axes[row_index, col_index].axis("off")
            if row_index == 0:
                axes[row_index, col_index].set_title(title, fontsize=11, fontweight="bold")
            if col_index == 0:
                axes[row_index, col_index].text(
                    -0.08,
                    0.5,
                    class_name.title(),
                    transform=axes[row_index, col_index].transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

    fig.suptitle("Privacy Filter Examples", fontsize=15, fontweight="bold")
    fig.tight_layout()

    output_path = plots_dir / "transformation_examples.png"
    return save_with_aliases(fig, output_path)


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    df = build_results_table(args.metrics_dir, include_smoke=args.include_smoke)
    if df.empty:
        raise RuntimeError(f"No metrics found in {args.metrics_dir}")

    table_outputs = save_tables(df, args.tables_dir)
    summary_df = pd.read_csv(args.tables_dir / "results_summary.csv")
    plot_outputs = [
        plot_baseline_metrics(summary_df, args.plots_dir),
        plot_macro_f1_drop_by_transformation(summary_df, args.plots_dir),
        plot_vit_confusion_from_predictions(args.plots_dir),
        plot_recall_by_emotion_from_predictions(args.plots_dir),
        plot_filter_examples(args.data_root, args.plots_dir),
    ]

    print("Final tables:")
    for label, path in table_outputs.items():
        print(f"- {label}: {path}")

    print("Final plots:")
    for path in plot_outputs:
        if path is not None:
            print(f"- {path}")


if __name__ == "__main__":
    main()
