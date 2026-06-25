from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .analysis import (
    AttentionCondition,
    apply_attention_mask,
    apply_condition_filter,
    attention_cosine_similarity,
    attention_entropy,
    collect_condition_predictions,
    load_checkpoint_model,
    overlay_attention,
    predict_image,
    read_grayscale_image,
    vit_attention_map,
)
from ..configs import (
    CLASS_NAMES,
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    RESULTS_ATTENTION_PLOTS_DIR,
    RESULTS_ATTENTION_TABLES_DIR,
    BaselineExperimentConfig,
)
from .landmarks import (
    REGION_ORDER,
    compute_landmark_attention_metrics,
    create_face_landmarker,
    is_mediapipe_available,
)
from ..privacy.reporting import (
    FIXED_DEID_EPOCHS,
    FIXED_DEID_RUN_SUFFIX,
    build_fixed_deid_comparison,
    load_fixed_deid_metrics,
    load_original_baselines,
)


ATTENTION_FILTERS = [
    {"label": "Crop/context removal", "mode": "crop", "intensity": 0.75},
    {"label": "Blur", "mode": "blur", "intensity": 3.0},
    {"label": "Mosaic", "mode": "mosaic", "intensity": 8.0},
]

ATTENTION_DIR = RESULTS_ATTENTION_PLOTS_DIR / "vit_attention_analysis"
PREDICTIONS_CSV_PATH = RESULTS_ATTENTION_TABLES_DIR / "vit_attention_predictions.csv"
LANDMARKER_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


@dataclass(frozen=True)
class AttentionOutputPaths:
    example_set_name: str
    attention_dir: Path
    predictions_csv: Path
    selected_examples_csv: Path
    attention_metrics_csv: Path
    attention_metrics_plot: Path
    inverse_attention_csv: Path
    inverse_attention_dir: Path
    landmark_attention_csv: Path
    landmark_attention_plot: Path

    @classmethod
    def for_example_set(cls, example_set_name: str = "class_balanced") -> "AttentionOutputPaths":
        slug = slugify(example_set_name)
        attention_dir = ATTENTION_DIR / slug
        return cls(
            example_set_name=slug,
            attention_dir=attention_dir,
            predictions_csv=PREDICTIONS_CSV_PATH,
            selected_examples_csv=RESULTS_ATTENTION_TABLES_DIR / f"vit_attention_selected_examples_{slug}.csv",
            attention_metrics_csv=RESULTS_ATTENTION_TABLES_DIR / f"vit_attention_metrics_{slug}.csv",
            attention_metrics_plot=attention_dir / "vit_attention_entropy_similarity.png",
            inverse_attention_csv=RESULTS_ATTENTION_TABLES_DIR / f"vit_inverse_attention_masking_{slug}.csv",
            inverse_attention_dir=attention_dir / "inverse_attention_masking",
            landmark_attention_csv=RESULTS_ATTENTION_TABLES_DIR / f"vit_landmark_region_attention_{slug}.csv",
            landmark_attention_plot=attention_dir / "vit_landmark_region_attention.png",
        )


def ensure_attention_dirs(paths: AttentionOutputPaths) -> None:
    paths.attention_dir.mkdir(parents=True, exist_ok=True)
    paths.inverse_attention_dir.mkdir(parents=True, exist_ok=True)
    paths.selected_examples_csv.parent.mkdir(parents=True, exist_ok=True)


def slugify(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative_path(path_value: object) -> str:
    path = resolve_project_path(path_value)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


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


def get_vit_conditions(
    fixed_deid_epochs: int = FIXED_DEID_EPOCHS,
    run_suffix: str = FIXED_DEID_RUN_SUFFIX,
) -> list[AttentionCondition]:
    conditions = [
        AttentionCondition(
            label="Original",
            mode="none",
            intensity=0.0,
            config=get_baseline_configs()["vit_b_16"],
        )
    ]

    for filter_config in ATTENTION_FILTERS:
        conditions.append(
            AttentionCondition(
                label=str(filter_config["label"]),
                mode=str(filter_config["mode"]),
                intensity=float(filter_config["intensity"]),
                config=BaselineExperimentConfig.deid_experiment_config(
                    model="vit_b_16",
                    privacy_mode=str(filter_config["mode"]),
                    privacy_intensity=float(filter_config["intensity"]),
                    epochs=fixed_deid_epochs,
                    run_suffix=run_suffix,
                ),
            )
        )
    return conditions


def condition_labels(conditions: Iterable[AttentionCondition]) -> list[str]:
    return [condition.label for condition in conditions]


def missing_checkpoint_labels(conditions: Iterable[AttentionCondition]) -> list[str]:
    return [
        condition.label
        for condition in conditions
        if not condition.config.checkpoint_path.exists()
    ]


def fixed_filter_table() -> pd.DataFrame:
    return pd.DataFrame(ATTENTION_FILTERS)


def load_context_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    baseline_df, missing_baselines = load_original_baselines()
    deid_df, missing_deid = load_fixed_deid_metrics()
    comparison_df = build_fixed_deid_comparison(baseline_df, deid_df)
    return baseline_df, comparison_df, missing_baselines, missing_deid


def vit_deid_context(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()
    return comparison_df[comparison_df["model"].eq("vit_b_16")].copy()


def load_or_collect_predictions(
    conditions: list[AttentionCondition],
    paths: AttentionOutputPaths,
    recompute: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = 64,
    num_workers: int = 0,
    max_samples: int | None = None,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, str]:
    ensure_attention_dirs(paths)
    missing = missing_checkpoint_labels(conditions)
    if missing:
        return pd.DataFrame(), f"Missing ViT checkpoints: {missing}"

    if paths.predictions_csv.exists() and not recompute:
        return pd.read_csv(paths.predictions_csv), f"Loaded cached predictions: {paths.predictions_csv}"

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    for condition in conditions:
        print(f"Collecting predictions for: {condition.label}")
        frames.append(
            collect_condition_predictions(
                condition=condition,
                data_root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                max_samples=max_samples,
                device=device,
            )
        )

    predictions = pd.concat(frames, ignore_index=True)
    predictions["image_path"] = predictions["image_path"].map(project_relative_path)
    predictions.to_csv(paths.predictions_csv, index=False)
    return predictions, f"Saved predictions: {paths.predictions_csv}"


def prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return (
        predictions.groupby("condition")
        .agg(
            n=("image_path", "count"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            classes=("true_class", "nunique"),
        )
        .reset_index()
    )


def select_class_balanced_examples(
    predictions: pd.DataFrame,
    source_condition: str = "Original",
    correct_per_class: int = 1,
    error_count: int = 2,
    class_order: Iterable[str] = CLASS_NAMES,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    source_rows = predictions[predictions["condition"].eq(source_condition)].copy()
    if source_rows.empty:
        return pd.DataFrame()

    selected_rows: list[dict[str, object]] = []
    used_paths: set[str] = set()

    for class_name in class_order:
        class_rows = source_rows[source_rows["true_class"].eq(class_name)]
        correct_rows = class_rows[class_rows["correct"]]
        candidates = correct_rows if not correct_rows.empty else class_rows
        for _, row in candidates.sort_values("image_path").head(correct_per_class).iterrows():
            row_dict = row.to_dict()
            row_dict["case_type"] = "correct" if bool(row_dict["correct"]) else "fallback"
            row_dict["source_condition"] = source_condition
            selected_rows.append(row_dict)
            used_paths.add(str(row_dict["image_path"]))

    error_rows = source_rows[
        (~source_rows["correct"]) & (~source_rows["image_path"].astype(str).isin(used_paths))
    ].copy()
    for class_name in class_order:
        if len([row for row in selected_rows if row.get("case_type") == "error"]) >= error_count:
            break
        class_errors = error_rows[error_rows["true_class"].eq(class_name)]
        if class_errors.empty:
            continue
        row_dict = class_errors.sort_values("image_path").iloc[0].to_dict()
        row_dict["case_type"] = "error"
        row_dict["source_condition"] = source_condition
        selected_rows.append(row_dict)
        used_paths.add(str(row_dict["image_path"]))

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if not selected.empty:
        selected["image_path"] = selected["image_path"].map(project_relative_path)
    return selected


def load_or_select_examples(
    predictions: pd.DataFrame,
    paths: AttentionOutputPaths,
    reselect: bool = True,
    source_condition: str = "Original",
    correct_per_class: int = 1,
    error_count: int = 2,
) -> tuple[pd.DataFrame, str]:
    ensure_attention_dirs(paths)
    if paths.selected_examples_csv.exists() and not reselect:
        return (
            pd.read_csv(paths.selected_examples_csv),
            f"Loaded cached selected examples: {paths.selected_examples_csv}",
        )

    selected = select_class_balanced_examples(
        predictions=predictions,
        source_condition=source_condition,
        correct_per_class=correct_per_class,
        error_count=error_count,
    )
    if selected.empty:
        return selected, "No selected examples available."

    selected.to_csv(paths.selected_examples_csv, index=False)
    return selected, f"Saved selected examples: {paths.selected_examples_csv}"


def selected_examples_summary(selected_examples: pd.DataFrame) -> pd.DataFrame:
    if selected_examples.empty:
        return pd.DataFrame()
    return (
        selected_examples.groupby(["case_type", "true_class"])
        .size()
        .reset_index(name="n")
        .sort_values(["case_type", "true_class"])
    )


def _clear_existing_images(directory: Path, pattern: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob(pattern):
        path.unlink()


def _load_models_by_condition(
    conditions: list[AttentionCondition],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    return {
        condition.label: load_checkpoint_model(condition.config, device=device)
        for condition in conditions
    }


def load_or_create_attention_maps(
    selected_examples: pd.DataFrame,
    conditions: list[AttentionCondition],
    paths: AttentionOutputPaths,
    recompute: bool = False,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, list[Path], str]:
    ensure_attention_dirs(paths)
    cached_images = sorted(paths.attention_dir.glob("example_*.png"))
    if paths.attention_metrics_csv.exists() and cached_images and not recompute:
        return (
            pd.read_csv(paths.attention_metrics_csv),
            cached_images,
            f"Loaded cached attention maps: {paths.attention_dir}",
        )

    missing = missing_checkpoint_labels(conditions)
    if missing:
        return pd.DataFrame(), [], f"Missing ViT checkpoints: {missing}"
    if selected_examples.empty:
        return pd.DataFrame(), [], "No selected examples available."

    _clear_existing_images(paths.attention_dir, "example_*.png")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_by_condition = _load_models_by_condition(conditions, device=device)

    records: list[dict[str, object]] = []
    saved_paths: list[Path] = []

    for local_index, (_, example) in enumerate(selected_examples.iterrows()):
        image_path = resolve_project_path(example["image_path"])
        base_image = read_grayscale_image(image_path)
        attentions = {}
        rendered = {}
        predictions = {}

        for condition in conditions:
            filtered_image = apply_condition_filter(
                base_image,
                mode=condition.mode,
                intensity=condition.intensity,
            )
            attention_map, prediction = vit_attention_map(
                models_by_condition[condition.label],
                filtered_image,
            )
            attentions[condition.label] = attention_map
            rendered[condition.label] = {
                "input": filtered_image,
                "overlay": overlay_attention(filtered_image, attention_map),
            }
            predictions[condition.label] = CLASS_NAMES[prediction]

        original_attention = attentions["Original"]
        fig, axes = plt.subplots(2, len(conditions), figsize=(4 * len(conditions), 8))

        for col_index, condition in enumerate(conditions):
            label = condition.label
            entropy = attention_entropy(attentions[label])
            similarity = attention_cosine_similarity(attentions[label], original_attention)

            records.append(
                {
                    "example_index": local_index,
                    "source_condition": example["source_condition"],
                    "case_type": example["case_type"],
                    "true_class": example["true_class"],
                    "source_pred_class": example["pred_class"],
                    "condition": label,
                    "condition_prediction": predictions[label],
                    "attention_entropy": entropy,
                    "similarity_to_original": similarity,
                    "image_path": project_relative_path(image_path),
                    "file_name": example["file_name"],
                }
            )

            axes[0, col_index].imshow(rendered[label]["input"], cmap="gray")
            axes[0, col_index].set_title(f"{label}\ninput")
            axes[0, col_index].axis("off")
            axes[1, col_index].imshow(rendered[label]["overlay"])
            axes[1, col_index].set_title(
                f"attention\npred: {predictions[label]}\n"
                f"H={entropy:.2f} sim={similarity:.2f}"
            )
            axes[1, col_index].axis("off")

        fig.suptitle(
            f"{example['case_type']} example | true: {example['true_class']} | "
            f"file: {example['file_name']}",
            fontsize=14,
        )
        fig.tight_layout()

        output_path = paths.attention_dir / (
            f"example_{local_index:02d}_"
            f"{slugify(example['case_type'])}_"
            f"{slugify(example['true_class'])}.png"
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    metrics = pd.DataFrame(records)
    metrics.to_csv(paths.attention_metrics_csv, index=False)
    return metrics, saved_paths, f"Saved attention maps: {paths.attention_dir}"


def load_or_create_inverse_attention_results(
    selected_examples: pd.DataFrame,
    conditions: list[AttentionCondition],
    paths: AttentionOutputPaths,
    recompute: bool = False,
    mask_fraction: float = 0.2,
    max_examples: int = 5,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, list[Path], str]:
    ensure_attention_dirs(paths)
    cached_images = sorted(paths.inverse_attention_dir.glob("inverse_attention_*.png"))
    if paths.inverse_attention_csv.exists() and cached_images and not recompute:
        return (
            pd.read_csv(paths.inverse_attention_csv),
            cached_images,
            f"Loaded cached inverse-attention results: {paths.inverse_attention_csv}",
        )

    missing = missing_checkpoint_labels(conditions)
    if missing:
        return pd.DataFrame(), [], f"Missing ViT checkpoints: {missing}"
    if selected_examples.empty:
        return pd.DataFrame(), [], "No selected examples available."

    _clear_existing_images(paths.inverse_attention_dir, "inverse_attention_*.png")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_condition = next(condition for condition in conditions if condition.label == "Original")
    original_model = load_checkpoint_model(original_condition.config, device=device)

    examples = selected_examples.head(max_examples)
    records: list[dict[str, object]] = []
    saved_paths: list[Path] = []

    for local_index, (_, example) in enumerate(examples.iterrows()):
        image_path = resolve_project_path(example["image_path"])
        base_image = read_grayscale_image(image_path)
        filtered_image = apply_condition_filter(
            base_image,
            mode=original_condition.mode,
            intensity=original_condition.intensity,
        )
        attention_map, _ = vit_attention_map(original_model, filtered_image)
        baseline_pred, baseline_confidence, baseline_probs = predict_image(
            original_model,
            filtered_image,
        )
        true_index = int(example["true_index"])
        baseline_true_confidence = float(baseline_probs[true_index])

        masked_images = {
            "high_attention": apply_attention_mask(
                filtered_image,
                attention_map,
                mask_fraction=mask_fraction,
                region="high",
            ),
            "low_attention": apply_attention_mask(
                filtered_image,
                attention_map,
                mask_fraction=mask_fraction,
                region="low",
            ),
        }
        masked_predictions = {}

        for masked_region, masked_image in masked_images.items():
            masked_pred, masked_confidence, masked_probs = predict_image(
                original_model,
                masked_image,
            )
            masked_predictions[masked_region] = {
                "pred": masked_pred,
                "confidence": masked_confidence,
            }
            records.append(
                {
                    "example_index": local_index,
                    "case_type": example["case_type"],
                    "true_class": example["true_class"],
                    "image_path": project_relative_path(image_path),
                    "file_name": example["file_name"],
                    "mask_fraction": mask_fraction,
                    "masked_region": masked_region,
                    "baseline_pred_class": CLASS_NAMES[baseline_pred],
                    "masked_pred_class": CLASS_NAMES[masked_pred],
                    "prediction_changed": masked_pred != baseline_pred,
                    "baseline_pred_confidence": baseline_confidence,
                    "masked_pred_confidence": masked_confidence,
                    "pred_confidence_drop": masked_confidence - baseline_confidence,
                    "baseline_true_class_confidence": baseline_true_confidence,
                    "masked_true_class_confidence": float(masked_probs[true_index]),
                    "true_class_confidence_drop": float(masked_probs[true_index])
                    - baseline_true_confidence,
                }
            )

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        axes[0].imshow(filtered_image, cmap="gray")
        axes[0].set_title(
            f"Original\npred: {CLASS_NAMES[baseline_pred]}\n"
            f"conf={baseline_confidence:.2f}"
        )
        axes[0].axis("off")
        axes[1].imshow(overlay_attention(filtered_image, attention_map))
        axes[1].set_title("ViT attention")
        axes[1].axis("off")

        for axis, masked_region, title in [
            (axes[2], "high_attention", "Mask high attention"),
            (axes[3], "low_attention", "Mask low attention"),
        ]:
            result = masked_predictions[masked_region]
            axis.imshow(masked_images[masked_region], cmap="gray")
            axis.set_title(
                f"{title}\npred: {CLASS_NAMES[result['pred']]}\n"
                f"conf={result['confidence']:.2f} "
                f"drop={result['confidence'] - baseline_confidence:+.2f}"
            )
            axis.axis("off")

        fig.suptitle(
            f"Inverse-attention masking | true: {example['true_class']} | "
            f"case: {example['case_type']} | file: {example['file_name']}",
            fontsize=13,
        )
        fig.tight_layout()
        output_path = paths.inverse_attention_dir / (
            f"inverse_attention_{local_index:02d}_"
            f"{slugify(example['case_type'])}_{slugify(example['true_class'])}.png"
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    inverse = pd.DataFrame(records)
    inverse.to_csv(paths.inverse_attention_csv, index=False)
    return inverse, saved_paths, f"Saved inverse-attention results: {paths.inverse_attention_csv}"


def inverse_attention_summary(inverse_attention: pd.DataFrame) -> pd.DataFrame:
    if inverse_attention.empty:
        return pd.DataFrame()
    return (
        inverse_attention.groupby("masked_region")
        .agg(
            n=("image_path", "count"),
            prediction_change_rate=("prediction_changed", "mean"),
            mean_pred_confidence_drop=("pred_confidence_drop", "mean"),
            mean_true_class_confidence_drop=("true_class_confidence_drop", "mean"),
        )
        .reset_index()
    )


def load_or_create_landmark_attention(
    selected_examples: pd.DataFrame,
    conditions: list[AttentionCondition],
    paths: AttentionOutputPaths,
    recompute: bool = False,
    landmarker_model_path: Path = LANDMARKER_MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, Path | None, str]:
    ensure_attention_dirs(paths)
    if (
        paths.landmark_attention_csv.exists()
        and paths.landmark_attention_plot.exists()
        and not recompute
    ):
        return (
            pd.read_csv(paths.landmark_attention_csv),
            paths.landmark_attention_plot,
            f"Loaded cached landmark attention: {paths.landmark_attention_csv}",
        )

    if not is_mediapipe_available():
        return pd.DataFrame(), None, "MediaPipe is not installed."
    if not landmarker_model_path.exists():
        return pd.DataFrame(), None, f"Face Landmarker model not found: {landmarker_model_path}"

    missing = missing_checkpoint_labels(conditions)
    if missing:
        return pd.DataFrame(), None, f"Missing ViT checkpoints: {missing}"
    if selected_examples.empty:
        return pd.DataFrame(), None, "No selected examples available."

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_by_condition = _load_models_by_condition(conditions, device=device)

    records: list[dict[str, object]] = []
    with create_face_landmarker(landmarker_model_path) as landmarker:
        for example_index, example in selected_examples.iterrows():
            image_path = resolve_project_path(example["image_path"])
            base_image = read_grayscale_image(image_path)

            for condition in conditions:
                filtered_image = apply_condition_filter(
                    base_image,
                    mode=condition.mode,
                    intensity=condition.intensity,
                )
                attention_map, prediction = vit_attention_map(
                    models_by_condition[condition.label],
                    filtered_image,
                )
                result = compute_landmark_attention_metrics(
                    landmarker=landmarker,
                    image=filtered_image,
                    attention_map=attention_map,
                )

                row = {
                    "example_index": int(example_index),
                    "source_condition": example["source_condition"],
                    "case_type": example["case_type"],
                    "true_class": example["true_class"],
                    "condition": condition.label,
                    "condition_prediction": CLASS_NAMES[prediction],
                    "landmarks_detected": result.landmarks_detected,
                    "num_landmarks": result.num_landmarks,
                    "image_path": project_relative_path(image_path),
                    "file_name": example["file_name"],
                }
                row.update(result.metrics)
                records.append(row)

    landmark_attention = pd.DataFrame(records)
    landmark_attention.to_csv(paths.landmark_attention_csv, index=False)
    figure = plot_landmark_attention(landmark_attention, condition_labels(conditions))
    if figure is not None:
        figure.savefig(paths.landmark_attention_plot, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return (
        landmark_attention,
        paths.landmark_attention_plot if paths.landmark_attention_plot.exists() else None,
        f"Saved landmark attention: {paths.landmark_attention_csv}",
    )


def landmark_detection_summary(
    landmark_attention: pd.DataFrame,
    labels: Iterable[str],
) -> pd.DataFrame:
    if landmark_attention.empty:
        return pd.DataFrame()
    return (
        landmark_attention.groupby("condition")
        .agg(
            n=("image_path", "count"),
            detection_rate=("landmarks_detected", "mean"),
            mean_landmarks=("num_landmarks", "mean"),
        )
        .reindex(list(labels))
        .reset_index()
    )


def landmark_region_summary(
    landmark_attention: pd.DataFrame,
    labels: Iterable[str],
) -> pd.DataFrame:
    if landmark_attention.empty:
        return pd.DataFrame()

    mass_columns = [f"{region}_attention_mass" for region in REGION_ORDER]
    detected = landmark_attention[landmark_attention["landmarks_detected"]].copy()
    if detected.empty:
        return pd.DataFrame()
    return detected.groupby("condition")[mass_columns].mean().reindex(list(labels))


def plot_landmark_attention(
    landmark_attention: pd.DataFrame,
    labels: Iterable[str],
) -> plt.Figure | None:
    summary = landmark_region_summary(landmark_attention, labels)
    if summary.empty:
        return None

    pretty_names = {
        "eyes_attention_mass": "Eyes",
        "eyebrows_attention_mass": "Eyebrows",
        "nose_attention_mass": "Nose",
        "mouth_attention_mass": "Mouth",
        "face_other_attention_mass": "Other face",
        "outside_face_attention_mass": "Outside face",
    }
    plot_df = summary.rename(columns=pretty_names)
    colors = ["#2f4858", "#f6ae2d", "#86bbd8", "#d1495b", "#8f8f8f", "#d9d9d9"]

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(
        kind="bar",
        stacked=True,
        color=colors,
        width=0.72,
        ax=ax,
    )
    ax.set_title("ViT Attention Mass by MediaPipe Facial Region")
    ax.set_ylabel("Mean attention mass")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def attention_metrics_summary(attention_metrics: pd.DataFrame) -> pd.DataFrame:
    if attention_metrics.empty:
        return pd.DataFrame()
    return (
        attention_metrics.groupby(["condition", "case_type"])
        .agg(
            n=("image_path", "count"),
            entropy_mean=("attention_entropy", "mean"),
            entropy_std=("attention_entropy", "std"),
            similarity_mean=("similarity_to_original", "mean"),
            similarity_std=("similarity_to_original", "std"),
        )
        .reset_index()
    )


def plot_attention_metric_summary(
    attention_metrics: pd.DataFrame,
    labels: Iterable[str],
    output_path: Path | None = None,
) -> plt.Figure | None:
    if attention_metrics.empty:
        return None

    plot_df = (
        attention_metrics.groupby("condition")
        .agg(
            entropy=("attention_entropy", "mean"),
            similarity=("similarity_to_original", "mean"),
        )
        .reindex(list(labels))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(plot_df["condition"], plot_df["entropy"], color="#2f4858")
    axes[0].set_title("Mean Attention Entropy")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(plot_df["condition"], plot_df["similarity"], color="#f6ae2d")
    axes[1].set_title("Mean Similarity To Original")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("ViT Attention Metrics Under De-identification", fontsize=15)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
